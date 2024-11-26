from collections import deque
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from gym import Env, spaces
import numpy as np

class ReplayBuffer:
    def __init__(self, buffer_size, batch_size):
        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self):
        batch = random.sample(self.buffer, self.batch_size)
        return [torch.tensor(x, dtype=torch.float32) for x in zip(*batch)]

    def __len__(self):
        return len(self.buffer)

# class QNet(nn.Module):
#     def __init__(self, obssize, actsize, hidden_dim, depth):
#         super().__init__()
#         self.layers = nn.ModuleList([nn.Linear(obssize, hidden_dim), nn.ReLU()])

#         for _ in range(depth):
#             self.layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])

#         self.layers.append(nn.Linear(hidden_dim, actsize))
    
#     def forward(self, x):
#         for layer in self.layers:
#             x = layer(x)
#         return x

# Sequential more stable. We can try different things here

class QNet(nn.Module):
    def __init__(self, obssize, actsize, hidden_dim, depth):
        super().__init__()
        layers = [nn.Linear(obssize, hidden_dim), nn.ReLU()]
        for _ in range(depth):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
        layers.append(nn.Linear(hidden_dim, actsize))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class Environment:
    def __init__(self, price_paths, s_0=100, K=100, t1 = 0, t2 = 1.0, r = 0.01, option_type = "call",  **kwargs):
        """
        Environment for American options with early exercise.
        Args:
            price_paths: Generatted price paths using (generate_gbm_paths or generate_heston_paths).
            s_0: Initial stock price.
            K: Strike price.
            t1: Intitial time.
            t2: Final time.
            r: Risk-free rate.
            steps: Number of steps to maturity.
            path_params: Additional parameters for the price generator function.
        """
        # parameters
        self.curr_path = -1  # Track current path index
        self.current_step = 0  # Current step in the path
        self.done = False
        self.s_0 = s_0
        self.K = K
        self.t1 = t1
        self.t2 = t2
        self.r = r
        self.price_paths = price_paths
        self.nsim, self.nstep = price_paths.shape
        self.dt =  (t2 - t1) / self.nstep
        self.option_type = option_type

        self.prev_ratio = 0  # For momentum

        # Observation space: [Stock price, Time to maturity, Ratio, Delta ratio]
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, -np.inf, -np.inf]),
            high=np.array([np.inf, t2 - t1, np.inf, np.inf]),
            dtype=np.float32
        )


        # # Observation space: stock price, time to maturity, and intrinsic value (or ratio)
        # self.observation_space = spaces.Box(
        #     low=np.array([0.0, 0.0, 0.0]),
        #     high=np.array([np.inf, t2 - t1, np.inf]),
        #     dtype=np.float32
        # )

        # Action space: 0 = Hold, 1 = Exercise
        self.action_space = spaces.Discrete(2)
        self.reset()

    def reset(self):
        self.curr_path += 1
        if self.curr_path >= self.price_paths.shape[0]:
            self.curr_path = 0  # Loop back to the first path

        self.current_step = 0
        self.done = False
        self.S = self.price_paths[self.curr_path, self.current_step]  # Initial stock price
        self.t = self.t2 - self.t1
        intrinsic_value = self.intrinsic_value(self.S)
        ratio = self.S/self.K
        self.prev_ratio = ratio  # Initialize prev_ratio for delta_ratio calculation
        delta_ratio = 0.0  # No change at the start
        return np.array([self.S, self.t, ratio, delta_ratio], dtype=np.float32)


    def step(self, action):
        if self.done:
            return

        intrinsic_value = self.intrinsic_value(self.S)
        disc_factor = np.exp(-self.r * (self.current_step / self.nstep))

        reward = 0
        if action == 1: # Exercise
            reward = disc_factor * intrinsic_value
            self.done = True
        else:
            reward = 0
            #-0.01 * np.exp(self.r * (self.current_step / self.nstep))
            #reward = -0.01 # Not sure what to do here... trying to add some theta decay
            #WHAT IF: we price the vanilla equivalent and then do -price


        if not self.done:
            self.current_step += 1
            if self.current_step < self.nstep:
                self.S = self.price_paths[self.curr_path, self.current_step]
                self.t -= self.dt
            else:
                # At expiry, the payoff is intrinsic value
                reward = disc_factor * intrinsic_value

                # if intrinsic_value > 0:
                #     reward = disc_factor * intrinsic_value
                # else:
                #     reward = -0.01 * np.exp(self.r * (self.current_step / self.nstep))

                self.done = True


        #intrinsic_value = disc_factor * self.intrinsic_value(self.S)
        ratio = self.S/self.K
        delta_ratio = ratio - self.prev_ratio
        self.prev_ratio = ratio  # Update for next step

        # Observation
        #obs = np.array([self.S, self.t, ratio], dtype=np.float32)
        obs = np.array([self.S, self.t, ratio, delta_ratio], dtype=np.float32)
        info = {"intrinsic_value": intrinsic_value}

        return obs, reward, self.done, info


    def intrinsic_value(self, S):

        if self.option_type == "put":
            return max(self.K - S, 0)
        else:
            return max(S - self.K, 0)

    # def render(self):
    #     print(f"Step: {self.current_step}, Stock Price: {self.S:.2f}, Time to Maturity: {self.t:.2f}")

    def close(self):
        pass



# TODO: include epsilon decay, update_params() interval
class Agent:
    def __init__(self, obssize, actsize, hidden_dim, depth, lr, buffer_size, batch_size, gamma, eps):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.principal = QNet(obssize=obssize, actsize=actsize, hidden_dim=hidden_dim, depth=depth).to(self.device)
        self.target = QNet(obssize=obssize, actsize=actsize, hidden_dim=hidden_dim, depth=depth).to(self.device)

        self.optimizer = optim.Adam(self.principal.parameters(), lr=lr)

        self.buffer = ReplayBuffer(buffer_size=buffer_size, batch_size=batch_size)

        self.gamma = gamma
        self.eps = eps

        #self.initialize_buffer()
        self.update_params()
    
    def initialize_buffer(self, env, steps=1000):
        """
        Populate the replay buffer with random experiences.
        Args:
            env: The environment to interact with.
            steps: Number of random steps to populate the buffer.
        """
        for _ in range(steps):
                state = env.reset()  # Start a new episode
                done = False

                while not done:
                    action = env.action_space.sample()  # Random action
                    next_state, reward, done, _ = env.step(action)
                    self.buffer.add(state, action, reward, next_state, done)
                    state = next_state

                    # Debugging: Check if the final state is reached
                    if done:
                        print(f"Final state added to buffer: {next_state}, Reward: {reward}, Done: {done}")

        # for _ in range(steps):
        #     state = env.reset()
        #     done = False
        #     while not done:
        #         action = env.action_space.sample()  # Random action
        #         next_state, reward, done, _ = env.step(action)
        #         self.buffer.add(state, action, reward, next_state, done)
        #         state = next_state

    def update_params(self):
        self.target.load_state_dict(self.principal.state_dict())
    
    def act(self, state):
        """Select an action based on epsilon-greedy policy."""
        if random.random() < self.eps:
            return random.randint(0, 1)  # Random action: Hold or Exercise
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return torch.argmax(self.principal(state)).item()

    def learn(self):
        """Train the principal network on sampled experience."""
        if len(self.buffer) < self.buffer.batch_size:
            return

        states, actions, rewards, next_states, dones = self.buffer.sample()
        states = states.to(self.device)
        next_states = next_states.to(self.device)
        actions = actions.long().to(self.device)
        rewards = rewards.to(self.device)
        dones = dones.to(self.device)

        q_values = self.principal(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            max_next_q_values = self.target(next_states).max(1)[0]
            targets = rewards + self.gamma * max_next_q_values * (1 - dones)

        loss = nn.MSELoss()(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

if __name__ == '__main__':
    pass
