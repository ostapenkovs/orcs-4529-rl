from collections import deque
import random
import torch
import torch.nn as nn
import torch.optim as optim
from gym import spaces
import numpy as np

# --- Replay Buffer ---
class ReplayBuffer:
    def __init__(self, buffer_size, batch_size):
        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size

    def add(self, state, action, reward, next_state, done):
        """Add an experience tuple to the buffer."""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self):
        """Sample a batch of experiences from the buffer."""
        batch = random.sample(self.buffer, self.batch_size)
        batch_np = np.array(batch, dtype=object)
        states, actions, rewards, next_states, dones = map(np.stack, zip(*batch_np))
        return (
            torch.tensor(states, dtype=torch.float32),
            torch.tensor(actions, dtype=torch.int64),
            torch.tensor(rewards, dtype=torch.float32),
            torch.tensor(next_states, dtype=torch.float32),
            torch.tensor(dones, dtype=torch.float32),
        )

    def __len__(self):
        return len(self.buffer)

# --- Q Network ---
class DuelingQNet(nn.Module):
    def __init__(self, obssize, actsize, hidden_dim, depth):
        super(DuelingQNet, self).__init__()
        self.feature_layer = nn.Sequential(
            nn.Linear(obssize, hidden_dim),
            nn.ReLU()
        )
        # Build the advantage and value streams
        self.advantage_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, actsize)
        )
        self.value_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        features = self.feature_layer(x)
        advantages = self.advantage_layer(features)
        values = self.value_layer(features)
        q_values = values + (advantages - advantages.mean())
        return q_values

# --- Environment ---
class Environment:
    def __init__(self, price_paths, s_0=100, K=100, t1=0, t2=1.0, r=0.01, option_type="call"):
        self.s_0 = s_0
        self.K = K
        self.t1 = t1
        self.t2 = t2
        self.r = r
        self.option_type = option_type
        self.price_paths = price_paths
        self.nsim, self.nstep = price_paths.shape
        self.dt = (t2 - t1) / self.nstep
        self.reset()

        # Observation space: [Stock price, Time to maturity, Ratio, Delta ratio, Momentum, Expected payoff]
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, -np.inf, -np.inf, -np.inf, -np.inf]),
            high=np.array([np.inf, t2 - t1, np.inf, np.inf, np.inf, np.inf]),
            dtype=np.float32
        )

        # Action space: 0 = Hold, 1 = Exercise
        self.action_space = spaces.Discrete(2)

    def reset(self):
        self.curr_path = (self.curr_path + 1) % self.nsim if hasattr(self, 'curr_path') else 0
        self.current_step = 0
        self.done = False
        self.S = self.price_paths[self.curr_path, self.current_step]
        self.t = self.t2 - self.t1
        self.prev_ratio = self.S / self.K
        return self._get_observation()

    def step(self, action):
        if self.done:
            raise RuntimeError("Step called on a finished episode.")

        intrinsic_value = self._intrinsic_value(self.S)
        disc_factor = np.exp(-self.r * self.current_step * self.dt)
        reward = disc_factor * intrinsic_value if action == 1 else 0

        if action == 1 or self.current_step == self.nstep - 1:
            self.done = True
        else:
            self.current_step += 1
            self.S = self.price_paths[self.curr_path, self.current_step]
            self.t -= self.dt

        return self._get_observation(), reward, self.done, {"intrinsic_value": intrinsic_value}

    def _get_observation(self):
        ratio = self.S / self.K
        delta_ratio = ratio - self.prev_ratio
        self.prev_ratio = ratio
        momentum = self._compute_momentum()
        expected_payoff = self._compute_expected_future_payoff()
        obs = np.array([self.S, self.t, ratio, delta_ratio, momentum, expected_payoff], dtype=np.float32)
        # Normalize the observation
        obs_mean = np.mean(obs)
        obs_std = np.std(obs) + 1e-5  # Add epsilon to prevent division by zero
        normalized_obs = (obs - obs_mean) / obs_std
        return normalized_obs

    def _compute_momentum(self):
        if self.current_step > 0:
            prev_price = self.price_paths[self.curr_path, self.current_step - 1]
            return self.S - prev_price
        return 0.0

    def _compute_expected_future_payoff(self):
        remaining_prices = self.price_paths[self.curr_path, self.current_step:]
        payoffs = np.maximum(remaining_prices - self.K, 0) if self.option_type == "call" else np.maximum(self.K - remaining_prices, 0)
        return np.mean(payoffs) if len(payoffs) > 0 else 0.0

    def _intrinsic_value(self, S):
        return max(S - self.K, 0) if self.option_type == "call" else max(self.K - S, 0)

# --- Agent ---
class Agent:
    def __init__(self, obssize, actsize, hidden_dim, depth, lr, buffer_size, batch_size, gamma, eps_start, eps_min, eps_decay):
        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available()  else \
            'mps' if torch.backends.mps.is_available() else 'cpu'
        )
        self.principal = DuelingQNet(obssize, actsize, hidden_dim, depth).to(self.device)
        self.target = DuelingQNet(obssize, actsize, hidden_dim, depth).to(self.device)
        self.optimizer = optim.Adam(self.principal.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_size, batch_size)
        self.gamma = gamma
        self.eps = eps_start
        self.eps_min = eps_min
        self.eps_decay = eps_decay
        self.update_params()

    def initialize_buffer(self, env, steps=1000):
        for _ in range(steps):
            state = env.reset()
            done = False
            while not done:
                action = env.action_space.sample()
                next_state, reward, done, _ = env.step(action)
                self.buffer.add(state, action, reward, next_state, done)
                state = next_state

    def update_params(self):
        self.target.load_state_dict(self.principal.state_dict())

    def act(self, state):
        if random.random() < self.eps:
            return random.randint(0, 1)
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return torch.argmax(self.principal(state_tensor)).item()

    def learn(self):
        if len(self.buffer) < self.buffer.batch_size:
            return
        states, actions, rewards, next_states, dones = self.buffer.sample()
        states, next_states = states.to(self.device), next_states.to(self.device)
        actions, rewards, dones = actions.to(self.device), rewards.to(self.device), dones.to(self.device)

        # Current Q values
        q_values = self.principal(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Double DQN target calculation
        with torch.no_grad():
            # Action selection is done using the principal network
            next_actions = self.principal(next_states).argmax(1).unsqueeze(1)
            # Q-values are evaluated using the target network
            next_q_values = self.target(next_states).gather(1, next_actions).squeeze(1)
            targets = rewards + self.gamma * next_q_values * (1 - dones)

        # Compute loss
        loss = nn.MSELoss()(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

if __name__ == '__main__':
    pass