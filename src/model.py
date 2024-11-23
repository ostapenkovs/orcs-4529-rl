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

class QNet(nn.Module):
    def __init__(self, obssize, actsize, hidden_dim, depth):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(obssize, hidden_dim), nn.ReLU()])

        for _ in range(depth):
            self.layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])

        self.layers.append(nn.Linear(hidden_dim, actsize))
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class Environment:
    def __init__(self, price_paths, s_0=100, K=100, t1 = 0, t2 = 1.0, r = 0.01,  **kwargs):
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


        # Observation space: stock price, time to maturity, and intrinsic value
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0]),
            high=np.array([np.inf, t2 - t1, np.inf]),
            dtype=np.float32
        )

        # Action space: 0 = Hold, 1 = Exercise
        self.action_space = spaces.Discrete(2)
        self.reset()

    def reset(self):
        self.curr_path += 1
        self.current_step = 0
        self.done = False
        self.S = self.price_paths[self.curr_path, self.current_step]  # Initial stock price
        self.t = self.t2 - self.t1  
        intrinsic_value = self.intrinsic_value(self.S)
        return np.array([self.S, self.t, intrinsic_value], dtype=np.float32)


    def step(self, action):
        if self.done:
            return

        intrinsic_value = self.intrinsic_value(self.S)

        reward = 0
        if action == 1: # Exercise
            reward = intrinsic_value
            self.done = True
        else:
            reward = 0 # Not sure what to do here...

        if not self.done:
            self.current_step += 1
            if self.current_step < self.nstep:
                self.S = self.price_paths[self.curr_path, self.current_step]  
                self.t -= self.dt  
            else:
                self.done = True  # End of path

        intrinsic_value = self.intrinsic_value(self.S)
        obs = np.array([self.S, self.t, intrinsic_value], dtype=np.float32)
        info = {"intrinsic_value": intrinsic_value}

        return obs, reward, self.done, info


    def intrinsic_value(self, S):
        return max(S - self.K, 0)

    # def render(self):
    #     print(f"Step: {self.current_step}, Stock Price: {self.S:.2f}, Time to Maturity: {self.t:.2f}")

    def close(self):
        pass


# TODO: include epsilon decay, update_params() interval
class Agent:
    def __init__(self, obssize, actsize, hidden_dim, depth, lr, buffer_size, batch_size, gamma, eps):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.principal = QNet(obsize=obssize, actsize=actsize, hidden_dim=hidden_dim, depth=depth).to(self.device)
        self.target = QNet(obsize=obssize, actsize=actsize, hidden_dim=hidden_dim, depth=depth).to(self.device)

        self.optimizer = optim.Adam(self.principal.parameters(), lr=lr)

        self.buffer = ReplayBuffer(buffer_size=buffer_size, batch_size=batch_size)

        self.initialize_buffer()
        self.update_params()
    
    def initialize_buffer(self):
        pass

    def update_params(self):
        self.target.load_state_dict(self.principal.state_dict())
    
    def act(self, state):
        pass

    def learn(self):
        pass

if __name__ == '__main__':
    pass
