from collections import deque
import random

import numpy as np
from numpy import log, exp, sqrt
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from src.util import (
    generate_gbm_paths,
    generate_heston_paths,
    get_mc_price
)

# --- Action ---
class Action:
    HOLD = 0
    EXER = 1
    NUM_ACTIONS = 2

# --- Replay Buffer ---
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

# --- Q Network v1 ---
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

# --- Q Network v2 ---
# class DuelingQNet(nn.Module):
#     def __init__(self, obssize, actsize, hidden_dim, depth):
#         super().__init__()
#         self.feature_layer = nn.Sequential(
#             nn.Linear(obssize, hidden_dim),
#             nn.ReLU()
#         )
#         # Build the advantage and value streams
#         self.advantage_layer = nn.Sequential(
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, actsize)
#         )
#         self.value_layer = nn.Sequential(
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, 1)
#         )

#     def forward(self, x):
#         features = self.feature_layer(x)
#         advantages = self.advantage_layer(features)
#         values = self.value_layer(features)
#         q_values = values + (advantages - advantages.mean())
#         return q_values

# --- Environment ---
class Environment:
    def __init__(self, nsim, nstep, t1, t2, s_0, r, q, path_kwargs, h, k, gbm):
        self.rng = np.random.default_rng()

        path_func = generate_gbm_paths if gbm else generate_heston_paths
        
        self.prices = path_func(
            rng=self.rng, nsim=nsim, nstep=nstep, t1=t1, t2=t2, s_0=s_0, 
            r=r, q=q, **path_kwargs
        )

        self.nsim = nsim
        self.nstep = nstep
        self.t1 = t1
        self.t2 = t2
        self.dt = (t2 - t1) / nstep

        self.s_0 = s_0
        self.r = r
        self.q = q
        self.h = h
        self.k = k

    def random_action(self):
        return self.rng.choice(range(0, Action.NUM_ACTIONS))

    def _get_obs(self):
        # ratio = self.S / self.K
        # delta_ratio = ratio - self.prev_ratio
        # self.prev_ratio = ratio
        # momentum = self._compute_momentum()
        # expected_payoff = self._compute_expected_future_payoff()
        # obs = np.array([self.S, self.t, ratio, delta_ratio, momentum, expected_payoff], dtype=np.float32)
        # # Normalize the observation
        # obs_mean = np.mean(obs)
        # obs_std = np.std(obs) + 1e-5  # Add epsilon to prevent division by zero
        # normalized_obs = (obs - obs_mean) / obs_std
        obs = [
            self.s, self.t
        ]
        return obs

    def reset(self):
        self.curr_sim = (self.curr_sim + 1) % self.nsim if hasattr(self, 'curr_sim') else 0
        self.curr_step = 0

        self.done = False

        self.s = self.prices[self.curr_sim, self.curr_step]
        self.t = self.t2 - self.t1 - self.dt

        # self.prev_ratio = self.S / self.K
        return self._get_obs()

    def step(self, action):
        if self.done: raise RuntimeError('Step called on a finished episode in step().')

        # intrinsic_value = self._intrinsic_value(self.S)
        
        # reward = disc_factor * intrinsic_value if action == 1 else 0

        if action == Action.EXER or self.curr_step == self.nstep - 1:
            disc = exp(-self.r * (self.curr_step + 1) * self.dt)
            
            reward = disc * self.h(self.s, self.k)
            
            self.done = True
        
        else:
            # action == ACTION.HOLD
            reward = 0

            self.curr_step += 1
            
            self.s = self.prices[self.curr_sim, self.curr_step]
            self.t -= self.dt

        return self._get_obs(), reward, self.done

    # def _compute_momentum(self):
    #     if self.current_step > 0:
    #         prev_price = self.price_paths[self.curr_path, self.current_step - 1]
    #         return self.S - prev_price
    #     return 0.0

    # def _compute_expected_future_payoff(self):
    #     remaining_prices = self.price_paths[self.curr_path, self.current_step:]
    #     payoffs = np.maximum(remaining_prices - self.K, 0) if self.option_type == "call" else np.maximum(self.K - remaining_prices, 0)
    #     return np.mean(payoffs) if len(payoffs) > 0 else 0.0

    # def _intrinsic_value(self, S):
    #     return max(S - self.K, 0) if self.option_type == "call" else max(self.K - S, 0)

# --- Agent ---
class Agent:
    def __init__(self, env, obssize, actsize, hidden_dim, depth, lr, buffer_size, batch_size, update_freq, gamma, eps, eps_decay, eps_min):
        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available()  else \
            'mps' if torch.backends.mps.is_available() else 'cpu'
        )

        self.env = env

        self.principal = QNet(obssize=obssize, actsize=actsize, hidden_dim=hidden_dim, depth=depth).to(self.device)
        self.target = QNet(obssize=obssize, actsize=actsize, hidden_dim=hidden_dim, depth=depth).to(self.device)

        self.optimizer = optim.Adam(self.principal.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

        self.buffer = ReplayBuffer(buffer_size=buffer_size, batch_size=batch_size)

        self.update_freq = update_freq
        self.gamma = gamma
        self.eps = eps
        self.eps_decay = eps_decay
        self.eps_min = eps_min

        self.initialize_buffer()
        self.update_params()

    def initialize_buffer(self):
        obs = self.env.reset()
        for _ in range(self.buffer.batch_size):
            action = self.env.random_action()
            newobs, reward, done = self.env.step(action=action)
            self.buffer.add(
                state=obs, action=action, reward=reward, next_state=newobs, done=done
            )

            obs = self.env.reset() if done else newobs

    def update_params(self):
        self.target.load_state_dict(self.principal.state_dict())

    # def act(self, state):
    #     if random.random() < self.eps:
    #         return random.randint(0, 1)
    #     state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
    #     with torch.no_grad():
    #         return torch.argmax(self.principal(state_tensor)).item()

    # def learn(self):
    #     if len(self.buffer) < self.buffer.batch_size:
    #         return
    #     states, actions, rewards, next_states, dones = self.buffer.sample()
    #     states, next_states = states.to(self.device), next_states.to(self.device)
    #     actions, rewards, dones = actions.to(self.device), rewards.to(self.device), dones.to(self.device)

    #     # Current Q values
    #     q_values = self.principal(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
    #     # Double DQN target calculation
    #     with torch.no_grad():
    #         # Action selection is done using the principal network
    #         next_actions = self.principal(next_states).argmax(1).unsqueeze(1)
    #         # Q-values are evaluated using the target network
    #         next_q_values = self.target(next_states).gather(1, next_actions).squeeze(1)
    #         targets = rewards + self.gamma * next_q_values * (1 - dones)

    #     # Compute loss
    #     loss = nn.MSELoss()(q_values, targets)
    #     self.optimizer.zero_grad()
    #     loss.backward()
    #     self.optimizer.step()

if __name__ == '__main__':
    pass
