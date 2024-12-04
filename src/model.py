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
        ratio = self.s / self.k
        # delta_ratio = ratio - self.prev_ratio
        # self.prev_ratio = ratio
        momentum = self._compute_momentum()
        # expected_payoff = self._compute_expected_future_payoff()
        # obs = np.array([self.S, self.t, ratio, delta_ratio, momentum, expected_payoff], dtype=np.float32)
        # # Normalize the observation
        # obs_mean = np.mean(obs)
        # obs_std = np.std(obs) + 1e-5  # Add epsilon to prevent division by zero
        # normalized_obs = (obs - obs_mean) / obs_std
        obs = [
            self.s, self.t, ratio, momentum
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

    def _compute_momentum(self):
        if self.curr_step > 0:
            prev_price = self.prices[self.curr_sim, self.curr_step - 1]
            return self.s - prev_price
        return 0.0

    # def _compute_expected_future_payoff(self):
    #     remaining_prices = self.price_paths[self.curr_path, self.current_step:]
    #     payoffs = np.maximum(remaining_prices - self.K, 0) if self.option_type == "call" else np.maximum(self.K - remaining_prices, 0)
    #     return np.mean(payoffs) if len(payoffs) > 0 else 0.0

    # def _intrinsic_value(self, S):
    #     return max(S - self.K, 0) if self.option_type == "call" else max(self.K - S, 0)

# --- Agent ---
class Agent:
    def __init__(self, env, obssize, actsize, hidden_dim, depth, lr, buffer_size, batch_size, buffer_interval, model_interval, gamma, eps, eps_decay, eps_min):
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

        self.buffer_interval = buffer_interval
        self.model_interval = model_interval

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

    def act(self, state):
        if self.env.rng.uniform(low=0, high=1) > self.eps:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            
            self.principal.eval()
            with torch.no_grad():
                action = torch.argmax(self.principal(state_tensor)).item()
        
        else:
            action = self.env.random_action()
        
        return action

    def learn(self):
        states, actions, rewards, next_states, dones = self.buffer.sample()
        
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        
        ### TARGET ###
        self.target.eval()
        with torch.no_grad():
            q_target = self.target(next_states)
            max_q, _ = torch.max(q_target, dim=-1)
            q_target = rewards + self.gamma * (1-dones) * max_q
        ### TARGET ###

        ### PRINCIPAL ###
        self.principal.train()

        self.optimizer.zero_grad()

        # getting Q-value of action taken in every sample of the batch
        q_values = self.principal(states).gather(1, actions.long().view(-1, 1)).view(-1)
        
        loss = self.criterion(q_values, q_target)
        loss.backward()
        self.optimizer.step()
        ### PRINCIPAL ###

        return loss.item()

    def train(self, nepisode, notebook, verbose=True, ma_window=100):
        if notebook: from tqdm.notebook import tqdm
        else:        from tqdm import tqdm

        totalstep = 0
        losses = np.zeros(nepisode)
        rewards = np.zeros(nepisode)

        for episode in tqdm(range(nepisode), desc='Episode', leave=True):
            obs = self.env.reset()
            done = False
            loss_sum = rew_sum = 0

            while not done:
                action = self.act(state=obs)
                newobs, reward, done = self.env.step(action=action)

                self.buffer.add(
                    state=obs, action=action, reward=reward, next_state=newobs, done=done
                )

                if totalstep % self.buffer_interval == 0:
                    loss_sum += self.learn()
                
                if totalstep % self.model_interval == 0:
                    self.update_params()
                
                totalstep += 1
                obs = newobs
                rew_sum += reward
            
            self.eps = max(self.eps * self.eps_decay, self.eps_min)
            
            losses[episode] = loss_sum
            rewards[episode] = rew_sum
        

            if verbose and episode % 50 == 0:
                print(f"Episode {episode + 1}/{nepisode}: Total Reward = {rew_sum:.2f}, Loss = {loss_sum:.4f}, Eps = {self.eps:.4f}")

        # Plot rewards with moving average
        plt.figure(figsize=(12, 6))
        plt.plot(rewards, label="Total Reward per Episode", alpha=0.4, color='blue')
        if len(rewards) >= ma_window:
            moving_avg_rewards = np.convolve(rewards, np.ones(ma_window)/ma_window, mode='valid')
            plt.plot(
                range(ma_window - 1, len(rewards)),
                moving_avg_rewards,
                label=f"Moving Avg Reward (window={ma_window})",
                color='red'
            )
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.title("Training Progress: Rewards")
        plt.legend()
        plt.grid(True)
        plt.show()

        # Plot losses with moving average
        plt.figure(figsize=(12, 6))
        plt.plot(losses, label="Total Loss per Episode", alpha=0.4, color='blue')
        if len(losses) >= ma_window:
            moving_avg_losses = np.convolve(losses, np.ones(ma_window)/ma_window, mode='valid')
            plt.plot(
                range(ma_window - 1, len(losses)),
                moving_avg_losses,
                label=f"Moving Avg Loss (window={ma_window})",
                color='red'
            )
        plt.xlabel("Episode")
        plt.ylabel("Total Loss")
        plt.title("Training Progress: Losses")
        plt.legend()
        plt.grid(True)
        plt.show()

        return losses, rewards

if __name__ == '__main__':
    pass
