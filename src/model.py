from collections import deque
import random

import numpy as np
from numpy import log, exp, sqrt
import matplotlib as mpl
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from src.util import (
    generate_gbm_paths,
    generate_heston_paths
)

mpl.rcParams['figure.dpi'] = 100

class Action:
    HOLD        = 0
    EXER        = 1
    NUM_ACTIONS = 2

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
        layers = [nn.Linear(obssize, hidden_dim), nn.ReLU()]
        for _ in range(depth):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
        layers.append(nn.Linear(hidden_dim, actsize))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

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

        self.DIM_STATE = len(self.reset())

    def _compute_continuation_value(self):
        remaining_prices = self.prices[self.curr_sim, self.curr_step+1:]
        
        discounted_payoffs = [
            exp(-self.r * (t+1) * self.dt) * self.h(price, self.k) for t, price in enumerate(remaining_prices)
        ]
        
        return np.mean(discounted_payoffs) if discounted_payoffs else 0

    def _get_obs(self):
        obs = [
            (self.t2 - self.t) / (self.t2 - self.t1), 
            log(self.s/self.s_0),
            self._compute_continuation_value()
        ]
        return obs

    def random_action(self):
        return self.rng.choice(range(0, Action.NUM_ACTIONS))

    def reset(self):
        self.curr_sim = (self.curr_sim + 1) % self.nsim if hasattr(self, 'curr_sim') else 0
        self.curr_step = 0

        self.done = False

        self.s = self.prices[self.curr_sim, self.curr_step]
        self.t = self.t2 - self.t1 - self.dt

        return self._get_obs()

    def step(self, action):
        if self.done: raise RuntimeError('Step called on a finished episode in step().')

        if action == Action.EXER or self.curr_step == self.nstep - 1:            
            reward = exp(-self.r * (self.curr_step + 1) * self.dt) * self.h(self.s, self.k)
            
            self.done = True
        
        else:
            # action == ACTION.HOLD
            reward = 0

            self.curr_step += 1
            
            self.s = self.prices[self.curr_sim, self.curr_step]
            self.t -= self.dt

        return self._get_obs(), reward, self.done

class Agent:
    def __init__(self, env, hidden_dim, depth, lr, buffer_size, batch_size, buffer_interval, model_interval, gamma, eps, eps_decay, eps_min):
        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available()  else \
            'mps' if torch.backends.mps.is_available() else 'cpu'
        )

        self.env = env

        self.principal = QNet(obssize=self.env.DIM_STATE, actsize=Action.NUM_ACTIONS, hidden_dim=hidden_dim, depth=depth).to(self.device)
        self.target = QNet(obssize=self.env.DIM_STATE, actsize=Action.NUM_ACTIONS, hidden_dim=hidden_dim, depth=depth).to(self.device)

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
                return torch.argmax(self.principal(state_tensor)).item()
        
        return self.env.random_action()

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

    def train(self, nepisode, notebook, verbose=True, **kwargs):
        if notebook: from tqdm.notebook import tqdm
        else:        from tqdm import tqdm

        verbose_freq = kwargs.get('verbose_freq', 100)
        ma_window = kwargs.get('ma_window', 100)

        totalstep = 0
        losses = np.zeros(nepisode)
        rewards = np.zeros(nepisode)
        path_lengths = np.zeros(nepisode)

        for episode in tqdm(range(nepisode), desc='Episode', leave=False, position=0):
            obs = self.env.reset()
            done = False
            loss_sum = rew_sum = step = 0

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
                step += 1
            
            self.eps = max(self.eps * self.eps_decay, self.eps_min)
            
            losses[episode] = loss_sum
            rewards[episode] = rew_sum
            path_lengths[episode] = step
        
            if verbose and episode % verbose_freq == 0:
                start = max(0, episode - ma_window + 1)
                moving_avg_reward = np.mean(rewards[start:episode + 1])
                moving_avg_holding = np.mean(path_lengths[start:episode + 1])
                print(
                    'Episode %s/%s, Total Reward: %.2f, Moving Avg Reward: %.2f, Moving Avg Holding: %.2f, Epsilon: %.4f' % 
                    (episode, nepisode, rew_sum, moving_avg_reward, moving_avg_holding, self.eps)
                )

        fig, axes = plt.subplots(1, 2, figsize=(14, 4))
        for ax, (arr, label) in zip(axes.flatten(), ((rewards, 'reward'), (losses, 'loss'))):
            ax.plot(arr, label=f'Total {label.upper()} per Episode', alpha=0.4, color='blue')
            
            if len(arr) >= ma_window:
                moving_avg = np.convolve(arr, np.ones(ma_window)/ma_window, mode='valid')
                ax.plot(range(ma_window - 1, len(arr)), moving_avg, label=f'Moving Avg {label.upper()} (window={ma_window})', color='red')
            
            ax.set_xlabel('Episode')
            ax.set_ylabel(f'Total {label.upper()}')
            ax.set_title(f'Training Progress: {label.upper()}')
            ax.legend()
            ax.grid(True)
        
        plt.tight_layout()
        plt.close()

        return losses, rewards, fig

    def eval(self, nepisode, notebook):
        if notebook: from tqdm.notebook import tqdm
        else:        from tqdm import tqdm

        rewards = np.zeros(nepisode)
        history = list()

        fig, ax = plt.subplots(1, 1, figsize=(14, 4))

        for episode in tqdm(range(nepisode), desc='Episode', leave=False, position=0):
            obs = self.env.reset()
            done = False
            rew_sum = 0

            path = [self.env.s]
            time = [self.env.t1]

            while not done:
                action = self.act(state=obs)
                newobs, reward, done = self.env.step(action=action)
                step = (self.env.curr_step)*self.env.dt
                obs = newobs
                rew_sum += reward

                path.append(self.env.s)
                time.append(step)
                if done: history.append( (step, self.env.s, action) )

            rewards[episode] = rew_sum

            ax.plot(time, path, color='blue', alpha=0.3)

        steps, prices, actions = [np.array(x, dtype=np.float32) for x in zip(*history)]
        mask = (actions == Action.EXER)

        ax.scatter(steps[mask], prices[mask], color='red', label='Early Exercise')
        ax.scatter(steps[~mask], prices[~mask], color='green', label='Held to Maturity')
        
        ax.axhline(y=self.env.k, color='black', linestyle='--', label='Strike Price')
        
        plt.xlabel('Time (Years)')
        plt.ylabel('Asset Price')
        plt.title(f'Early Exercise Boundary Visualization ({nepisode} Paths)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.close()

        return rewards.mean(), fig

if __name__ == '__main__':
    pass
