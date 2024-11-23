from collections import deque
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

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
