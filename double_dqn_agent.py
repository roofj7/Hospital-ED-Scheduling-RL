# double_dqn_agent.py
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque


class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_size)
        )

    def forward(self, x):
        return self.layers(x)


class DoubleDQNAgent:
    def __init__(
        self, state_size, action_size,
        lr=1e-4, gamma=0.99,
        buffer_size=50000, batch_size=64,
        target_update_every=10
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_every = target_update_every

        self.memory = deque(maxlen=buffer_size)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.online = QNetwork(state_size, action_size).to(self.device)
        self.target = QNetwork(state_size, action_size).to(self.device)
        self.target.load_state_dict(self.online.state_dict())

        self.optimizer = optim.Adam(self.online.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()

    def remember(self, s, a, r, ns, d):
        self.memory.append((s, a, r, ns, d))

    def act(self, state, epsilon):
        if np.random.rand() < epsilon:
            return random.randint(0, self.action_size - 1)

        state = torch.tensor(state.flatten(), dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.online(state)
        return torch.argmax(q_values).item()

    def update_target_network(self):
        self.target.load_state_dict(self.online.state_dict())

    def replay(self):
        if len(self.memory) < self.batch_size:
            return None
        
        batch = random.sample(self.memory, self.batch_size)
        
        states = torch.tensor([s.flatten() for s, _, _, _, _ in batch], dtype=torch.float32, device=self.device)
        actions = torch.tensor([a for _, a, _, _, _ in batch], dtype=torch.long, device=self.device).unsqueeze(1)
        rewards = torch.tensor([r for _, _, r, _, _ in batch], dtype=torch.float32, device=self.device)
        next_states = torch.tensor([ns.flatten() for _, _, _, ns, _ in batch], dtype=torch.float32, device=self.device)
        dones = torch.tensor([d for _, _, _, _, d in batch], dtype=torch.float32, device=self.device)

        q_values = self.online(states).gather(1, actions).squeeze()

        with torch.no_grad():
            next_actions = self.online(next_states).argmax(dim=1, keepdim=True)
            q_next = self.target(next_states).gather(1, next_actions).squeeze()
            targets = rewards + (1 - dones) * self.gamma * q_next

        loss = self.loss_fn(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()
