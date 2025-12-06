# dqn_agent_fixed.py
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
from ed_env import SERVICE_TO_RESOURCE_MAP


class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_size)
        )

    def forward(self, x):
        return self.fc(x)


class DQNAgent:
    def __init__(self, state_size, action_size, lr=1e-4, gamma=0.99):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=100000)
        self.gamma = gamma
        
        # Epsilon values BUT NO DECAY in replay()
        self.epsilon = 1.0
        self.epsilon_min = 0.02
        self.epsilon_decay = 0.995
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DQN(state_size, action_size).to(self.device)
        self.target_model = DQN(state_size, action_size).to(self.device)
        self.update_target()

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()

    def _get_action_mask(self, env, flat_state):
        mask = np.zeros(self.action_size, dtype=bool)

        M = env.max_patients
        R = env.resource_names

        rs = flat_state[M * env.n_features:]
        pf = flat_state[:M * env.n_features].reshape(M, env.n_features)

        for p in range(M):
            for r_idx, r_name in enumerate(R):
                idx = p * len(R) + r_idx
                if rs[r_idx] > 0:
                    for svc, target_r in SERVICE_TO_RESOURCE_MAP.items():
                        si = env.service_indices.get(svc)
                        if si != -1 and target_r == r_name and pf[p, si] > 0:
                            mask[idx] = True

        mask[self.action_size - 1] = True
        return mask

    def update_target(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, s, a, r, ns, d):
        self.memory.append((s, a, r, ns, d))

    def act(self, env, state):
        flat = state.flatten()
        mask = self._get_action_mask(env, flat)

        if np.random.rand() <= self.epsilon:
            return np.random.choice(np.where(mask)[0])

        qs = self.model(torch.tensor(flat, dtype=torch.float32, device=self.device).unsqueeze(0)).squeeze()
        qs[~torch.tensor(mask).to(self.device)] = -1e9
        return qs.argmax().item()

    def replay(self, env, batch_size=64):
        if len(self.memory) < batch_size:
            return None

        batch = random.sample(self.memory, batch_size)
        states = torch.tensor([s.flatten() for s,_,_,_,_ in batch], dtype=torch.float32, device=self.device)
        actions = torch.tensor([a for _,a,_,_,_ in batch], dtype=torch.long, device=self.device).unsqueeze(1)
        rewards = torch.tensor([r for _,_,r,_,_ in batch], dtype=torch.float32, device=self.device)
        next_states = torch.tensor([ns.flatten() for _,_,_,ns,_ in batch], dtype=torch.float32, device=self.device)
        dones = torch.tensor([d for _,_,_,_,d in batch], dtype=torch.bool, device=self.device)

        q_curr = self.model(states).gather(1, actions).squeeze()

        with torch.no_grad():
            next_q = self.target_model(next_states)
            for i, ns in enumerate(next_states):
                mask = self._get_action_mask(env, ns.cpu().numpy())
                next_q[i][~torch.tensor(mask).to(self.device)] = -1e9
            next_q = next_q.max(1)[0]

            q_target = rewards + (~dones) * self.gamma * next_q

        loss = self.loss_fn(q_curr, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()
