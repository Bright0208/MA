import torch
import torch.nn as nn
import torch.nn.functional as F


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, n_agents, hidden_dim):
        super().__init__()

        # centralized input
        self.input_dim = n_agents * (state_dim + action_dim)

        self.fc1 = nn.Linear(self.input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.q_out = nn.Linear(hidden_dim, 1)

    def forward(self, all_states, all_actions):
        # flatten
        x = torch.cat([all_states, all_actions], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q = self.q_out(x)
        return q
