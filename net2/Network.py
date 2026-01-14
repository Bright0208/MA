import torch
import torch.nn as nn
import torch.nn.functional as F


class Critic(nn.Module):
    def __init__(self, n_agent, dim_observation, dim_action):
        super(Critic, self).__init__()
        # Critic 输入的是：所有智能体的观测 + 所有智能体的动作 (CTDE机制)
        self.n_agent = n_agent
        self.dim_observation = dim_observation
        self.dim_action = dim_action

        # 输入维度 = (单个Obs维度 * 智能体数) + (单个Action维度 * 智能体数)
        # 注意：如果你用了GNN，这里会由 GNN 提取特征代替简单的拼接
        obs_dim = dim_observation * n_agent
        act_dim = dim_action * n_agent

        self.FC1 = nn.Linear(obs_dim + act_dim, 256)
        self.FC2 = nn.Linear(256, 128)
        self.FC3 = nn.Linear(128, 1)  # 输出 Q 值

    def forward(self, obs, acts):
        # obs: [batch_size, n_agent * dim_obs]
        # acts: [batch_size, n_agent * dim_act]
        combined = torch.cat([obs, acts], dim=1)
        result = F.relu(self.FC1(combined))
        result = F.relu(self.FC2(result))
        return self.FC3(result)


class Actor(nn.Module):
    def __init__(self, dim_observation, dim_action):
        super(Actor, self).__init__()
        self.FC1 = nn.Linear(dim_observation, 128)
        self.FC2 = nn.Linear(128, 64)
        self.FC3 = nn.Linear(64, dim_action)  # 输出动作的 Logits

    def forward(self, obs):
        result = F.relu(self.FC1(obs))
        result = F.relu(self.FC2(result))
        result = self.FC3(result)
        # 注意：这里不直接做 softmax，因为我们在 Agent 里做 Gumbel-Softmax
        return result