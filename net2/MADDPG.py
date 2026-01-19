import torch
import numpy as np
from net2.net import Actor, Critic  # 假设上面的代码存为 net.py


class MADDPG:
    def __init__(self, n_agents, dim_obs, dim_act, batch_size, gamma=0.95, tau=0.01):
        self.agents = {}
        self.n_agents = n_agents
        self.dim_obs = dim_obs
        self.dim_act = dim_act
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau

        # 设备配置
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 初始化每个 Agent 的网络
        for i in range(n_agents):
            # 每个 Agent 都有自己的 Actor 和 Critic
            agent = Agent(dim_obs, dim_act, n_agents, self.device)
            self.agents[f"Rsu_{i}"] = agent

    def select_action(self, obs_n):
        """
        执行阶段：分散执行。
        obs_n: list of observations, 每个 agent 一个 obs
        """
        actions = {}
        for name, agent in self.agents.items():
            # 只输入当前 agent 的观测
            action = agent.get_action(obs_n[name])
            # print("MADDPG 34 aciton",action)

            actions[name] = action

        return actions  # 返回动作字典

    def update(self, replay_buffer):
        """
        训练阶段：集中训练。
        """
        # 1. 从经验池采样
        # sample 包含所有智能体的 (obs, action, reward, next_obs, done)
        obs_n, act_n, r_n, next_obs_n, done_n = replay_buffer.sample(self.batch_size)

        # print("MADDPG 48 obs_n",obs_n)

        # 转为 Tensor 并放到 GPU
        # obs_n 的形状: [batch, n_agent, dim_obs]
        obs_n = torch.FloatTensor(obs_n).to(self.device)
        act_n = torch.FloatTensor(act_n).to(self.device)
        r_n = torch.FloatTensor(r_n).to(self.device)
        next_obs_n = torch.FloatTensor(next_obs_n).to(self.device)
        done_n = torch.FloatTensor(done_n).to(self.device)

        # === 新增：用于记录总 Loss 的变量 ===
        total_critic_loss = 0.0
        total_actor_loss = 0.0

        # ----------------------------------------
        # 关键步骤：为了计算 Critic Loss，我们需要所有 Agent 下一步的动作
        # ----------------------------------------
        target_act_next_n = []
        for name, agent in self.agents.items():
            # 使用 Target Actor 网络预测下一步动作
            # 注意：这里必须使用 Gumbel-Softmax 保持可微性，或者在 Target 中直接 argmax
            i = int(name.split('_')[1])
            target_act_next = agent.actor_target(next_obs_n[:, i])
            # Gumbel-Softmax 处理
            target_act_next = torch.nn.functional.gumbel_softmax(target_act_next, hard=True)
            target_act_next_n.append(target_act_next)

        # 拼接所有 Agent 的下一步动作 [batch, n_agent * dim_act]
        target_act_next_n = torch.cat(target_act_next_n, dim=1)
        # 拼接所有 Agent 的下一步观测 [batch, n_agent * dim_obs]
        next_obs_n_flat = next_obs_n.view(self.batch_size, -1)
        # 拼接当前观测和动作
        obs_n_flat = obs_n.view(self.batch_size, -1)
        act_n_flat = act_n.view(self.batch_size, -1)

        # ----------------------------------------
        # 循环更新每一个 Agent
        # ----------------------------------------
        for name, agent in self.agents.items():
            i = int(name.split('_')[1])
            # --- 1. 更新 Critic ---
            # 计算目标 Q 值 (Target Q)
            with torch.no_grad():
                q_next = agent.critic_target(next_obs_n_flat, target_act_next_n)
                # Bellman 公式: y = r + gamma * q_next * (1 - done)
                target_q = r_n[:, i].view(-1, 1) + self.gamma * q_next * (1 - done_n[:, i].view(-1, 1))

            # 计算当前 Q 值
            current_q = agent.critic(obs_n_flat, act_n_flat)

            # Critic Loss (MSE)
            critic_loss = torch.nn.functional.mse_loss(current_q, target_q)

            # === 新增：累加 Critic Loss ===
            total_critic_loss += critic_loss.item()

            agent.optimizer_c.zero_grad()
            critic_loss.backward()
            agent.optimizer_c.step()

            # --- 2. 更新 Actor ---
            # 这里的目的是最大化 Critic 的评分
            # 技巧：我们需要重新计算当前的 actions，因为 buffer 里的 actions 是旧策略产生的，没有梯度

            curr_act_n = []
            for name_, ag in self.agents.items():
                j = int(name_.split('_')[1])
                act_logits = ag.actor(obs_n[:, j])
                # 必须使用 Gumbel-Softmax !
                # 这样 Critic 对 action 的梯度才能传回给 Actor
                act = torch.nn.functional.gumbel_softmax(act_logits, hard=False)
                curr_act_n.append(act)

            curr_act_n_flat = torch.cat(curr_act_n, dim=1)

            # Actor Loss = -Q (最大化 Q 等于最小化 -Q)
            # 这里的 Critic 是 Agent i 自己的 Critic，但输入了所有人的动作
            actor_loss = -agent.critic(obs_n_flat, curr_act_n_flat).mean()

            # === 新增：累加 Actor Loss ===
            total_actor_loss += actor_loss.item()

            agent.optimizer_a.zero_grad()
            actor_loss.backward()
            agent.optimizer_a.step()

            # --- 3. 软更新 Target 网络 ---
            agent.update_target(self.tau)
            # === 新增：返回平均 Loss ===
        return total_critic_loss / self.n_agents, total_actor_loss / self.n_agents

class Agent:
    def __init__(self, dim_obs, dim_act, n_agents, device):
        self.actor = Actor(dim_obs, dim_act).to(device)
        self.critic = Critic(n_agents, dim_obs, dim_act).to(device)
        self.actor_target = Actor(dim_obs, dim_act).to(device)
        self.critic_target = Critic(n_agents, dim_obs, dim_act).to(device)

        # 复制参数
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.optimizer_a = torch.optim.Adam(self.actor.parameters(), lr=1e-4)
        self.optimizer_c = torch.optim.Adam(self.critic.parameters(), lr=1e-3)
        self.device = device

    def get_action(self, obs):
        # 探索策略 (Testing时)
        state = torch.FloatTensor(obs).to(self.device).unsqueeze(0)
        logits = self.actor(state)

        # print("MADDPG.py: logits:",logits)

        # 在执行阶段，我们可以使用 Gumbel-Softmax 进行采样，也可以直接 argmax
        action = torch.nn.functional.gumbel_softmax(logits, hard=True)
        # print("MADDPG 145 action",action)
        return action.detach().cpu().numpy()[0]

    def update_target(self, tau):
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)