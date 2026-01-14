import random
import numpy as np


class ReplayBuffer:
    def __init__(self, capacity):
        """
        :param capacity: 经验池的最大容量 (例如 100000)
        """
        self.capacity = capacity
        self.buffer = []
        self.position = 0  # 当前写入位置指针

    def push(self, obs_n, act_n, r_n, next_obs_n, done_n):
        """
        存入一条多智能体的联合经验
        注意：输入参数都是 list，长度等于 agent 数量

        obs_n:      [obs_1, obs_2, ..., obs_N]
        act_n:      [act_1, act_2, ..., act_N] (通常是 One-hot 或 Softmax 输出)
        r_n:        [r_1, r_2, ..., r_N]
        next_obs_n: [next_obs_1, ..., next_obs_N]
        done_n:     [done_1, ..., done_N]
        """
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)

        # 存储元组
        self.buffer[self.position] = (obs_n, act_n, r_n, next_obs_n, done_n)

        # 循环覆盖 (Ring Buffer)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        """
        随机采样一个 Batch
        """
        batch = random.sample(self.buffer, batch_size)

        # 解压缩 batch
        # batch 是一个 list，里面有 batch_size 个元组
        # zip(*) 会把它们按列重新组合
        obs_n_batch, act_n_batch, r_n_batch, next_obs_n_batch, done_n_batch = zip(*batch)

        # 转换为 numpy array 以便转 Tensor
        # 最终形状: [batch_size, n_agents, feature_dim]
        return (
            np.array(obs_n_batch),
            np.array(act_n_batch),
            np.array(r_n_batch),
            np.array(next_obs_n_batch),
            np.array(done_n_batch)
        )

    def __len__(self):
        return len(self.buffer)