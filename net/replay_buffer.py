import numpy as np
import random
from collections import deque


class ReplayBuffer:
    def __init__(self, max_size, state_dim, action_dim, n_agents):
        self.max_size = max_size
        self.n_agents = n_agents

        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, n_agents, state_dim), dtype=np.float32)
        self.action = np.zeros((max_size, n_agents, action_dim), dtype=np.float32)
        self.reward = np.zeros((max_size, n_agents), dtype=np.float32)
        self.next_state = np.zeros((max_size, n_agents, state_dim), dtype=np.float32)
        self.done = np.zeros((max_size, n_agents), dtype=np.float32)

    def add(self, state, action, reward, next_state, done):
        idx = self.ptr

        self.state[idx] = state
        self.action[idx] = action
        self.reward[idx] = reward
        self.next_state[idx] = next_state
        self.done[idx] = done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        idxs = np.random.choice(self.size, batch_size, replace=False)
        return (
            self.state[idxs],
            self.action[idxs],
            self.reward[idxs],
            self.next_state[idxs],
            self.done[idxs]
        )


if __name__ == "__main__":
    # 简单自测：写入并采样一次
    buf = ReplayBuffer(max_size=10, state_dim=5, action_dim=3, n_agents=2)
    sample_state = np.ones((2, 5), dtype=np.float32)
    sample_action = np.zeros((2, 3), dtype=np.float32)
    sample_reward = np.array([1.0, 1.0], dtype=np.float32)
    sample_next_state = np.zeros((2, 5), dtype=np.float32)
    sample_done = np.zeros((2,), dtype=np.float32)

    for _ in range(5):
        buf.add(sample_state, sample_action, sample_reward, sample_next_state, sample_done)

    batch = buf.sample(2)
    print(batch)

    print("sample shapes:", [x.shape for x in batch])
