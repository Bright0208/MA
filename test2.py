import numpy as np
import torch

from env.environment import Environment
from config import device, MODELS_CONFIG, RSU_CONFIGS, Vehicle_CONFIGS
from env.llm_model import LLMModel
from env.rsu import Rsu
from env.task import SubTask, ParentTask
from env.vehicle import Vehicle
from net2.MADDPG import MADDPG
from net2.ReplayBuffer import ReplayBuffer

# 实例化模型库
model_library = {name: LLMModel(name, conf) for name, conf in MODELS_CONFIG.items()}
# 实例化RSU
Rsu_dict = {name :Rsu(conf) for name, conf in RSU_CONFIGS.items()}
# TODO ：构建邻居
# 实例化车辆
Vehicle_dict = {name : Vehicle(conf) for name, conf in Vehicle_CONFIGS.items()}

# 给Rsu任务队列里面加点任务
# 假设这辆车需要同时处理 图像、雷达 和 LLM
# models = ['vit-image', 'radar-former', 'llama-8b']
# precision = [2, 2, 2]
# Token_in = [40, 40, 25]
# Token_out = [30, 5, 40]
# cot_paths = [1, 1, 3]
# required_info = {'models': models,
#                  'precision': precision,
#                  'Token_in': Token_in,
#                  'Token_out': Token_out,
#                  'deadline':   2.0,
#                  'cot_paths': cot_paths}

# task_id = f"v{vehicle.id}_t{self.current_step}"
# task_id = "0"
# parent_task = ParentTask(task_id, "V_0", required_info)

# for name, rsu in Rsu_dict.items():
#     for name, sub_task in parent_task.sub_tasks.items():
#         rsu.task_queue.append(sub_task)
#     rsu.deployed_models ={'llama-8b':2,"vit-image":2,"point-lidar":1,"radar-former":3}
#     # print(rsu.task_queue)


if __name__ == "__main__":
    env = Environment(model_library, Rsu_dict, Vehicle_dict)
    env.determine_vehicle_ownership()

    n_agent = env.get_agent_numbers()   #获得智能体数量
    state_dim = env.get_state_dim()          #通过state获得state_dim
    action_dim = env.get_action_dim()   #获得action_dim

    maddpg = MADDPG(n_agent,state_dim,action_dim,64)
    replay_buffer = ReplayBuffer(capacity=100000)  # 你需要自己写一个简单的 Buffer 类
    MAX_EPISODES = 5000

    for episode in range(MAX_EPISODES):

        obs_n = env.reset(seed=42)  # 获取所有 RSU 的初始观测 list
        total_reward = 0
        for step in range(200):  # 每个 Episode 200 步
            # 1. 选择动作
            actions_n = maddpg.select_action(obs_n)

            next_obs_n, reward_n, done_n, _ = env.step(actions_n)

            # print("obs_n:",obs_n)
            # print("actions_n:",actions_n)
            # print("next_obs_n", next_obs_n)
            # print("reward_n", reward_n)
            # print("done_n", done_n)

            # ================= 关键修改开始 =================
            # 定义一个按顺序的 ID 列表，确保所有数据对齐！
            # 假设 n_agents = 6，生成 ['Rsu_0', 'Rsu_1', ..., 'Rsu_5']
            agent_ids = [f'Rsu_{i}' for i in range(n_agent)]
            # 1. 转换 Observation (Dict -> List of Arrays)
            obs_list = [obs_n[key] for key in agent_ids]
            # 2. 转换 Action (Dict -> List of One-Hot Arrays)
            # 注意：这里存的一定要是 One-Hot 数组，不是整数！
            # 如果 actions_n 里的值已经是 array([0,0,1..]) 就直接取
            # 如果是整数，这里需要加 to_onehot 函数
            act_list = [actions_n[key] for key in agent_ids]
            # 3. 转换 Reward (Dict -> List of floats)
            rew_list = [reward_n[key] for key in agent_ids]
            # 4. 转换 Next Observation (Dict -> List of Arrays)
            next_obs_list = [next_obs_n[key] for key in agent_ids]
            # 5. 转换 Done (Dict -> List of bools/floats)
            done_list = [done_n[key] for key in agent_ids]
            # ================= 关键修改结束 =================

            # 3. 存入 Buffer
            replay_buffer.push(obs_list, act_list, rew_list, next_obs_list, done_list)
            # replay_buffer.push(obs_n, actions_n, reward_n, next_obs_n, done_n)

            obs_n = next_obs_n
            for name, reward_ in reward_n.items():
                total_reward += reward_

            # 4. 开始训练 (当数据足够时)
            if len(replay_buffer) > 64:
                maddpg.update(replay_buffer)

            if all(done_n):
                break

        print(f"Episode {episode}: Reward = {total_reward}")






