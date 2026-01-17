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
models = ['vit-image', 'radar-former', 'llama-8b']
precision = [2, 2, 2]
Token_in = [40, 40, 25]
Token_out = [30, 5, 40]
cot_paths = [1, 1, 3]
required_info = {'models': models,
                 'precision': precision,
                 'Token_in': Token_in,
                 'Token_out': Token_out,
                 'deadline':   2.0,
                 'cot_paths': cot_paths}

# task_id = f"v{vehicle.id}_t{self.current_step}"
task_id = "0"
parent_task = ParentTask(task_id, "V_0", required_info)

for name, rsu in Rsu_dict.items():
    for name, sub_task in parent_task.sub_tasks.items():
        rsu.task_queue.append(sub_task)
    rsu.deployed_models ={'llama-8b':2,"vit-image":2,"point-lidar":1,"radar-former":3}
    # print(rsu.task_queue)


if __name__ == "__main__":
    env = Environment(model_library, Rsu_dict, Vehicle_dict)
    env.task_registry[task_id] = parent_task
    env.determine_vehicle_ownership()

    n_agent = env.get_agent_numbers()   #获得智能体数量

    # state = env.reset()
    state_dim = env.get_state_dim()          #通过state获得state_dim
    action_dim = env.get_action_dim()   #获得action_dim
    # dim_obs = 50  # 观测维度 (任务队列长度 + 车辆位置 + 信道状态等)
    # dim_act = 10  # 动作维度 (比如: 3个卸载目标 * 3种精度 + 1个本地 = 10种离散选择)
    # maddpg = MADDPG(n_agents, dim_obs, dim_act, batch_size=64)

    maddpg = MADDPG(n_agent,state_dim,action_dim,64)
    replay_buffer = ReplayBuffer(capacity=100000)  # 你需要自己写一个简单的 Buffer 类
    MAX_EPISODES = 5000

    for episode in range(MAX_EPISODES):
        # action_dict = {'Rsu_0':2, 'Rsu_1':4, 'Rsu_2':7, 'Rsu_3':1, 'Rsu_4':3, 'Rsu_5':9}
        # # obs_n = env.reset(seed=42)
        # obs_n = env.reset()
        # print(obs_n)
        # break
        # # env.step(action_dict)

        obs_n = env.reset()  # 获取所有 RSU 的初始观测 list
        total_reward = 0
        for step in range(200):  # 每个 Episode 200 步
            # 1. 选择动作
            actions_n = maddpg.select_action(obs_n)
            print("动作:",actions_n)
            # 2. 环境执行动作
            # 这里你需要将 one-hot 的 actions_n 转换为具体的部署/卸载指令传给 env
            for name, action_ in actions_n.items():
                action = np.argmax(action_)
                actions_n[name] = action

            next_obs_n, reward_n, done_n, _ = env.step(actions_n)

            # 3. 存入 Buffer
            replay_buffer.push(obs_n, actions_n, reward_n, next_obs_n, done_n)

            obs_n = next_obs_n
            total_reward += sum(reward_n)

            # 4. 开始训练 (当数据足够时)
            if len(replay_buffer) > 64:
                maddpg.update(replay_buffer)

            if all(done_n):
                break

        print(f"Episode {episode}: Reward = {total_reward}")






