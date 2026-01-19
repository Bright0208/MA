import numpy as np
from torch.utils.tensorboard import SummaryWriter
import datetime
from env.environment import Environment
from config import device, MODELS_CONFIG, RSU_CONFIGS, Vehicle_CONFIGS
from env.llm_model import LLMModel
from env.rsu import Rsu
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


env = Environment(model_library, Rsu_dict, Vehicle_dict)
env.determine_vehicle_ownership()

n_agent = env.get_agent_numbers()  # 获得智能体数量
state_dim = env.get_state_dim()  # 通过state获得state_dim
action_dim = env.get_action_dim()  # 获得action_dim

maddpg = MADDPG(n_agent, state_dim, action_dim, 64)
replay_buffer = ReplayBuffer(capacity=100000)  # 你需要自己写一个简单的 Buffer 类
MAX_EPISODES = 5000


if __name__ == "__main__":
    # ... (初始化) ...
    # === 1. 新增记录列表 ===
    all_ep_rewards = []  # 记录 Reward
    all_critic_losses = []  # 记录 Critic Loss
    all_actor_losses = []  # 记录 Actor Loss
    # === 初始化 Tensorboard Writer ===
    # 日志会保存在 runs/文件夹下，加上时间戳防止覆盖
    log_dir = "runs/MADDPG_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    writer = SummaryWriter(log_dir)
    print(f"TensorBoard 启动中... 日志目录: {log_dir}")
    print("请在终端运行: tensorboard --logdir=runs 来查看实时图表")
    for episode in range(MAX_EPISODES):
        # ... (训练循环) ...
        obs_n = env.reset()  # 获取所有 RSU 的初始观测 list
        total_reward = 0
        # 记录这一个 Episode 里每一步的 Loss，最后求平均
        step_c_losses = []
        step_a_losses = []

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
                c_loss, a_loss = maddpg.update(replay_buffer)
                step_c_losses.append(c_loss)
                step_a_losses.append(a_loss)

            if all(done_n):
                break
        # === 3. 记录本轮数据 ===
        all_ep_rewards.append(total_reward)

        # 在 Episode 结束时记录
        print(f"Episode {episode}: Reward = {total_reward}")

        # 如果这一轮进行了训练，计算平均 Loss；否则记为 0
        if len(step_c_losses) > 0:
            avg_c = np.mean(step_c_losses)
            avg_a = np.mean(step_a_losses)
        else:
            avg_c, avg_a = 0, 0

        all_critic_losses.append(avg_c)
        all_actor_losses.append(avg_a)
        # === 写入 Tensorboard ===
        # 记录总奖励
        writer.add_scalar('Reward/Total_Reward', total_reward, episode)
        # ======================================================
        # [新增] 计算平均值并写入 TensorBoard
        # ======================================================
        # 1. 记录总奖励 (Total Reward)
        writer.add_scalar('Main/Total_Reward', total_reward, episode)

        # 2. 记录平均 Loss (如果本轮进行了训练)
        if len(step_c_losses) > 0:
            avg_c_loss = np.mean(step_c_losses)
            avg_a_loss = np.mean(step_a_losses)

            writer.add_scalar('Loss/Critic_Loss', avg_c_loss, episode)
            writer.add_scalar('Loss/Actor_Loss', avg_a_loss, episode)

        # 打印进度
        # 如果你想看每一步的平均 Loss (需要在 maddpg.update 里返回 loss)
        # writer.add_scalar('Loss/Critic_Loss', critic_loss, episode)

    writer.close()


