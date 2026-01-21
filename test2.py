import os

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import datetime
from env.environment import Environment
from config import device, MODELS_CONFIG, RSU_CONFIGS, Vehicle_CONFIGS
from env.llm_model import LLMModel
from env.rsu import Rsu
from env.vehicle import Vehicle
from net2.MADDPG import MADDPG
from net2.ReplayBuffer import ReplayBuffer



if __name__ == "__main__":

    # 实例化模型库
    model_library = {name: LLMModel(name, conf) for name, conf in MODELS_CONFIG.items()}
    # 实例化RSU
    Rsu_dict = {name: Rsu(conf) for name, conf in RSU_CONFIGS.items()}
    # TODO ：构建邻居
    # 实例化车辆
    Vehicle_dict = {name: Vehicle(conf) for name, conf in Vehicle_CONFIGS.items()}

    env = Environment(model_library, Rsu_dict, Vehicle_dict)
    env.determine_vehicle_ownership()

    n_agent = env.get_agent_numbers()  # 获得智能体数量
    state_dim = env.get_state_dim()  # 通过state获得state_dim
    action_dim = env.get_action_dim()  # 获得action_dim

    # === 1. 初始化两个大脑 ===
    # 慢智能体 (Deploy): 动作空间 12 (4模型 x 3精度)
    maddpg_deploy = MADDPG(n_agent, state_dim, dim_act=12, batch_size=64)
    buffer_deploy = ReplayBuffer(capacity=10000)

    # 快智能体 (Task): 动作空间 10 (Local/Offload/Drop)
    maddpg_task = MADDPG(n_agent, state_dim, dim_act=10, batch_size=256)
    buffer_task = ReplayBuffer(capacity=100000)

    DEPLOY_INTERVAL = 50  # 时间尺度 K
    MAX_EPISODES = 5000

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

    # 创建模型保存目录
    os.makedirs('models', exist_ok=True)

    for episode in range(MAX_EPISODES):
        # ... (训练循环) ...
        obs_n = env.reset(seed=42 + episode)  # 获取所有 RSU 的初始观测 list

        # 统计变量
        episode_reward = 0

        # --- 慢智能体 状态记录 ---
        deploy_obs = obs_n  # 记录 K 步开始时的状态 (S_slow)
        deploy_actions = None  # 记录 K 步开始时的动作 (A_slow)
        # 记录 K 步内的累积奖励 (Reward_slow)
        interval_rewards = {f'Rsu_{i}': 0.0 for i in range(n_agent)}

        # 用于 TensorBoard 的 Loss 记录
        task_c_losses, task_a_losses = [], []
        deploy_c_losses, deploy_a_losses = [], []

        for step in range(200):  # 每个 Episode 200 步
            agent_ids = [f'Rsu_{i}' for i in range(n_agent)]
            # print("episode:", episode, "step:", step)
            # ==========================================
            # A. 慢速层决策 (每 50 步执行一次)
            # ==========================================
            if step % DEPLOY_INTERVAL == 0:
                # 1. [关键] 结算上一轮慢动作 (如果有)
                # 将 (S_slow, A_slow, Sum_R, S'_slow, Done) 存入慢 Buffer
                if step > 0 and deploy_actions is not None:
                    # 准备数据
                    obs_list_d = [deploy_obs[key] for key in agent_ids]
                    # 动作需要转 One-Hot (dim=12)
                    act_list_d = [deploy_actions[key] for key in agent_ids]
                    # 累积奖励
                    rew_list_d = [interval_rewards[key] for key in agent_ids]
                    # 下一状态 (当前的 obs_n 就是上一轮部署后的结果状态 S')
                    next_obs_list_d = [obs_n[key] for key in agent_ids]
                    # Done (每 50 步不算 Done，除非 Episode 结束，这里暂填 False)
                    done_list_d = [False for _ in agent_ids]
                    # 存入慢 Buffer
                    buffer_deploy.push(obs_list_d, act_list_d, rew_list_d, next_obs_list_d, done_list_d)

                    # 训练慢智能体
                    if len(buffer_deploy) > 64:
                        c_loss, a_loss = maddpg_deploy.update(buffer_deploy)
                        deploy_c_losses.append(c_loss)
                        deploy_a_losses.append(a_loss)
                # 2. 慢智能体决策 (输出 0-11)
                deploy_actions = maddpg_deploy.select_action(obs_n)
                # 3. 环境执行部署 (step_deploy)
                # 注意：这会改变 RSU 的内存状态，从而改变 obs
                obs_after_deploy, deploy_infos = env.step_deploy(deploy_actions)
                # 4. 更新状态与重置累积奖励
                obs_n = obs_after_deploy  # 更新环境状态
                deploy_obs = obs_n  # 记录新的 S_slow
                # 重置累积奖励，并扣除部署成本 (Cost Penalty)
                # 我们给慢智能体一个负奖励，防止它频繁乱动
                for key in agent_ids:
                    info = deploy_infos.get(key, {'cost_time': 0})
                    cost_penalty = -2.0 * info['cost_time']  # 这里的系数可以调
                    interval_rewards[key] = cost_penalty

            # ==========================================
            # B. 快速层决策 (每 1 步执行一次)
            # ==========================================
            # 1. 选择动作
            actions_n = maddpg_task.select_action(obs_n)

            next_obs_n, reward_n, done_n, _ = env.step(actions_n)

            # print("obs_n:",obs_n)
            # print("actions_n:",actions_n)
            # print("next_obs_n", next_obs_n)
            # print("reward_n", reward_n)
            # print("done_n", done_n)

            # ================= 关键修改开始 =================
            # 定义一个按顺序的 ID 列表，确保所有数据对齐！
            # 假设 n_agents = 6，生成 ['Rsu_0', 'Rsu_1', ..., 'Rsu_5']
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
            buffer_task.push(obs_list, act_list, rew_list, next_obs_list, done_list)
            # replay_buffer.push(obs_n, actions_n, reward_n, next_obs_n, done_n)

            for name, reward_ in reward_n.items():
                interval_rewards[name] += reward_
                episode_reward += reward_

            obs_n = next_obs_n

            # 4. 开始训练 (当数据足够时)
            if len(buffer_task) > 256:
                c_loss, a_loss = maddpg_task.update(buffer_task)
                task_c_losses.append(c_loss)
                task_a_losses.append(a_loss)

            if all(done_n.values()):
                break
        # === 3. 记录本轮数据 ===
        all_ep_rewards.append(episode_reward)
        # =========================================================
        # End of Episode: 处理最后一次部署的经验
        # =========================================================
        if deploy_actions is not None:
            # 把最后一段的累积奖励存进去
            obs_list_d = [deploy_obs[key] for key in agent_ids]
            act_list_d = [deploy_actions[key] for key in agent_ids]
            # 最后一步的 next_obs
            next_obs_list_d = [obs_n[key] for key in agent_ids]
            # 这里可以算 True Done
            done_list_d = [True for _ in agent_ids]
            buffer_deploy.push(obs_list_d, act_list_d, rew_list_d, next_obs_list_d, done_list_d)
        # =========================================================
        # Logging: 记录数据到 TensorBoard
        # =========================================================
        # 1. Main Reward
        writer.add_scalar('Main/Total_Reward', episode_reward, episode)
        # 2. Task Agent Losses
        if len(task_c_losses) > 0:
            writer.add_scalar('Loss_Task/Critic', np.mean(task_c_losses), episode)
            writer.add_scalar('Loss_Task/Actor', np.mean(task_a_losses), episode)
        # 3. Deploy Agent Losses
        if len(deploy_c_losses) > 0:
            writer.add_scalar('Loss_Deploy/Critic', np.mean(deploy_c_losses), episode)
            writer.add_scalar('Loss_Deploy/Actor', np.mean(deploy_a_losses), episode)
        # 4. Fairness Metrics (最小完成率)
        rsu_rates = []
        for r_name, rsu in env.rsus.items():
            total_tasks = sum([m['total'] for m in rsu.stats.values()])
            total_success = sum([m['success'] for m in rsu.stats.values()])
            rate = total_success / total_tasks if total_tasks > 0 else 0.0
            rsu_rates.append(rate)
        min_rate = min(rsu_rates) if len(rsu_rates) > 0 else 0.0

        # # Jain's Index
        # rates_arr = np.array(rsu_rates)
        # if np.sum(rates_arr ** 2) == 0:
        #     jains = 0.0
        # else:
        #     jains = (np.sum(rates_arr) ** 2) / (len(rates_arr) * np.sum(rates_arr ** 2))

        writer.add_scalar('Fairness/Min_Rate', min_rate, episode)
        # writer.add_scalar('Fairness/Jains_Index', jains, episode)

        print(
            f"Ep {episode}: Reward={episode_reward:.2f} | MinRate={min_rate:.2f} | TaskLoss={np.mean(task_c_losses) if task_c_losses else 0:.2f}")

        # =========================================================
        # Save Models (Every 100 episodes)
        # =========================================================
        if episode > 0 and episode % 100 == 0:
            # 保存 Fast Agent
            for name, agent in maddpg_task.agents.items():
                torch.save(agent.actor.state_dict(), f'models/task_{name}_actor_ep{episode}.pth')
                torch.save(agent.critic.state_dict(), f'models/task_{name}_critic_ep{episode}.pth')
            # 保存 Slow Agent
            for name, agent in maddpg_deploy.agents.items():
                torch.save(agent.actor.state_dict(), f'models/deploy_{name}_actor_ep{episode}.pth')
                torch.save(agent.critic.state_dict(), f'models/deploy_{name}_critic_ep{episode}.pth')

    writer.close()
    print("训练结束！")

