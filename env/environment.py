import gymnasium as gym
import numpy as np
from gymnasium import spaces
import math

from config import Vehicle_CONFIGS
from env.task import ParentTask
from env.vehicle import Vehicle


def _map_model_id(name):
    mapping = {'llama-8b': [1,0,0,0], 'vit-image': [0,1,0,0], 'point-lidar': [0,0,1,0], 'radar-former': [0,0,0,1]}
    return mapping.get(name, 0)


class Environment(gym.Env):
    def __init__(self,Model,Rsu,Vehicle):
        super(Environment, self).__init__()
        # --- 1. 环境配置 ---
        self.rsus = Rsu
        self.vehicles = Vehicle
        self.LLMModel  = Model
        self.dt = 0.1  # 仿真步长 (秒)
        self.current_step = 0
        self.max_steps = 200  # 每个 Episode 的长度
        self.task_registry = {}
        # --- 2. 动作空间 (Action Space) ---
        # 假设每个 Agent (RSU) 一次决策一个任务
        # 动作定义: [处理方式]
        # 方式包括:
        #   0-2: 本地执行 (精度 k=1, 2, 3)
        #   3-5: 卸载给左邻居 (精度 k=1, 2, 3)
        #   6-8: 卸载给右邻居 (精度 k=1, 2, 3)
        #   9: 拒绝任务 (Drop)
        self.n_actions = 10
        self.action_space = spaces.Discrete(self.n_actions)

        # --- 3. 观测空间 (Observation Space) ---
        # [任务类型ID, 任务数据量, 容忍时延, 本地CPU利用率, 本地内存利用率, 邻居1负载, 邻居2负载...]
        self.obs_dim = 3 + 2 + 2  # 简单示例: 7维
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.obs_dim,), dtype=np.float32)


    def determine_vehicle_ownership(self):
        for r_name, rsu in self.rsus.items():
            for v_name ,vehicle in self.vehicles.items():
                rsu.is_vehicle_in_range(vehicle)

    def agent_ids(self):
        return [rsu.id for rsu in self.rsus]

    def get_agent_numbers(self):
        return len(self.rsus)

    def get_action_dim(self):
        return 10
    def get_state_dim(self):
        return 11

    def reset(self, seed=None):
        """
        重置环境状态，开始一个新的 Episode。
        """
        # 1. 设置随机种子 (Gym 标准做法，保证实验可复现)
        if seed is not None:
            np.random.seed(seed)

        # 2. 重置全局计数器
        self.current_step = 0
        self.task_registry = {}  # 关键！清空之前的任务记录，防止内存泄漏和ID冲突

        # 3. 重置所有 RSU 的内部状态
        for name, rsu in self.rsus.items():
            rsu.current_memory_used = 0
            rsu.deployed_models = {}  # 清空已部署的模型 (显存归零)
            rsu.task_queue = []  # 清空等待队列

            # 重置统计数据 (用于计算这一轮 Episode 的公平性)
            for m in rsu.stats:
                rsu.stats[m] = {'success': 0, 'total': 0}

        # 4. 重置车辆 (重新生成一波新的车流)
        self.vehicles = {name : Vehicle(conf) for name, conf in Vehicle_CONFIGS.items()}


        # 这一步很重要，能让 Agent 遇到不同的流量分布，提高泛化能力
        # self.vehicles = []
        # for i in range(self.n_vehicles):
        #     # 按照一定比例混合生成不同类型的车
        #     if i < 3:
        #         v_type = 'uav_support'  # 3架无人机
        #     elif i < 8:
        #         v_type = 'autonomous_car'  # 5辆自动驾驶车
        #     else:
        #         v_type = 'connected_car'  # 其余普通车
        #
        #     # 创建车辆对象 (位置随机分布在道路上)
        #     v = Vehicle(v_id=i, v_type=v_type)
        #     self.vehicles.append(v)

        # 5. (可选) 环境预热 / Warm-up
        # 有时候我们希望 Agent 一上来就面临一个“半忙”的状态，而不是全空的状态
        # 如果需要，可以在这里调用几次 _generate_new_request
        # self._random_init_traffic()

        # 6. 生成并返回第一个观测值
        # 注意: 这里只返回 obs，不返回 info (旧版 Gym API)
        # 如果是新版 Gym (v0.26+)，需要返回 (obs, info)
        return self._get_all_observations()

    def _random_init_traffic(self):
        """
        辅助函数：随机生成一些初始任务，让 RSU 不那么闲
        """
        for _ in range(5):  # 随机生成 5 个初始任务
            v = np.random.choice(self.vehicles)
            self._generate_new_request(v)

    def _get_all_observations(self):
        # 获取所有 RSU 的观测
        obs_n = {}
        for name, rsu in self.rsus.items():
            # 1. 获取当前队头任务信息 (如果有)
            if len(rsu.task_queue) > 0:
                task = rsu.task_queue[0]
                # task_feat = [_map_model_id(task.model_type), task.Token_in / 1024, task.deadline]
                task_feat = _map_model_id(task.model_type)
                task_feat.append(task.Token_in / 1024)
                task_feat.append(task.deadline)

                # 2. 自身资源状态
                local_load = [rsu.current_memory_used / rsu.memory_capacity, 0.5]  # 假设计算负载
                model_cache = []
                if rsu.deployed_models.get(task.model_type, 0) == task.precision_req :
                    model_cache = [1]
                    print("部署了模型",task.model_type,"的",task.precision_req ,"精度的")
                else:
                    model_cache = [0]
                    print("没部署模型",task.model_type)
                #与车辆之间的链路状态，还有车辆是靠近自己还是远离自己
                parent_task = self.task_registry.get(task.parent_id)
                vehicle = self.vehicles.get(parent_task.vehicle_id)
                rate = self._calculate_transmission_rate(vehicle,rsu)
                rate_ = [rate/1e9]
                if vehicle.x < rsu.x:
                    direction_ = [1]
                else:
                    direction_ = [0]
                rsu_state = local_load + model_cache + rate_ + direction_

                # 3. 邻居信息 (可以通过 GNN 聚合，这里先放原始值)
                # Todo
                # neighbor_load = [0.0, 0.0]
                neighbor_load = []
                obs = np.array(task_feat + rsu_state + neighbor_load)
                obs_n[name] = obs
            else:
                obs = np.zeros(11)  # 空闲
                obs_n[name] = obs

        return obs_n

    def step(self, actions_dict):
        """
        核心循环: 执行动作 -> 计算时延 -> 计算 Reward
        actions: list of action indices, e.g., [2, 9, 3] 对应 3 个 RSU 的决策
        """
        reward_n = {}  # 每一个RSU（Agent） 的奖励
        done_n = [False] * len(self.rsus)
        info = {}

        # --- Phase 1: 执行每个 RSU 的决策 ---
        for name, action in actions_dict.items():   #actions 是列表，不合适，应该也是字典
            rsu = self.rsus[name]

            print("此时的Rsu是：",rsu.id,"此时的动作是：",action)

            reward = 0
            success = False
            finish_time = math.inf  #完成时间初始为正无穷
            exe_time = 0
            mem_cost = 0
            load_time = 0

            # 如果队列为空，动作无效，给予小惩罚鼓励空闲时休眠或预取
            if len(rsu.task_queue) == 0:
                print("队列为空，跳出循环")
                continue

            # 取出子任务
            sub_task = rsu.task_queue.pop(0)
            print("sub_task",sub_task)
            parent_id = sub_task.parent_id
            parent = self.task_registry[parent_id]  # 获取父任务对象

            # 如果父任务之前已经有别的子任务失败了，这个子任务就没必要做了（剪枝）
            if parent.failed:
                reward_n[name] += -0.2  # 或者给一点微小的负分，惩罚资源浪费

                continue

            # 给这类服务的总数加一
            rsu.stats[sub_task.model_type]['total'] += 1

            # 解码动作
            is_local = action < 3
            is_offload = 3 <= action < 9
            is_drop = action == 9


            precision = (action % 3) + 1  # k=1,2,3

            if is_drop: # 丢弃，不执行，此时父任务直接失败
                # 拒绝惩罚
                reward = -5

            elif is_local:
                # --- 本地RSU直接执行逻辑 ---
                # 1. 检查内存是否足够部署 (公式 C3)
                # 这是一个简化：如果没部署，需要加载时间；如果已部署但精度不同，需要重载
                # 计算 cost (调用 LLMModel)

                # todo 检查 RSU已经部署的模型列表
                if sub_task.model_type in rsu.deployed_models and rsu.deployed_models[sub_task.model_type] == sub_task.precision_req:

                    # 代表对应精度的模型已经部署，可以直接执行
                    exe_time, mem_cost = self.LLMModel[sub_task.model_type].get_resource_costs(rsu, sub_task)

                # 对应精度的模型没有部署，需要加载。
                else:
                    print("没有部署",sub_task.model_type,sub_task.precision_req,"精度的模型")

                    load_time = 0.02  # 先假设加载时间是0.02
                    mem = self.LLMModel[sub_task.model_type].precisions[sub_task.precision_req]['beta'] * self.LLMModel[sub_task.model_type].theta
                    if mem > rsu.memory_capacity - rsu.current_memory_used:
                        # 没有内存部署了，无法执行
                        reward = -2
                    else:
                        exe_time, mem_cost = self.LLMModel[sub_task.model_type].get_resource_costs(rsu, sub_task)
                        total_time = exe_time + load_time
                        mem_cost = mem_cost + mem


                if rsu.current_memory_used + mem_cost > rsu.memory_capacity:
                    # OOM 失败
                    reward = -10  # 严重惩罚
                else:
                    # 成功执行
                    # rsu.current_memory_used += mem_cost
                    # total_time = exe_time  # + 传输时间 (本地忽略 V2I)

                    finish_time = self.current_step  * self.dt + total_time
                    if total_time <= (parent.deadline - self.current_step * self.dt):
                        rsu.stats[sub_task.model_type]['success'] += 1
                        # 奖励 = 基础分 + 剩余时间奖励
                        reward = 10 + (1.0 / total_time)
                    else:
                        reward = -2  # 超时惩罚
                    success = True

            elif is_offload:
                # --- 卸载逻辑 ---
                # 确定目标邻居 ID
                target_rsu_id = (rsu_idx + 1) if action < 6 else (rsu_idx - 1)
                target_rsu_id = max(0, min(target_rsu_id, self.n_rsus - 1))  # 边界处理

                # 计算 RSU 间传输时延
                trans_time = 0.1  # 假设光纤直连 100ms

                # ... 类似本地执行的检查逻辑，但要加上 trans_time ...
                finish_time = self.current_step * self.dt + total_time

                reward = 5  # 卸载成功的奖励通常略低于本地（因为占用了带宽）
                success = True

            reward_n[name] = reward


            if success:
                # 1. 更新父任务状态
                parent.mark_subtask_done(sub_task.model_type, finish_time)

                # 2. 给一点点“进度奖励” (Step Reward)
                # 鼓励 Agent 推进任务，但不代表最终胜利
                reward_n[name] += 1.0

                # 3. 关键检查：是否所有子任务都完成了？ (All-or-Nothing)
                if parent.is_fully_complete():
                    # --- 最终胜利结算 ---

                    # 计算端到端时延 (木桶效应：取决于最慢的那个)
                    total_latency = parent.get_final_latency()

                    if total_latency <= (parent.deadline - parent.created_time):
                        # 成功且未超时：给予巨额奖励
                        # 只有触发这个，才算这辆车的任务真正被 satisfying
                        big_bonus = 50 + (10 / total_latency)

                        # 问题：这个奖励给谁？
                        # 方案1：给当前完成最后一步的 RSU (简单)
                        # 方案2：广播给所有参与过该父任务的 RSU (更符合协作精神，但难实现)
                        # 这里我们简化：给全队共享，或者只给当前 RSU
                        reward_n[name] += big_bonus

                        # 更新公平性统计 (只有这里才算一次有效完成)
                        # self._update_fairness_stats(parent)
                    else:
                        # 虽然做完了但整体超时
                        reward_n[name] -= 5
            else:
                # 子任务失败 (OOM 或单体超时)
                parent.failed = True
                parent.sub_task_status[sub_task['model_type']] = 'failed'
                reward_n[rsu_idx] -= 20  # 严重惩罚，因为搞砸了整个父任务
        # --- Phase 2: 全局公平性修正 (Fairness) ---
        # 你的论文核心: max min g_m
        # 计算所有任务类型的完成率
        global_stats = {}  # 聚合所有 RSU 的统计
        completion_rates = []
        for name, rsu in self.rsus.items():

            for m_type, dat in rsu.stats.items():
                if dat['total'] > 0:
                    rate = dat['success'] / dat['total']
                    completion_rates.append(rate)

        if len(completion_rates) > 0:
            min_completion_rate = min(completion_rates)
            jains_index = self._calc_jains_index(completion_rates)

            # 关键：将公平性加入每个 Agent 的 Reward
            # 这样 Agent 就不仅关注自己的任务，还会关注整体“短板”
            fairness_bonus = 20 * min_completion_rate + 10 * jains_index
            reward_n = [r + fairness_bonus for r in reward_n]

        # --- Phase 3: 环境更新 ---
        self._update_vehicles_and_generate_tasks()
        self.current_step += 1

        if self.current_step >= self.max_steps:
            done_n = [True] * len(self.rsus)

        return self._get_all_observations(), reward_n, done_n, info

    def _update_vehicles_and_generate_tasks(self):
        # 1. 车辆移动
        # 2. 概率生成新任务并放入对应 RSU 的 task_queue
        # 模拟生成一个任务放入 RSU 0
        if np.random.rand() > 0.5:
            new_task = {
                'model_type': 'llama-8b',
                'input_tokens': 512,
                'deadline': self.current_step * self.dt + 2.0
            }
            self.rsus[0].task_queue.append(new_task)

    def _generate_new_request(self, vehicle):
        # 1. 创建父任务
        # 假设这辆车需要同时处理 图像、雷达 和 LLM
        models = ['vit-image', 'radar-former', 'llama-8b']
        precision =[2, 2, 2]
        Token_in = [40, 40, 25]
        Token_out = [30, 5, 40]
        cot_paths = 1
        required_info = {'models': models,
                         'precision': precision,
                         'Token_in':Token_in,
                         'Token_out':Token_out,
                         'deadline': self.current_step * self.dt + 2.0,
                         'cot_paths': cot_paths}

        task_id = f"v{vehicle.id}_t{self.current_step}"
        parent_task = ParentTask(task_id, vehicle.id, required_info)
        parent_task.created_time = self.current_step * self.dt

        # 2. 注册到全局表
        self.task_registry[task_id] = parent_task

        # 3. 拆解子任务并放入最近 RSU 的队列
        nearest_rsu = self._find_nearest_rsu(vehicle)

        for name, sub_task in parent_task.sub_tasks.items():
            nearest_rsu.task_queue.append(sub_task)


    def _calc_jains_index(self, rates):
        # Jain's Fairness Index 公式
        rates = np.array(rates)
        if np.sum(rates) == 0: return 0
        return (np.sum(rates) ** 2) / (len(rates) * np.sum(rates ** 2))

    def _map_model_id(self, name):
        mapping = {'llama-8b': 1, 'vit-image': 2, 'point-lidar': 3, 'radar-former': 4}
        return mapping.get(name, 0)

    def _find_nearest_rsu(self, vehicle):
        """
        根据欧氏距离找到最近的 RSU
        对应论文公式 (5) 中的距离计算 L_{v_i, e}
        """
        nearest_rsu = None
        min_dist = float('inf')

        # 获取车辆当前的 3D 坐标 [x, y, z]
        # 假设 Vehicle 类有 get_position() 方法返回 np.array([x, y, z])
        v_pos = vehicle.get_position()

        for rsu in self.rsus:
            # 计算距离: sqrt((x1-x2)^2 + ...)
            # rsu.pos 在 RSU 初始化时已定义为 np.array
            dist = np.linalg.norm(v_pos - rsu.pos)

            if dist < min_dist:
                min_dist = dist
                nearest_rsu = rsu

        # (可选) 你可以在这里加一个逻辑：
        # 如果 min_dist > 通信半径 (例如 500m)，则返回 None，表示车辆在盲区
        return nearest_rsu

    def _calculate_transmission_rate(self, vehicle, rsu):
        """
        计算 V2I 通信速率 (Shannon's Formula)
        对应论文公式 (4) 和 (5) [cite: 64-66]
        """
        if rsu is None: return 0.0

        # 1. 计算距离 L_{v,e}
        distance = np.linalg.norm(vehicle.get_position() - rsu.pos)
        if distance < 1.0: distance = 1.0  # 避免除以0

        # 2. 计算信道增益 H_{v,e} = A * (distance ** -alpha)
        # alpha: 路损指数 (通常 3 到 4)
        # A: 瑞利衰落系数 (这里简化为常数 * 随机扰动)
        path_loss_exponent = 3.5
        rayleigh_fading = np.random.exponential(scale=1.0)  # 模拟信道随机性
        channel_gain = (distance ** -path_loss_exponent) * rayleigh_fading

        # 3. 计算信噪比 SNR [cite: 65]
        # p_v: 车辆发射功率 (vehicle.tx_power)
        # sigma: 环境噪声功率 (假设 -100dBm -> 1e-13 Watts)
        noise_power = 1e-13
        snr = (vehicle.tx_power * channel_gain) / noise_power

        # 4. 香农公式: R = B * log2(1 + SNR) [cite: 65]
        bandwidth = 10e6  # 假设带宽 10 MHz
        rate = bandwidth * np.log2(1 + snr)

        return rate  # 单位: bits per second (bps)