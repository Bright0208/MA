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
        for v_name, vehicle in self.vehicles.items():
            self._generate_new_request(vehicle)

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
                    # print("部署了模型",task.model_type,"的",task.precision_req ,"精度的")
                else:
                    model_cache = [0]
                    # print("没部署模型",task.model_type)
                #与车辆之间的链路状态，还有车辆是靠近自己还是远离自己
                parent_task = self.task_registry.get(task.parent_id)
                # print("parent_task:",parent_task.vehicle_id)
                vehicle = self.vehicles.get(f"V_{parent_task.vehicle_id}")
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
        reward_n = {}
        done_n = {}
        for name, rsu in self.rsus.items():
            reward_n[name] = 0  # 每一个RSU（Agent） 的奖励
            done_n[name] = False
        info = {}

        # --- Phase 1: 执行每个 RSU 的决策 ---
        for name, action_index in actions_dict.items():   #actions 是列表，不合适，应该也是字典

            # 1. 在局部变量里转换，不要写回 actions_dict！
            # 只要不执行 actions_dict[rsu_name] = ...，外面就不会变
            if isinstance(action_index, np.ndarray):
                action = np.argmax(action_index)
            else:
                action = action_index  # 防止已经是int的情况

            rsu = self.rsus[name]

            print("此时的Rsu是：",rsu.id,"此时的动作是：",action)

            reward = 0
            success = False
            finish_time = math.inf  #完成时间初始为正无穷
            total_time = 0
            mem_cost = 0
            load_time = 0

            # 如果队列为空，动作无效，给予小惩罚鼓励空闲时休眠或预取
            if len(rsu.task_queue) == 0:
                print("队列为空，跳出循环")
                continue

            # 取出子任务
            sub_task = rsu.task_queue.pop(0)
            print("此时的sub_task是：",sub_task)
            parent_id = sub_task.parent_id
            parent = self.task_registry[parent_id]  # 获取父任务对象

            # 如果父任务之前已经有别的子任务失败了，这个子任务就没必要做了（剪枝）
            if parent.failed:
                reward_n[name] -= 0.2  # 或者给一点微小的负分，惩罚资源浪费

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
                    total_time = exe_time
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
                    if finish_time <= (parent.deadline - self.current_step * self.dt):
                        rsu.stats[sub_task.model_type]['success'] += 1
                        # 奖励 = 基础分 + 剩余时间奖励
                        reward = 10 + (1.0 / total_time)
                    else:
                        reward = -2  # 超时惩罚
                    success = True

            elif is_offload:
                # --- 卸载逻辑 ---
                # 先判断自身是否有邻居

                # 确定目标邻居 ID
                target_rsu_id = (rsu.id - 1) if action < 6 else (rsu.id + 1)
                # target_rsu_id = max(0, min(target_rsu_id, self.n_rsus - 1))  # 边界处理

                if target_rsu_id == -1  or target_rsu_id == 6: #没有左邻居、 没有右邻居 惩罚
                    reward = -5
                else:

                    # 计算 RSU 间传输时延
                    trans_time = 0.1  # 假设光纤直连 100ms

                    exe_time, mem_cost = self.LLMModel[sub_task.model_type].get_resource_costs(rsu, sub_task)

                    total_time = exe_time + trans_time
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
                parent.sub_task_status[sub_task.model_type] = 'failed'
                reward_n[name] -= 20  # 严重惩罚，因为搞砸了整个父任务
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
            # reward_n = [r + fairness_bonus for name, r in reward_n.items()]
            for name, r in reward_n.items():
                reward_n[name] = r + fairness_bonus

        # --- Phase 3: 环境更新 ---
        for name, v in self.vehicles.items():
            self._update_vehicles(v)
            self._generate_new_request(v)
        self.current_step += 1

        if self.current_step >= self.max_steps:
            for name, done in done_n.items():
                done_n[name] = True

        return self._get_all_observations(), reward_n, done_n, info


    def _update_vehicles(self, vehicle):
        vehicle.move()

    def _generate_new_request(self, vehicle):
        """
        为车辆随机生成一个多模态父任务
        """
        # 1. 确定该任务包含哪些模态 (专家模型)
        # 获取所有可用的模型类型 (llama-8b, vit-image, etc.)
        all_model_types = list(self.LLMModel.keys())

        # 随机决定这个父任务需要协同几个专家 (例如 1 到 3 个)
        # 复杂任务可能需要 3 个模型 (如: 图像 + 雷达 + LLM 融合)
        num_models = np.random.randint(1, len(all_model_types) + 1)

        # 无放回抽样，选出具体的模型列表
        selected_models = np.random.choice(all_model_types, num_models, replace=False).tolist()

        # 2. 为每个选定的模型生成具体的任务参数
        token_in_list = []
        token_out_list = []
        precision_req_list = []  # 这里指任务产生时的"期望精度"或"原始精度"

        # 设定 SC-CoT 的路径数 (仅对 LLM 有效) [cite: 108]
        # 如果任务不含 LLM，这个值为 1；如果有 LLM，随机生成 1~3 条推理链
        cot_paths = 1

        for m_name in selected_models:
            # --- A. 针对 LLM (llama-8b) 的特殊生成逻辑 ---
            if 'llama' in m_name:
                # LLM 输入通常较长 (Prompt)，输出也较长 (推理/生成)
                t_in = np.random.randint(128, 1024)
                t_out = np.random.randint(32, 256)
                # 随机决定推理链条数 (模拟复杂推理任务的不确定性)
                cot_paths = np.random.randint(1, 4)

                # --- B. 针对感知模型 (ViT, Lidar, Radar) 的生成逻辑 ---
            else:
                # 感知模型输入通常是固定的 Patch/Point 数量，这里模拟为 Token 数
                # ViT/Lidar 输入大，Radar 输入小
                if 'radar' in m_name:
                    t_in = np.random.randint(32, 128)
                else:
                    t_in = np.random.randint(196, 512)  # 如 ViT patch 数

                # 感知模型的输出通常很短 (分类标签或坐标)，计算量主要在 Prefill
                t_out = np.random.randint(5, 20)

            # 随机生成该子任务的最低精度要求 (1:FP16, 2:INT8, 3:INT4)
            # 模拟：有些关键安全任务要求 FP16，有些容忍 INT4
            prec = np.random.choice([1, 2, 3], p=[0.2, 0.5, 0.3])

            token_in_list.append(t_in)
            token_out_list.append(t_out)
            precision_req_list.append(prec)

        # 3. 生成截止时间 (Deadline)
        # 任务越复杂 (Token多)，给的容忍时间通常稍微宽裕一点，但也有随机扰动
        # 基础容忍时间 0.5s ~ 3.0s
        tolerance = np.random.uniform(0.5, 3.0)
        deadline = self.current_step * self.dt + tolerance

        # 4. 打包所需信息字典 (对应 Image 2 的格式)
        required_info = {
            'models': selected_models,
            'precision': precision_req_list,  # 注意：这是需求，Agent决策时可能会调整
            'Token_in': token_in_list,
            'Token_out': token_out_list,
            'deadline': deadline,
            'cot_paths': cot_paths  # [cite: 114]
        }

        # 5. 实例化父任务并注册
        task_id = f"v{vehicle.id}_t{self.current_step}"
        parent_task = ParentTask(task_id, vehicle.id, required_info)
        parent_task.created_time = self.current_step * self.dt

        # 6. 注册到全局表
        self.task_registry[task_id] = parent_task

        # 7. (关键步骤) 将子任务分发到最近 RSU 的等待队列
        # 这样 Agent 在 step() 时才能看到任务
        nearest_rsu = self._find_nearest_rsu(vehicle)

        for name, sub_task in parent_task.sub_tasks.items():
            nearest_rsu.task_queue.append(sub_task)

        # if nearest_rsu:
        #     # 遍历生成的模型列表，拆解为子任务
        #     for idx, model_type in enumerate(selected_models):
        #         sub_task = {
        #             'parent_id': task_id,
        #             'vehicle_id': vehicle.id,
        #             'model_type': model_type,
        #             'input_tokens': token_in_list[idx],
        #             'output_tokens': token_out_list[idx],
        #             'req_precision': precision_req_list[idx],  # 记录原始需求
        #             'deadline': deadline,
        #             'cot_paths': cot_paths if 'llama' in model_type else 1,
        #             'timestamp': self.current_step * self.dt
        #         }
        #         nearest_rsu.task_queue.append(sub_task)

        # return task_id



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

        for name, rsu in self.rsus.items():
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