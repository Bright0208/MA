import numpy as np
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#网络部分
Actor_deploy_input_dim = 683
Actor_deploy_action_dim = 5
hidden_dim = 1024




#强化学习部分
gamma = 0.95
tau = 0.01
lr_actor = 1e-4
lr_critic = 2e-4

#buffer 部分
buffer_size = 50000
batch_size = 128



# 模拟不同精度的模型属性
MODELS_CONFIG = {
    # 1. 语言专家模型
    'llama-8b': {
        'L': 32, 'd': 4096, 'df': 11008, 'theta': 8000, # 参数量 8B (单位: M)
        'precisions': {
            1: {'name': 'FP16', 'beta': 2.0, 'speed_factor': 1.0, 'acc': 0.95}, # 高精度 [cite: 165]
            2: {'name': 'INT8', 'beta': 1.0, 'speed_factor': 2.0, 'acc': 0.92}, # 中精度 [cite: 165]
            3: {'name': 'INT4', 'beta': 0.5, 'speed_factor': 3.5, 'acc': 0.88}  # 低精度 [cite: 165]
        },
        'SoT-use': True  # 语言模型通常涉及复杂推理，开启 SC-CoT 逻辑 [cite: 108, 185]
    },

    # 2. 图像专家模型 (基于 Vision Transformer - ViT-Large)
    'vit-image': {
        'L': 24, 'd': 1024, 'df': 4096, 'theta': 300,  # 参数量约 300M
        'precisions': {
            1: {'name': 'FP16', 'beta': 2.0, 'speed_factor': 1.0, 'acc': 0.94},
            2: {'name': 'INT8', 'beta': 1.0, 'speed_factor': 2.0, 'acc': 0.91},
            3: {'name': 'INT4', 'beta': 0.5, 'speed_factor': 3.8, 'acc': 0.84}
        },
        'SoT-use': False # 感知类任务通常不使用思维链 [cite: 114]
    },

    # 3. 点云专家模型 (基于 Point-Transformer)
    'point-lidar': {
        'L': 24, 'd': 768, 'df': 3072, 'theta': 150,   # 参数量约 150M
        'precisions': {
            1: {'name': 'FP16', 'beta': 2.0, 'speed_factor': 1.0, 'acc': 0.92},
            2: {'name': 'INT8', 'beta': 1.0, 'speed_factor': 2.2, 'acc': 0.88},
            3: {'name': 'INT4', 'beta': 0.5, 'speed_factor': 4.0, 'acc': 0.81}
        },
        'SoT-use': False
    },

    # 4. 雷达专家模型 (基于 RadarFormer - 轻量级 Transformer)
    'radar-former': {
        'L': 12, 'd': 512, 'df': 2048, 'theta': 50,    # 参数量约 50M
        'precisions': {
            1: {'name': 'FP16', 'beta': 2.0, 'speed_factor': 1.0, 'acc': 0.90},
            2: {'name': 'INT8', 'beta': 1.0, 'speed_factor': 2.5, 'acc': 0.86},
            3: {'name': 'INT4', 'beta': 0.5, 'speed_factor': 4.5, 'acc': 0.78}
        },
        'SoT-use': False
    }
}

# 模拟不同等级的 RSU 硬件配置
RSU_CONFIGS = {
    'Rsu_0': {
        'type':'standard_rsu',
        'id':0, 'x':0, 'y':0, 'z':6,
        'memory_capacity': 48 * (1024**3),  # 48 GB (e.g., NVIDIA L40)      standard_rsu
        'compute_capacity': 180e12          # 180 TFLOPs (FP16 Tensor Core)
    },
    'Rsu_1': {
        'type':'premium_rsu',
        'id': 1, 'x': 200, 'y': 0, 'z': 6,
        'memory_capacity': 64 * (1024**3),  # 64 GB (e.g., Jetson Orin Cluster)  premium_rsu
        'compute_capacity': 275e12
    },
    'Rsu_2': {
        'type':'legacy_rsu',
        'id': 2, 'x': 400, 'y': 0, 'z': 6,
        'memory_capacity': 24 * (1024**3),  # 24 GB (e.g., NVIDIA 3090/4090)   legacy_rsu
        'compute_capacity': 80e12           # 算力较弱，更容易拥堵
    },
    'Rsu_3': {
        'type':'standard_rsu',
        'id': 3, 'x': 600, 'y': 0, 'z': 6,
        'memory_capacity': 48 * (1024 ** 3),  # 48 GB (e.g., NVIDIA L40)      standard_rsu
        'compute_capacity': 180e12  # 180 TFLOPs (FP16 Tensor Core)
    },
    'Rsu_4': {
        'type':'premium_rsu',
        'id': 4, 'x': 800, 'y': 0, 'z': 6,
        'memory_capacity': 64 * (1024 ** 3),  # 64 GB (e.g., Jetson Orin Cluster)  premium_rsu
        'compute_capacity': 275e12
    },
    'Rsu_5': {
        'type':'legacy_rsu',
        'id': 5, 'x': 1000, 'y': 0, 'z': 6,
        'memory_capacity': 24 * (1024 ** 3),  # 24 GB (e.g., NVIDIA 3090/4090)   legacy_rsu
        'compute_capacity': 80e12  # 算力较弱，更容易拥堵
    },
}

# 模拟不同车辆的配置
# --- 车辆生成配置 ---
NUM_VEHICLES = 30  # 你想要多少辆车
ROAD_LENGTH = 1000  # 道路长度 (米)
MIN_SPEED = 1.0  # 最小速度 (m/s)
MAX_SPEED = 4.0  # 最大速度 (m/s)


def generate_random_vehicle_configs(num_vehicles=NUM_VEHICLES):
    """
    随机生成车辆配置字典
    """
    configs = {}
    for i in range(num_vehicles):
        v_id = f"V_{i}"

        # 随机位置 (0 到 1000米之间)
        random_x = np.random.uniform(0, ROAD_LENGTH)

        # 随机速度 (5 到 20 m/s)
        random_speed = np.random.uniform(MIN_SPEED, MAX_SPEED)

        # 随机方向 (可选：目前假设都向右走，速度为正)
        # 如果需要双向车道，可以随机把 speed 设为负数

        configs[v_id] = {
            'id': i,
            'x': random_x,
            'y': 0,  # 假设单车道，y=0
            'z': 1.5,
            'speed': random_speed,
            'tx_power': 0.5,
            # 可以添加更多随机属性，例如车辆类型偏好等
        }
    return configs