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
Vehicle_CONFIGS = {
    'V_0': {
        'id':0, 'x':0, 'y':0, 'z':1.5, 'speed':'4','tx_power':0.5
    },
    'V_1': {
        'id': 0, 'x': 10, 'y': 0, 'z': 1.5, 'speed': '4','tx_power':0.5
    },
    'V_2': {
        'id': 0, 'x': 30, 'y': 0, 'z': 1.5, 'speed': '4','tx_power':0.5
    },
    'V_3': {
        'id': 0, 'x': 50, 'y': 0, 'z': 1.5, 'speed': '4','tx_power':0.5
    },
    'V_4': {
        'id': 0, 'x': 60, 'y': 0, 'z': 1.5, 'speed': '4','tx_power':0.5
    },
    'V_5': {
        'id': 0, 'x': 100, 'y': 0, 'z': 1.5, 'speed': '4','tx_power':0.5
    },
    'V_6': {
        'id': 0, 'x': 120, 'y': 0, 'z': 1.5, 'speed': '4','tx_power':0.5
    },
    'V_7': {
        'id': 0, 'x': 150, 'y': 0, 'z': 1.5, 'speed': '4','tx_power':0.5
    },
    'V_8': {
        'id': 0, 'x': 180, 'y': 0, 'z': 1.5, 'speed': '4','tx_power':0.5
    },
    'V_9': {
        'id': 0, 'x': 200, 'y': 0, 'z': 1.5, 'speed': '4','tx_power':0.5
    },
    'V_10': {
        'id': 0, 'x': 250, 'y': 0, 'z': 1.5, 'speed': '4','tx_power':0.5
    },
    'V_11': {
        'id': 0, 'x': 300, 'y': 0, 'z': 1.5, 'speed': '4','tx_power':0.5
    },
    'V_12': {
        'id': 0, 'x': 320, 'y': 0, 'z': 1.5, 'speed': '4','tx_power':0.5
    },
    'V_13': {
        'id': 0, 'x': 350, 'y': 0, 'z': 1.5, 'speed': '4','tx_power':0.5
    },
    'V_14': {
        'id': 0, 'x': 380, 'y': 0, 'z': 1.5, 'speed': '4','tx_power':0.5
    },
    'V_15': {
        'id': 0, 'x': 400, 'y': 0, 'z': 1.5, 'speed': '4','tx_power':0.5
    },
    'V_16': {
        'id': 0, 'x': 420, 'y': 0, 'z': 1.5, 'speed': '4','tx_power':0.5
    },
    'V_17': {
        'id': 0, 'x': 480, 'y': 0, 'z': 1.5, 'speed': '4','tx_power':0.5
    },
    'V_18': {
        'id': 0, 'x': 510, 'y': 0, 'z': 1.5, 'speed': '4','tx_power':0.5
    },
    'V_19': {
        'id': 0, 'x': 550, 'y': 0, 'z': 1.5, 'speed': '4','tx_power':0.5
    },
    'V_20': {
        'id': 0, 'x': 600, 'y': 0, 'z': 1.5, 'speed': '4','tx_power':0.5
    },
    'V_21': {
        'id': 0, 'x': 650, 'y': 0, 'z': 1.5, 'speed': '4','tx_power':0.5
    },
    'V_22': {
        'id': 0, 'x': 670, 'y': 0, 'z': 1.5, 'speed': '4','tx_power':0.5
    },
    'V_23': {
        'id': 0, 'x': 690, 'y': 0, 'z': 1.5, 'speed': '4','tx_power':0.5
    },
    'V_24': {
        'id': 0, 'x': 700, 'y': 0, 'z': 1.5, 'speed': '4','tx_power':0.5
    },
    'V_25': {
        'id': 0, 'x': 720, 'y': 0, 'z': 1.5, 'speed': '4','tx_power':0.5
    },
    'V_26': {
        'id': 0, 'x': 750, 'y': 0, 'z': 1.5, 'speed': '4','tx_power':0.5
    },
    'V_27': {
        'id': 0, 'x': 780, 'y': 0, 'z': 1.5, 'speed': '4','tx_power':0.5
    },
    'V_28': {
        'id': 0, 'x': 800, 'y': 0, 'z': 1.5, 'speed': '4','tx_power':0.5
    },
    'V_29': {
        'id': 0, 'x': 810, 'y': 0, 'z': 1.5, 'speed': '4','tx_power':0.5
    },
    'V_30': {
        'id': 0, 'x': 850, 'y': 0, 'z': 1.5, 'speed': '4','tx_power':0.5
    },
    'V_31': {
        'id': 0, 'x': 860, 'y': 0, 'z': 1.5, 'speed': '4','tx_power':0.5
    },
    'V_32': {
        'id': 0, 'x': 900, 'y': 0, 'z': 1.5, 'speed': '4','tx_power':0.5
    },
    'V_33': {
        'id': 0, 'x': 910, 'y': 0, 'z': 1.5, 'speed': '4','tx_power':0.5
    },
    'V_34': {
        'id': 0, 'x': 950, 'y': 0, 'z': 1.5, 'speed': '4','tx_power':0.5
    },
}