import numpy as np
import torch

from env.environment import Environment
from config import device, MODELS_CONFIG, RSU_CONFIGS, Vehicle_CONFIGS
from env.llm_model import LLMModel
from env.rsu import Rsu
from env.vehicle import Vehicle
from net.m_agent import Magent

# 实例化模型库
model_library = {name: LLMModel(name, conf) for name, conf in MODELS_CONFIG.items()}
# 实例化RSU
Rsu_dict = {name :Rsu(conf) for name, conf in RSU_CONFIGS.items()}
# TODO ：构建邻居
# 实例化车辆
Vehicle_dict = {name : Vehicle(conf) for name, conf in Vehicle_CONFIGS.items()}

if __name__ == "__main__":
    env = Environment(model_library, Rsu_dict, Vehicle_dict)

    env.determine_vehicle_ownership()

    n_agent = env.get_agent_numbers()   #获得智能体数量

    state = env.reset()
    state_dim = len(state[0])           #通过state获得state_dim
    action_dim = env.get_action_dim()   #获得action_dim
    maddpg = Magent(n_agent,state_dim,action_dim)

    for episode in range(3):
        state = env.reset()
        for step in range(100):
            for i  in range(len(state)):
                state_tensor = torch.tensor(state[i],dtype=torch.float,device=device).unsqueeze(0)
                # print(state_tensor)
                action = maddpg.select_action(i, state_tensor)
                print(action)



