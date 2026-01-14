import numpy as np



class Environment:
    def __init__(self,Model,Rsu,Vehicle):
        self.rsus = Rsu
        self.vehicles = Vehicle
        self.LLMModel  = Model

    def determine_vehicle_ownership(self):
        for r_name, rsu in self.rsus.items():
            for v_name ,vehicle in self.vehicles.items():
                rsu.is_vehicle_in_range(vehicle)

    def agent_ids(self):
        return [rsu.id for rsu in self.rsus]

    def get_agent_numbers(self):
        return len(self.rsus)

    def get_action_dim(self):
        # 暂时 5
        return 5

    def reset(self):
        state = []

        for rsu in self.rsus:
            #reset RSU
            rsu.model = np.zeros((5, 3))  # 模型种类5种，精度3种
            # print("RSU重置")
            #reset 车辆
            for vehicle in rsu.vehicles:
                if vehicle is not None:
                    vehicle.x = vehicle.original_x
                    # print("车辆重置")
            rsu_state = rsu.get_state()
            # print(rsu_state)
            state.append(rsu_state)
        return state


