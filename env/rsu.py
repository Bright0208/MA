import numpy as np
from math import sqrt

from env.vehicle import Vehicle


class Rsu:
    def __init__(self, config):
        self.type = config['type']
        self.id = config['id']
        self.x = config['x']
        self.y = config['y']
        self.z = config['z']
        self.memory_capacity = config['memory_capacity']
        self.compute_capacity = config['compute_capacity']
        self.range_radius = 100
        self.vehicles = [None] * 20
        # 状态记录
        self.current_memory_used = 0
        self.deployed_models = {}  # 模型种类4种，精度3种
        self.task_queue = []  # 等待处理的任务列表
        self.pos = np.array([self.x, self.y, self.z])

        # 统计数据 (用于计算公平性 g_m)
        self.stats = {m: {'success': 0, 'total': 0} for m in ['llama-8b', 'vit-image', 'point-lidar', 'radar-former']}

        self.last_exec_status = 0.0

    def add_vehicle(self, vehicle):

        for i, value in enumerate(self.vehicles):
            if value is None:
                self.vehicles[i] = vehicle
                return True
        return False  # 20个位置已满，返回False

    def del_vehicle(self, vehicle):
        for i, value in enumerate(self.vehicles):
            if value is vehicle:
                self.vehicles[i] = None
                # print("车辆已删除")
                return True
        return False


    def is_vehicle_in_range(self, vehicle: Vehicle) -> bool:
        """判断车辆是否在感知范围内，并维护 vehicles_in_range 列表。"""
        r = self.range_radius
        dx = self.x - vehicle.x
        dy = self.y - vehicle.y

        d = sqrt(dx * dx + dy * dy  )

        in_range = d <= r
        if in_range:
            self.add_vehicle(vehicle)
        else:
            self.del_vehicle(vehicle)
        return in_range




    def get_state(self):
        state = []
        rsu_state = [self.x, self.y, self.z]
        state.extend(rsu_state)

        for i, vehicle in enumerate(self.vehicles):
            if vehicle is not None:
                veh_state = vehicle.get_state()
                if veh_state is None:
                    raise ValueError("vehicle.get_state() 返回 None，请检查 Vehicle 类")
                state.extend(np.concatenate(veh_state))
            else:
                state.extend([0] * 34)
        return state



if __name__ == '__main__':
    rsu = Rsu(1, 2, 0, 15)
    vehicle1 = Vehicle(0, 12, 0, 0, 5)
    vehicle2 = Vehicle(1, 52, 0, 0, 5)
    rsu.add_vehicle(vehicle1)
    rsu.add_vehicle(vehicle2)
    state = rsu.get_state()
    print(state)
    print(np.array(state))
    print(np.array(state).shape)