import numpy as np

from env.task import Task

class Vehicle:
    def __init__(self, config):
        self.id = config['id']
        self.original_x = config['x']
        self.x = config['x']
        self.y = config['y']
        self.z = config['z']
        self.speed = config['speed']
        self.task = Task(self.id)

    def move(self):
        self.x += self.speed

    def get_state(self):
        state = []
        vehicle_state = [ self.x, self.y, self.z, self.speed]
        task_state = self.task.generate_task()
        state.append(vehicle_state)
        state.extend(task_state)

        return state

if __name__ == '__main__':
    vehicle = Vehicle(1, 12, 0, 0, 5)
    state = vehicle.get_state()
    print(state)
    state = np.concatenate(state)
    print(state.shape)