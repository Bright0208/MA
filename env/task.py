import random

model_dict = {"model_id_list": [0, 1, 2, 3, 4], "precision_choices": [1, 2, 3],
              "max_output_tokens_list": [20, 40, 45, 40, 20]}

class Task:
    """
    子任务 ϕ_{v_i, m_j}(t)
    对应论文  的任务模型部分
    """
    def __init__(self,vehicle_id):
        """
        :param vehicle_id: 车辆 id (i)
        :param model_type: 模型类型 mj
        :param precision_req: 子任务要求的模型精度 kvi,mj(t)
        :param token_input: 输入 Token 数量
        :param token_output: 输出 Token 最大数量
        :param delay_req: 时延要求 dvi,mj(t)
        """
        self.vehicle_id = vehicle_id


    def generate_task(self):
        """
        返回一个状态数组即可
        :return:
        """
        model_id = model_dict["model_id_list"]
        precision_choices = model_dict["precision_choices"]
        max_output_tokens_list = model_dict["max_output_tokens_list"]

        token_input_range = (32, 128)
        delay_req_range = (0.2, 1.0)
        subtasks = []
        for model_id in model_id:
            # 80% 概率请求该模态
            request_flag = random.random() < 0.7
            if not request_flag:
                #如果没选的话就补零
                # print("没选",model_id,"号模型，选择补0")
                st = [0,0,0,0,0,0]
                subtasks.append(st)
                continue

            precision_req = random.choice(precision_choices)
            token_input = random.randint(*token_input_range)
            delay_req = random.uniform(*delay_req_range)
            max_output_tokens = max_output_tokens_list[model_id]
            st = [self.vehicle_id, model_id, precision_req, token_input, max_output_tokens,delay_req]
            subtasks.append(st)
        return subtasks

if __name__ == '__main__':
    task = Task(vehicle_id=1)
    task.generate_task()

