import random
from config import MODELS_CONFIG

class ParentTask:
    def __init__(self, task_id, vehicle_id, required_info):
        """
        :论文里叫 psi ψ^{v_i}(t)
        :param task_id: 唯一任务ID
        :param required_models: 该任务包含哪些模态 (e.g., ['vit-image', 'point-lidar'])
        """
        self.task_id = task_id
        self.vehicle_id = vehicle_id
        self.deadline = required_info["deadline"]
        self.created_time = 0  # 会在生成时赋值


        self.sub_tasks = {}
        for i, model in enumerate(required_info['models']):
            self.sub_tasks[model] = SubTask(self.task_id,model,
                                            required_info['precision'][i],
                                            required_info['Token_in'][i],
                                            required_info['Token_out'][i],
                                            required_info['cot_paths'],
                                            self.deadline)

        # 记录需要完成的子任务状态
        # 格式: {'vit-image': 'pending', 'point-lidar': 'pending'}
        self.sub_task_status = {m: 'pending' for m in required_info['models']}

        # 记录每个子任务的完成时间 (用于计算最终的最大时延)
        self.finish_times = {}

        # 标记是否失败 (只要有一个子任务失败，整个大任务就失败)
        self.failed = False


    def mark_subtask_done(self, model_name, finish_time):
        self.sub_task_status[model_name] = 'success'
        self.finish_times[model_name] = finish_time

    def is_fully_complete(self):
        # 只有当所有子任务都 success 时才算完成
        return all(s == 'success' for s in self.sub_task_status.values())

    def get_final_latency(self):
        # 整个任务的时延 = 最慢那个子任务的完成时间 - 创建时间
        if not self.finish_times: return 0
        last_finish_time = max(self.finish_times.values())
        return last_finish_time - self.created_time



class SubTask:
    """
    子任务 ϕ_{v_i, m_j}(t)
    对应论文  的任务模型部分
    """
    def __init__(self,parent_id,model_type,precision_req,Token_in,Token_out,cot_paths,deadline):
        """
        :param vehicle_id: 车辆 id (i)
        :param model_type: 模型类型 mj
        :param precision_req: 子任务要求的模型精度 kvi,mj(t)
        :param token_input: 输入 Token 数量
        :param token_output: 输出 Token 最大数量
        :param delay_req: 时延要求 dvi,mj(t)
        """
        self.parent_id = parent_id
        self.model_type = model_type
        self.precision_req = precision_req
        self.Token_in = Token_in
        self.Token_out = Token_out
        self.cot_paths = cot_paths
        self.deadline = deadline


    # def generate_task(self):
    #     """
    #     返回一个状态数组即可
    #     :return:
    #     """
    #
    #     token_input_range = (32, 128)
    #     delay_req_range = (0.2, 1.0)
    #     subtasks = []
    #     for model_id in model_id:
    #         # 80% 概率请求该模态
    #         request_flag = random.random() < 0.7
    #         if not request_flag:
    #             #如果没选的话就补零
    #             # print("没选",model_id,"号模型，选择补0")
    #             st = [0,0,0,0,0,0]
    #             subtasks.append(st)
    #             continue
    #
    #         precision_req = random.choice(precision_choices)
    #         token_input = random.randint(*token_input_range)
    #         delay_req = random.uniform(*delay_req_range)
    #         max_output_tokens = max_output_tokens_list[model_id]
    #         st = [self.vehicle_id, model_id, precision_req, token_input, max_output_tokens,delay_req]
    #         subtasks.append(st)
    #     return subtasks

if __name__ == '__main__':
    # 测试生成父任务
    # 任务id 0，车辆id 0，时延要求 1秒，请求4个模型 ['llama-8b','vit-image', 'point-lidar','radar-former']
    models = ['vit-image', 'radar-former', 'llama-8b']
    precision = [2, 2, 2]
    Token_in = [40, 40, 25]
    Token_out = [30, 5, 40]
    required_info = {'models': models,
                     'precision': precision,
                     'Token_in': Token_in,
                     'Token_out': Token_out,
                     'deadline': 10 + 2.0}

    parentTask =  ParentTask(0,0,required_info)
    parentTask.created_time = 2
    print(parentTask)