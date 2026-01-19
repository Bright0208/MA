import numpy as np

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



class LLMModel:
    def __init__(self, model_type, config):
        """
        初始化专家模型 (基于 Transformer 架构)
        :param model_type: 模型名称 (e.g., 'llama-8b', 'vit-image')
        :param config: 模型配置字典 (包含 L, d, df, theta, precisions 等)
        """
        self.model_type = model_type
        self.L = config['L']  # 层数
        self.d = config['d']  # 隐藏层维度 (Hidden dimension)
        self.df = config['df']  # FFN层维度 (通常是 4*d)
        self.theta = config['theta']  # 参数量 (单位: M, 需转换为数量)
        self.precisions = config['precisions']  # 精度配置字典
        self.use_sot = config.get('SoT-use', False)  # 是否使用思维链 (SC-CoT)

    def _calc_prefill_flops(self, P):
        """
        计算 Prefill 阶段 (Prompt 处理) 的计算量
        对应论文公式 (13): A_{m,k}
        :param P: 输入 Token 数量 (Input Length)
        """
        # A = L * (8Pd^2 + 4P^2d + 4Pdd_f)
        # 来源:
        term1 = 8 * P * (self.d ** 2)
        term2 = 4 * (P ** 2) * self.d
        term3 = 4 * P * self.d * self.df
        return self.L * (term1 + term2 + term3)

    def _calc_decoding_flops(self, P, n):
        """
        计算 Decoding 阶段 (自回归生成) 的计算量
        对应论文公式 (15): G_{m,k}
        :param P: 输入 Token 数量
        :param n: 输出 Token 数量 (Output Length)
        """
        if n == 0: return 0

        # G = L * [2dn^2 + (8d^2 + 4dd_f + 4dP + 2d)n]
        # 来源:
        term1 = 2 * self.d * (n ** 2)
        coeff_n = 8 * (self.d ** 2) + 4 * self.d * self.df + 4 * self.d * P + 2 * self.d
        term2 = coeff_n * n
        return self.L * (term1 + term2)

    def get_resource_costs(self, rsu, sub_task):
        """
        获取当前任务的资源消耗：执行时间 与 内存占用

        :param k: 精度等级 (1=FP16, 2=INT8, 3=INT4)
        :param input_tokens: 输入长度
        :param output_tokens: 输出长度
        :param rsu_compute_cap: RSU 算力 C_e_max (FLOPs/s)
        :param cot_paths: SC-CoT 推理链条数 J (仅当 use_sot=True 时生效)
        :return: (execution_time_sec, memory_usage_bytes)
        """
        k = sub_task.precision_req
        input_tokens = sub_task.Token_in
        output_tokens = sub_task.Token_out
        rsu_compute_cap = rsu.compute_capacity
        cot_paths = sub_task.cot_paths


        # 1. 获取精度属性
        prec_props = self.precisions[k]
        beta = prec_props['beta']  # 字节/参数
        speed_factor = prec_props['speed_factor']  # 硬件加速比

        # 2. 计算内存占用 (Static Model + Dynamic KV Cache)
        # 静态内存: R_model = theta * beta
        mem_static = (self.theta * 1e6) * beta

        # 动态 KV Cache: R_KV = 2 * d * L * (P + n) * beta
        total_tokens = input_tokens + output_tokens
        mem_kv = 2 * self.d * self.L * total_tokens * beta

        total_memory = mem_static + mem_kv

        # 3. 计算计算量 (FLOPs)
        flops_prefill = self._calc_prefill_flops(input_tokens)
        flops_decoding = self._calc_decoding_flops(input_tokens, output_tokens)

        # 如果是生成式任务且启用了 SC-CoT，总 FLOPs 需要翻倍吗？
        # 论文公式 (10) 指出是时延累加
        # 单次推理总 FLOPs
        single_chain_flops = flops_prefill + flops_decoding

        # 4. 计算执行时间
        # Time = FLOPs / (Capacity * SpeedFactor)
        # 来源:
        eff_capacity = rsu_compute_cap * speed_factor

        if self.use_sot and cot_paths > 1:
            # 只有语言模型才会有多条思维链
            # 公式 (10): sum(tau_single)
            total_flops = single_chain_flops * cot_paths
        else:
            total_flops = single_chain_flops

        execution_time = total_flops / eff_capacity

        return execution_time, mem_static, mem_kv



if __name__ == '__main__':
    # 假设 RSU 算力: NVIDIA Orin 级别 (约 200 TFLOPs INT8 -> 100 TFLOPs FP16)
    RSU_CAPACITY_FP16 = 100e12  # 100 TFLOPs
    # 注意: 这里的 capacity 是基准(FP16)算力，具体执行时会除以 capacity * speed_factor

    # 实例化模型库
    model_library = {name: LLMModel(name, conf) for name, conf in MODELS_CONFIG.items()}
    print(model_library)

    def test_simulation(model_name, k, p_len, o_len, j_paths=1):
        model = model_library[model_name]
        exe_time, mem_usage = model.get_resource_costs(
            k=k,
            input_tokens=p_len,
            output_tokens=o_len,
            rsu_compute_cap=RSU_CAPACITY_FP16,
            cot_paths=j_paths
        )

        mem_gb = mem_usage / (1024 ** 3)
        print(f"[{model_name}] 精度: {model.precisions[k]['name']}")
        print(f"  - 输入/输出: {p_len}/{o_len} Tokens")
        if model.use_sot:
            print(f"  - SC-CoT路径: {j_paths} 条")
        print(f"  - 耗时: {exe_time:.4f} s")
        print(f"  - 显存: {mem_gb:.2f} GB")
        print("-" * 30)


    # --- 运行测试案例 ---

    print("=== 场景 1: 复杂语言推理 (LLM) ===")
    # Llama-8B, FP16, 长文本生成, 3条思维链
    test_simulation('llama-8b', k=1, p_len=512, o_len=256, j_paths=3)

    print("=== 场景 2: 语言推理降级 (LLM) ===")
    # Llama-8B, INT4, 同样任务
    test_simulation('llama-8b', k=3, p_len=512, o_len=256, j_paths=3)

    print("=== 场景 3: 视觉感知 (ViT) ===")
    # ViT, FP16, 图片Patch输入(256), 输出很少(10)
    test_simulation('vit-image', k=1, p_len=256, o_len=10)

    print("=== 场景 4: 雷达探测 (Radar) ===")
    # Radar, FP16, 输入少, 几乎无输出Decoding
    test_simulation('radar-former', k=1, p_len=64, o_len=5)

    print("=== 场景 5: 点云探测 (Lidar) ===")
    # Radar, FP16, 输入少, 几乎无输出Decoding
    test_simulation('point-lidar', k=1, p_len=64, o_len=5)