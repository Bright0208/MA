import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Actor_deploy_input_dim ,Actor_deploy_action_dim,hidden_dim


class Actor_deploy(nn.Module):
    """
    卸载 Actor：专注“当前子任务卸载到哪儿、用什么精度”。
    建议 state（由环境准备）：
      - 当前任务特征：model_type、precision_req、token_input、token_output、delay_req
      - 候选 RSU 状态：距离/信道估计、可用内存、KV cache、队列长度、排队 FLOPs 占比
      - 当前 RSU 已部署模型列表（或 one-hot），可用精度选项
      - 可选：车辆速度/位置、历史时延等
    动作：可映射到 [rsu | precision]，如需接收/拒绝可增加 accept 头。
    """
    def __init__(self, input_dim = Actor_deploy_input_dim, action_dim = Actor_deploy_action_dim, hidden = hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)