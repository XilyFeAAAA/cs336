from typing import Iterable
import torch

def gradient_clipping(
    params: Iterable[torch.nn.Parameter],
    max_l2_norm: float,
    eps: float = 1e-6
):
    # 步骤1：只收集有梯度的参数的梯度（跳过grad=None的参数）
    grads = []
    # 额外记录：有梯度的参数列表 + 每个参数的元素数量（方便后续拆分）
    params_with_grad = []
    numels = []
    
    for param in params:
        if param.grad is not None:  # 只处理有梯度的参数
            grad_flat = param.grad.view(-1)  # 展平但不克隆（节省内存）
            grads.append(grad_flat)
            params_with_grad.append(param)
            numels.append(param.numel())  # 记录每个参数的元素总数
    
    # 无需要裁剪的梯度，直接返回
    if not grads:
        return
    
    # 步骤2：拼接梯度并计算L2范数（数值稳定版）
    grads = torch.cat(grads)
    l2_norm = torch.norm(grads, p=2)  # 替代(grads**2).sum().sqrt()
    
    # 步骤3：仅当范数超过阈值时，缩放梯度（eps避免除以0）
    if l2_norm > max_l2_norm:
        # eps加在l2_norm上，仅当l2_norm=0时生效（避免除以0）
        scale = max_l2_norm / (l2_norm + eps)
        grads *= scale
    
    # 步骤4：将缩放后的梯度拆分并赋值回原参数的grad
    pointer = 0
    for param, numel in zip(params_with_grad, numels):
        # 拆分梯度并恢复原参数的形状
        grad_chunk = grads[pointer: pointer + numel].view_as(param)
        # 原地替换原梯度（避免创建新张量，节省内存）
        param.grad.copy_(grad_chunk)
        pointer += numel