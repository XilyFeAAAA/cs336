from typing import Iterable
import torch

def gradient_clipping(
    params: Iterable[torch.nn.Parameter],
    max_l2_norm: float,
    eps: float = 1e-6
):
    grads = []
    params_with_grad = []
    numels = []
    
    for param in params:
        if param.grad is not None:
            grads.append(param.grad.view(-1))
            params_with_grad.append(param)
            numels.append(param.numel())
    
    if not grads:
        return
    
    grads = torch.cat(grads)
    l2_norm = torch.norm(grads, p=2)
    
    if l2_norm > max_l2_norm:
        grads *= max_l2_norm / (l2_norm + eps)
    
    pointer = 0
    for param, numel in zip(params_with_grad, numels):
        grad_chunk = grads[pointer: pointer + numel].view_as(param)
        param.grad.copy_(grad_chunk)
        pointer += numel