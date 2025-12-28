from typing import Optional, Iterable
import torch
import math


class SGD(torch.optim.Optimizer):
    
    def __init__(self, params, lr=1e-3):
        assert lr >=0, f"Invalid learning rate: {lr}"
        
        defaults = {"lr": lr}
        super().__init__(params, defaults)
    
    
    def step(self, closure: Optional[callable]=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            
            lr = group["lr"]
            for p in group["params"]:
            
                if p.grad is None:
                    continue
                 
                if not (state := self.state[p]):
                    state["step"] = 0
                state["step"] += 1                
            
                grad = p.grad.data
                p.data.sub_(lr / math.sqrt(1 + state["step"]) * grad)
                
        return loss

class AdamW(torch.optim.Optimizer):
    def __init__(
            self,
            params: Iterable[torch.nn.parameter.Parameter],
            lr: float = 1e-3,
            betas: tuple[float, float] = (0.9, 0.999),
            eps: float = 1e-8,
            weight_decay: float = 0.0,
            correct_bias: bool = True,
    ):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[1]))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(eps))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, correct_bias=correct_bias)
        super().__init__(params, defaults)

    def step(self, closure: Optional[callable] = None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]
            alpha = group["lr"]
            
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")

                state = self.state[p]
                
                if not state:
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)
                    state["step"] = 0
                
                state["step"] += 1
                
                state["m"].mul_(beta1).add_(grad, alpha=1-beta1)
                state["v"].mul_(beta2).addcmul_(grad, grad, value=1-beta2)

                if group["correct_bias"]:
                    step_size = alpha * math.sqrt(1 - beta2 ** state["step"]) / (1 - beta1 ** state["step"])
                else:
                    step_size = alpha
                    
                denom = state["v"].sqrt().add_(eps)
                
                p.data.addcdiv_(state["m"], denom, value=-step_size)
                
                if wd != 0:
                    p.data.add_(p.data, alpha=-alpha * wd)
            

        return loss


def lr_cosine_schedule(
    t: int,
    max_lr: float,
    min_lr: float,
    warmup_iters: int,
    cosine_iters: int
):
    if t < warmup_iters:
        return t / warmup_iters * max_lr
    elif t <= cosine_iters:
        return min_lr + 0.5 * (1 + math.cos((t - warmup_iters) / (cosine_iters - warmup_iters) * math.pi)) * (max_lr - min_lr)
    else:
        return min_lr


if __name__ == "__main__":
    weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
    opt = SGD([weights], lr=1)
    for t in range(100):
        opt.zero_grad() # Reset the gradients for all learnable parameters.
        loss = (weights**2).mean() # Compute a scalar loss value.
        print(loss.cpu().item())
        loss.backward() # Run backward pass, which computes gradients.
        opt.step() # Run optimizer step.