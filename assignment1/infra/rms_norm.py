from typing import Optional
from torch import nn
import torch



class RMSNorm(nn.Module):
    
    def __init__(
        self,
        d_model: int,
        eps: float=1e-5,
        device: Optional[torch.device]=None,
        dtype: Optional[torch.dtype]=None
    ):
        super(RMSNorm, self).__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.eps = eps
        self.weight = nn.Parameter(torch.ones((d_model), **factory_kwargs))
        
    
    def forward(self, x: torch.Tensor):
        in_type = x.dtype
        x = x.to(torch.float32)
        mean = torch.mean(x**2, dim=-1, keepdim=True)
        norm = x / torch.sqrt(mean + self.eps) * self.weight
        return norm.to(in_type)
        