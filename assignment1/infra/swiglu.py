from infra.silu import SiLU
from infra.linear import Linear
from typing import Optional
from torch import nn
import torch



class SwiGLU_FFN(nn.Module):
    
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        device: Optional[torch.device]=None,
        dtype: Optional[torch.dtype]=None
    ):
        super(SwiGLU_FFN, self).__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
      
        self.d_ff = d_ff
        self.d_model = d_model
        
        self.w1 = Linear(d_model, d_ff, **factory_kwargs)
        self.w2 = Linear(d_ff, d_model, **factory_kwargs)
        self.w3 = Linear(d_model, d_ff, **factory_kwargs)
        self.silu = SiLU()
    
    
    def forward(self, x: torch.Tensor):
        return self.w2(self.silu(self.w1(x)) * (self.w3(x)))
        
        