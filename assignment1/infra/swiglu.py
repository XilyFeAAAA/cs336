from typing import Optional
from torch import nn
import torch
import math



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
        
        self.w1 = nn.Parameter(torch.rand((d_ff, d_model), **factory_kwargs))
        self.w2 = nn.Parameter(torch.rand((d_model, d_ff), **factory_kwargs))
        self.w3 = nn.Parameter(torch.rand((d_ff, d_model), **factory_kwargs))
        
        self.init_parameter()
        
    def init_parameter(self):
        nn.init.normal_(self.w1, mean=0, std=math.sqrt(2/self.d_ff))
        nn.init.normal_(self.w2, mean=0, std=math.sqrt(2/self.d_model))
        nn.init.normal_(self.w3, mean=0, std=math.sqrt(2/self.d_ff))
    
    
    def SiLU(self, x: torch.Tensor):
        return x * torch.sigmoid(x)
    
    
    def forward(self, x: torch.Tensor):
        return (self.SiLU(x @ self.w1.T) * (x @ self.w3.T)) @ self.w2.T
        
        