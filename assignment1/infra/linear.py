from typing import Optional
from torch import nn
import torch
import math


class Linear(nn.Module):
    
    def __init__(
        self, 
        in_features: int,
        out_features: int,
        device: Optional[torch.device]=None,
        dtype: Optional[torch.dtype]=None
    ):
        """
        计算时候是 y=xW^T,存储的是 W 而不是 W^T,所以要反过来
        """
        super(Linear, self).__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.rand((out_features, in_features), **factory_kwargs))
        
        self.init_parameter()
    
    
    def init_parameter(self):
        std = math.sqrt(2 / (self.in_features + self.out_features))
        nn.init.trunc_normal_(self.weight, mean=0, std=std, a=-3*std, b=3*std)
    
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x 这里是行向量，所以进行的变化是 y=xW^T
        """
        return torch.matmul(x, self.weight.T)