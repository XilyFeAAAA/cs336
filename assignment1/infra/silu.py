import torch.nn as nn
import torch


class SiLU(nn.Module):
    
    def __init__(self):
        super(SiLU, self).__init__()
    
    
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(x)