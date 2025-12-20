from typing import Optional
from torch import nn
import torch


class Embedding(nn.Module):
    
    def __init__(
        self, 
        num_embeddings: int, 
        embedding_dim: int, 
        device: Optional[torch.device]=None, 
        dtype: Optional[torch.dtype]=None
    ) -> None:
        super(Embedding, self).__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.embed = nn.Parameter(torch.rand((num_embeddings, embedding_dim), **factory_kwargs))
        self.init_parameter()
        
    def init_parameter(self):
        nn.init.trunc_normal_(self.embed, mean=0, std=1, a=-3, b=3)
        
    def forward(self, x: torch.Tensor):
        return self.embed[x]