from typing import Optional
from torch import nn
import torch


class RoPE(nn.Module):
    
    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: Optional[torch.device]=None
    ):
        super(RoPE, self).__init__()
        
        position = torch.arange(max_seq_len, device=device, dtype=torch.float32) # [max_len]
        freq = 1.0 / (theta ** (torch.arange(0, d_k, 2, device=device, dtype=torch.float32) / d_k))
        sinusoid = torch.outer(position, freq)
        rot = torch.exp(1j * sinusoid)  # [max_seq_len, dim//2]
        self.register_buffer("rot_cache", rot)
    
    def forward(self, qk: torch.Tensor, token_positions: torch.Tensor):
        *_, dim = qk.shape
    
        assert dim % 2 == 0, "dim must be even"
        
        qk_complex = qk.view(*qk.shape[:-1], dim//2, 2) # [bsize, nheads, seq_len, dim//2, 2]
        qk_complex = torch.view_as_complex(qk_complex)  # [bsize, nheads, seq_len, dim//2]
        
        
        rotated_qk_complex = qk_complex * self.rot_cache[token_positions]
        rotated_qk = torch.view_as_real(rotated_qk_complex)  # [bsize, nheads, seq_len, dim//2, 2]
        rotated_qk = rotated_qk.view_as(qk)
        
        return rotated_qk