from infra.attention import Multihead_Self_Attention
from infra.embedding import Embedding
from infra.softmax import Softmax
from infra.swiglu import SwiGLU_FFN
from infra.linear import Linear
from infra.rms_norm import RMSNorm
from typing import Optional
import torch.nn as nn
import torch




class TransformerBlock(nn.Module):
    
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        num_heads: int,
        use_rope: bool,
        device: Optional[torch.device],
        dtype: Optional[torch.dtype],
        theta: float,
        d_k: int,
        max_seq_len: int,
        eps: float = 1e-5
    ):
        super(TransformerBlock, self).__init__()
        
        self.attn = Multihead_Self_Attention(d_model, num_heads, use_rope, device, theta=theta, d_k=d_k, max_seq_len=max_seq_len)
        self.ffn = SwiGLU_FFN(d_model, d_ff, device, dtype)
        self.ln1 = RMSNorm(d_model, eps, device, dtype)
        self.ln2 = RMSNorm(d_model, eps, device, dtype)
        
        
    def forward(self, x: torch.Tensor, token_positions: Optional[torch.Tensor]):
        attn_out = self.attn(self.ln1(x), token_positions)
        residual = x + attn_out
        ffn_out = self.ffn(self.ln2(residual))
        return residual + ffn_out



class Transformer(nn.Module):
    
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        d_ff: int,
        num_layers: int,
        num_heads: int,
        device: Optional[torch.device],
        dtype: Optional[torch.dtype],
        theta: float,
        d_k: int,
        eps: float = 1e-5,
        use_rope: bool = True,
    ):
        super(Transformer, self).__init__()

        self.device = device
        self.context_length = context_length
        
        
        self.token_embeddings = Embedding(vocab_size, d_model, device, dtype)
        self.layers = nn.ModuleList([
                TransformerBlock(d_model, d_ff, num_heads, use_rope, device, dtype, theta, d_k, context_length, eps)
            for _ in range(num_layers)])
        self.ln_final = RMSNorm(d_model, eps, device, dtype)
        self.lm_head = Linear(d_model, vocab_size)
        
    def forward(self, x: torch.Tensor):
        B, L = x.shape
        positions = torch.arange(L, device=self.device).unsqueeze(0).expand(B, L)
        
        embed = self.token_embeddings(x)
        attn = embed
        for layer in self.layers:
            attn = layer(attn, positions)
        
        norm = self.ln_final(attn)
        proj = self.lm_head(norm)
        
        return proj
        
        
        