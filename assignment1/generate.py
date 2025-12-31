from infra import (
    load_checkpoint,
    Transformer,
    Softmax
)
from bpe import BPE_Tokenizer
from typing import Optional
import pickle
import torch
import os

MAX_LENGTH = 250
EOS_TOKEN = "<|endoftext|>"
CHECKPOINT_PATH = ""
VOCAB_PATH = r"D:\dev\github\cs336\assignment1\data\vocab.pkl"
MERGES_PATH = r"D:\dev\github\cs336\assignment1\data\merges.pkl"
TRANSFORMER_CONFIG = {
    "vocab_size": 10_000,
    "context_length": 250,
    "d_model": 512,
    "d_ff": 2048,
    "num_layers": 12,
    "num_heads": 8,
    "device": "cuda",
    "dtype": torch.float32,
    "theta": 10_000,
    "d_k": 512 // 8,
    "eps": 1e-8,
    "use_rope": True
}


def top_p_sampling(probs: torch.Tensor, p: float):
    sorted_probs, indices = torch.sort(probs, descending=True)
    accum_probs = torch.cumsum(sorted_probs, dim=-1)
    mask = accum_probs <= p
    
    # 保证至少有一个
    mask[..., 0] = True
    filitered_probs = sorted_probs * mask
    filitered_probs = filitered_probs / filitered_probs.sum()
    return indices[torch.multinomial(filitered_probs, num_samples=1).item()]


def generate_text(
    prompt: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    model: Transformer,
    tokenizer: BPE_Tokenizer,
    eos_token: Optional[str] = None
):
    token_ids = tokenizer.encode(prompt)
  
    model.eval()
    with torch.no_grad():
        for _ in range(max_tokens):
            inp = torch.tensor(token_ids).unsqueeze(0)
            logits = model(inp)[0, -1, :]
            scaled_logits = logits / temperature
            probs = Softmax(scaled_logits, dim=-1)
            next_token_id = top_p_sampling(probs, top_p)
            
            if eos_token is not None and tokenizer.decode([next_token_id]).strip() == eos_token:
                break
            
            token_ids.append(next_token_id)
        
    return tokenizer.decode(token_ids)
            


if __name__ == "__main__":
    assert os.path.exists(CHECKPOINT_PATH), "checkpoint file doesn't exist"
    assert os.path.exists(VOCAB_PATH), "tokenizer vocab file doesn't exist"
    assert os.path.exists(MERGES_PATH), "tokenizer merges file doesn't exist"
    
    model = Transformer(**TRANSFORMER_CONFIG)
    checkpoint = load_checkpoint(CHECKPOINT_PATH, model)
    
    with open(VOCAB_PATH, "rb") as f:
        vocab = pickle.load(f)
    with open(MERGES_PATH, "rb") as f:
        merges = pickle.load(f)
    special_tokens = [EOS_TOKEN]
    tokenizer = BPE_Tokenizer(vocab, merges, special_tokens)
    
    prompt = input()
    generate_text(prompt, MAX_LENGTH, model, tokenizer, EOS_TOKEN)
    
   
        