from typing import Optional
from infra.softmax import LogSoftmax
import torch.nn as nn
import torch


def cross_entropy_with_logits(
    logits: torch.Tensor, 
    targets: torch.Tensor,
    ignore_index: Optional[int] = None
):
    assert logits.dim() in (2, 3), "dim not match"
    assert targets.dim() in (1, 2), "dim not match"
    
    # log_prob: [batch_size, seq_len, vocab_size]
    log_prob = LogSoftmax(logits, dim=-1)
    _, vocab_size = log_prob.shape
    # log_prob_flat: [batch_size * seq_len, vocab_size]
    log_prob_flat = log_prob.reshape(-1, vocab_size)
    # targets_flat: [batch_size * seq_len]
    targets_flat = targets.reshape(-1)
    # tgt_log_prob: [batch_size * seq_len]
    tgt_log_prob = torch.gather(
        input=log_prob_flat,
        dim=1,
        index=targets_flat.unsqueeze(-1)
    ).squeeze(1)
    
    if ignore_index is not None:
        mask = (targets_flat != ignore_index)
        tgt_log_prob = tgt_log_prob[mask]
    
    return (-tgt_log_prob.sum() / mask.float().sum()) if ignore_index else -tgt_log_prob.mean()
    