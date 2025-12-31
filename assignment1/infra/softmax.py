import torch

def Softmax(logits: torch.Tensor, dim: int):
    max_features = logits.max(dim=dim, keepdim=True).values
    exp = torch.exp(logits - max_features)
    return exp / exp.sum(dim=dim, keepdim=True)


def LogSoftmax(logits: torch.Tensor, dim: int):
    """
    log_softmax = log(e^{x_i}/sum{e^{x_j}})
                = x_i - log(sum{e^{x_j}})
    """
    max_features = logits.max(dim=dim, keepdim=True).values
    logits_stable = logits - max_features
    sum_exp = logits_stable.exp().sum(dim=dim, keepdim=True)
    return logits_stable - sum_exp.log()
