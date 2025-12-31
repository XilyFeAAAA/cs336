import os
import torch
import typing
from typing import Optional

def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]
):
    torch.save({
        "model": model.state_dict(),
        "optim": optimizer.state_dict(),
        "iter": iteration
    }, out)


def load_checkpoint(
    src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
    model: Optional[torch.nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None
):
    states = torch.load(src)
    if model is not None:
        model.load_state_dict(states["model"])
    if optimizer is not None:
        optimizer.load_state_dict(states["optim"])
    return states["iter"]