import os
import torch
import typing

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
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer
):
    states = torch.load(src)
    model.load_state_dict(states["model"])
    optimizer.load_state_dict(states["optim"])
    return states["iter"]