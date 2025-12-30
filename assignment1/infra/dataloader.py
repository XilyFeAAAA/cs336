import torch
import numpy as np
import numpy.typing as npt

def get_batch(
    dataset: npt.NDArray, 
    batch_size: int, 
    context_length: int, 
    device: str
):
    stas = np.random.randint(low=0, high=len(dataset)-context_length, size=batch_size)
    batch_x = np.stack([dataset[sta:sta+context_length] for sta in stas])
    batch_y = np.stack([dataset[sta+1:sta+1+context_length] for sta in stas])
    return torch.from_numpy(batch_x).to(device), torch.from_numpy(batch_y).to(device)