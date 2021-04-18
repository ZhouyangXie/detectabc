import torch
import numpy as np


def to_numpy(tensor: torch.Tensor):
    '''
        copy data from CUDA/CPU tensor to a numpy ndarray
    '''

    return tensor.cpu().detach().numpy()


def to_tensor(ndarray: np.array, device: str = 'cpu'):
    if device == 'cpu':
        format = torch.tensor
    else:
        format = torch.cuda.tensor
        return format(ndarray.astype(np.float32),
                      dtype=ndarray.dtype,
                      device=device)
