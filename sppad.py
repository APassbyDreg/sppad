from spconv.pytorch import SparseConvTensor
from typing import List, Tuple, Union
from torch import Tensor

import torch

from torch import nn


def __pad_dimention(
    features: Tensor,
    indices: Tensor,
    dim_size: int,
    dim: int,
    pad: Tuple[int, int],
    mode: str,
) -> Tuple[Tensor, Tensor]:
    """pad a dimention

    Args:
        feats (Tensor): original sparse tensor features
        indices (Tensor): original sparse tensor indices
        spatial_shapes (Tuple[int]): spatial shape of the sparse tensor
        dim (int): dimention id
        pad (Tuple[int, int]): padding (left / right)
        mode (str): padding mode (mirror or repeat)

    Returns:
        Tuple[Tensor, Tensor]: new sparse tensor features and indices
    """
    new_feats = []
    new_indices = []
    # left and right pad
    if mode == 'mirror': # ab -> ba|ab|ba
        if pad[0] > 0:
            l_pad = indices[:, dim + 1] < pad[0]
            l_indices = indices[l_pad]
            l_indices[:, dim + 1] = pad[0] - 1 - l_indices[:, dim + 1]
            new_indices.append(l_indices)
            new_feats.append(features[l_pad])
        if pad[1] > 0:
            r_pad = indices[:, dim + 1] >= dim_size - pad[1]
            r_indices = indices[r_pad]
            r_indices[:, dim + 1] = pad[0] + dim_size * 2 - 1 - r_indices[:, dim + 1]
            new_indices.append(r_indices)
            new_feats.append(features[r_pad])
    elif mode == 'repeat': # ab -> ab|ab|ab
        if pad[0] > 0:
            l_pad = indices[:, dim + 1] >= dim_size - pad[0]
            l_indices = indices[l_pad]
            l_indices[:, dim + 1] = pad[0] + l_indices[:, dim + 1] - dim_size
            new_indices.append(l_indices)
            new_feats.append(features[l_pad])
        if pad[1] > 0:
            r_pad = indices[:, dim + 1] < pad[1]
            r_indices = indices[r_pad]
            r_indices[:, dim + 1] = pad[0] + dim_size + r_indices[:, dim + 1]
            new_indices.append(r_indices)
            new_feats.append(features[r_pad])
    else:
        raise ValueError(f"invalid pad mode {mode}")
    # center part
    indices[:, dim + 1] = pad[0] + indices[:, dim + 1]
    new_indices.append(indices)
    new_feats.append(features)
    # combine and return
    features = torch.cat(new_feats, dim=0)
    indices = torch.cat(new_indices, dim=0)
    return features, indices


def sp_pad(
    tensor: SparseConvTensor, 
    padding: Union[list[int], int], 
    mode: Union[list[str], str],
) -> SparseConvTensor:
    """pad a sparse tensor
    
    Args:
        tensor (SparseConvTensor): sparse tensor to pad
        padding (Union[list[int], int]): padding of each dimension
        mode (Union[list[str], str]): padding mode of each dimension
        
    Returns:
        SparseConvTensor: the padded sparse tensor
    """
    MODE_OPTIONS = ['mirror', 'repeat']
    D = len(tensor.spatial_shape)
    B = tensor.batch_size
    # expand padding to ((x0, x1), (y0, y1), (z0, z1), ...)
    if isinstance(padding, int):
        padding = [[padding, padding] for i in range(D)]
    elif len(padding) == D:
        padding = [[padding[i], padding[i]] for i in range(D)]
    # expand mode to (x, y, z, ...)
    if isinstance(mode, str):
        mode = [mode] * D
    # check
    for d in range(D):
        cond = 0 <= padding[d][0] <= tensor.spatial_shape[d] and 0 <= padding[d][1] <= tensor.spatial_shape[d]
        assert cond, f"padding at {d}: {padding[d]} is invalid for shape {tensor.spatial_shape[d]}"
    for m in mode:
        assert m in MODE_OPTIONS, f"invalid mode {m}"
    # do padding
    features = tensor.features
    indices = tensor.indices.clone().detach()
    shape = [_ for _ in tensor.spatial_shape]
    for d in range(D):
        features, indices = __pad_dimention(features, indices, shape[d], d, padding[d], mode[d])
        shape[d] = shape[d] + padding[d][0] + padding[d][1]
    return SparseConvTensor(features, indices, shape, B)
    

class SparseRepeatPad(nn.Module):
    """simple module warpper for repeat padding"""
    def __init__(self, padding: Union[list[int], int]):
        super().__init__()
        self.padding = padding
    def forward(self, x: SparseConvTensor) -> SparseConvTensor:
        return sp_pad(x, self.padding, 'repeat')
    

class SparseMirrorPad(nn.Module):
    """simple module warpper for mirror padding"""
    def __init__(self, padding: Union[list[int], int]):
        super().__init__()
        self.padding = padding
    def forward(self, x: SparseConvTensor) -> SparseConvTensor:
        return sp_pad(x, self.padding, 'mirror')