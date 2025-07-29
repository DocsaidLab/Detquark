from typing import List

import torch
import torch.nn as nn

__all__ = [
    "Permute",
    "Transpose",
]


class Permute(nn.Module):
    """
    Reorders the dimensions of a tensor.

    Attributes:
        dims (List[int]): Desired ordering of dimensions.
    """

    def __init__(self, dims: List[int]) -> None:
        """
        Initialize Permute module.

        Args:
            dims (List[int]): Sequence of dimension indices to permute.
        """
        super().__init__()
        self.dims = dims

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Permute the input tensor according to `dims`.

        Args:
            x (torch.Tensor): Input tensor of any shape.

        Returns:
            torch.Tensor: Tensor with dimensions reordered.
        """
        return x.permute(*self.dims)


class Transpose(nn.Module):
    """
    Swaps two dimensions of a tensor.

    Attributes:
        dim1 (int): First dimension index.
        dim2 (int): Second dimension index.
    """

    def __init__(self, dim1: int, dim2: int) -> None:
        """
        Initialize Transpose module.

        Args:
            dim1 (int): First dimension to swap.
            dim2 (int): Second dimension to swap.
        """
        super().__init__()
        self.dim1 = dim1
        self.dim2 = dim2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Transpose the two specified dimensions of the input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Tensor with `dim1` and `dim2` swapped.
        """
        return x.transpose(self.dim1, self.dim2)
