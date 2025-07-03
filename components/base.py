from typing import List

import torch
import torch.nn as nn
from chameleon import build_backbone, build_neck


class Permute(nn.Module):

    def __init__(self, dims: List[int]) -> None:
        super().__init__()
        self.dims = dims

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.permute(*self.dims)


class Transpose(nn.Module):

    def __init__(self, dim1: int, dim2: int) -> None:
        super().__init__()
        self.dim1 = dim1
        self.dim2 = dim2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.transpose(self.dim1, self.dim2)


class Backbone(nn.Module):

    def __init__(self, name, **kwargs):
        super().__init__()
        self.backbone = build_backbone(name=name, **kwargs)

        with torch.no_grad():
            dummy = torch.rand(1, 3, 128, 128)
            self.channels = [i.size(1) for i in self.backbone(dummy)]

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        return self.backbone(x)


class Neck(nn.Module):

    def __init__(self, name, **kwargs):
        super().__init__()
        self.neck = build_neck(name=name, **kwargs)

    def forward(self, xs: List[torch.Tensor]) -> torch.Tensor:
        return self.neck(xs)
