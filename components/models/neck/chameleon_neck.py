from typing import List

import torch
import torch.nn as nn
from chameleon import build_neck

# TODO: 解耦 Chameleon 的依賴關係


class ChameleonNeck(nn.Module):

    def __init__(self, name: str, **kwargs):
        super().__init__()
        self.neck = build_neck(name=name, **kwargs)

    def forward(self, xs: List[torch.Tensor]) -> torch.Tensor:
        return self.neck(xs)
