from typing import Tuple

import torch
import torch.nn as nn


class ReShape(nn.Module):

    def __init__(self, shape: Tuple[int, ...]):
        super().__init__()
        self.shape = shape

    def forward(self, x: torch.Tensor):
        return torch.reshape(x, *self.shape)
