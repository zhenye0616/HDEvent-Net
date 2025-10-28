from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Tuple


class AnomalyHead(nn.Module):
    """
    Lightweight prediction head for anomaly detection.

    Produces:
      - Binary logits for anomaly vs. normal.
      - Optional multi-class logits over anomaly classes.
    """

    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.binary_head = nn.Linear(input_dim, 1)
        self.class_head = nn.Linear(input_dim, num_classes) if num_classes > 0 else None

    def forward(self, x: Tensor) -> Tuple[Tensor, Optional[Tensor]]:
        logits_bin = self.binary_head(x)
        logits_cls = self.class_head(x) if self.class_head is not None else None
        return logits_bin, logits_cls
