from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor


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


class MILAggregator(nn.Module):
    """
    Multiple-Instance Learning (MIL) pooling block for segment embeddings.

    Given all segment embeddings belonging to a video, reduces the sequence
    down to a single representation that can be consumed by ``AnomalyHead``.

    Supported pooling modes:
    - ``mean``: arithmetic mean across segments.
    - ``max``: element-wise maximum across segments.
    - ``attention``: attention pooling with a learnable scoring network.
    """

    def __init__(self, input_dim: int, pooling: str = "attention", hidden_dim: int = 128):
        super().__init__()
        self.input_dim = input_dim
        self.pooling = pooling

        if pooling == "attention":
            self.attention = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, 1),
            )

    def forward(self, feats: Tensor) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Parameters
        ----------
        feats:
            Tensor of shape ``[num_segments, input_dim]`` containing all segment
            embeddings for a video/bag.

        Returns
        -------
        pooled:
            Tensor of shape ``[input_dim]`` representing the aggregated bag.
        weights:
            Optional attention weights (length ``num_segments``) when
            ``pooling == 'attention'``; otherwise ``None``.
        """
        if feats.numel() == 0:
            return torch.zeros(self.input_dim, device=feats.device), None

        if self.pooling == "mean":
            return feats.mean(dim=0), None

        if self.pooling == "max":
            pooled, _ = feats.max(dim=0)
            return pooled, None

        if self.pooling == "attention":
            scores = self.attention(feats)
            weights = torch.softmax(scores.squeeze(-1), dim=0)
            pooled = torch.sum(weights.unsqueeze(-1) * feats, dim=0)
            return pooled, weights

        raise ValueError(f"Unsupported pooling mode: {self.pooling}")
