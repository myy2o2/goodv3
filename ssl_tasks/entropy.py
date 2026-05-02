from __future__ import annotations

import torch
import torch.nn as nn

from .common import softmax_entropy


class PredictionEntropyTask(nn.Module):
    name = "entropy"
    aliases = ("prediction_entropy", "confidence")

    def __init__(self, max_samples: int = 1000):
        super().__init__()
        self.max_samples = int(max_samples)

    def compute_node_loss(self, model, feat: torch.Tensor, edge_index: torch.Tensor, edge_weight=None, **kwargs) -> torch.Tensor:
        logits = model.forward(feat, edge_index, edge_weight)
        return softmax_entropy(logits)

    def compute_loss(self, model, feat: torch.Tensor, edge_index: torch.Tensor, edge_weight=None, **kwargs) -> torch.Tensor:
        node_loss = self.compute_node_loss(model, feat, edge_index, edge_weight=edge_weight, **kwargs)
        if self.max_samples <= 0 or node_loss.shape[0] <= self.max_samples:
            return node_loss.mean()
        sampled = torch.randperm(node_loss.shape[0], device=node_loss.device)[: self.max_samples]
        return node_loss[sampled].mean()
