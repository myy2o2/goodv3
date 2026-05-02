from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import model_embeddings


class NodeInfoNCEContrastiveTask(nn.Module):
    name = "contrastive"

    def __init__(self, hidden_dim: int, temperature: float = 0.2, emb_dropout: float = 0.1):
        super().__init__()
        self.temperature = float(temperature)
        self.emb_dropout = float(emb_dropout)
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def compute_node_loss(self, model, feat: torch.Tensor, edge_index: torch.Tensor, edge_weight=None, **kwargs) -> torch.Tensor:
        h = model_embeddings(model, feat, edge_index, edge_weight)
        z1 = self.proj(F.dropout(h, p=self.emb_dropout, training=self.training))
        z2 = self.proj(F.dropout(h, p=self.emb_dropout, training=self.training))
        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)
        logits12 = (z1 @ z2.t()) / self.temperature
        logits21 = (z2 @ z1.t()) / self.temperature
        target = torch.arange(z1.size(0), device=h.device)
        loss12 = F.cross_entropy(logits12, target, reduction="none")
        loss21 = F.cross_entropy(logits21, target, reduction="none")
        return 0.5 * (loss12 + loss21)

    def compute_loss(self, model, feat: torch.Tensor, edge_index: torch.Tensor, edge_weight=None, **kwargs) -> torch.Tensor:
        return self.compute_node_loss(model, feat, edge_index, edge_weight=edge_weight, **kwargs).mean()
