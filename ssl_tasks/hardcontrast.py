from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import augment


class HardAugmentedContrastiveTask(nn.Module):
    name = "hardcontrast"

    def __init__(
        self,
        hidden_dim: int,
        temperature: float = 0.2,
        edge_drop: float = 0.2,
        node_drop: float = 0.1,
        hard_k: int = 16,
        hard_margin: float = 0.2,
        hard_weight: float = 0.5,
    ):
        super().__init__()
        self.temperature = float(temperature)
        self.edge_drop = float(edge_drop)
        self.node_drop = float(node_drop)
        self.hard_k = int(max(hard_k, 1))
        self.hard_margin = float(hard_margin)
        self.hard_weight = float(hard_weight)
        self.projector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def _make_view(self, model, feat: torch.Tensor, edge_index: torch.Tensor, edge_weight=None) -> torch.Tensor:
        h_edge = augment(
            model,
            feat,
            strategy="dropedge",
            p=self.edge_drop,
            edge_index=edge_index,
            edge_weight=edge_weight,
        )
        h_node = augment(
            model,
            feat,
            strategy="dropnode",
            p=self.node_drop,
            edge_index=edge_index,
            edge_weight=edge_weight,
        )
        return 0.5 * (h_edge + h_node)

    def _hard_negative_term(self, logits: torch.Tensor, positive: torch.Tensor) -> torch.Tensor:
        n = logits.shape[0]
        mask = torch.eye(n, device=logits.device, dtype=torch.bool)
        neg_logits = logits.masked_fill(mask, -1e9)
        k = min(self.hard_k, max(n - 1, 1))
        hard_vals = torch.topk(neg_logits, k=k, dim=-1).values.mean(dim=-1)
        return F.relu(hard_vals - positive + self.hard_margin)

    def compute_node_loss(self, model, feat: torch.Tensor, edge_index: torch.Tensor, edge_weight=None, **kwargs) -> torch.Tensor:
        h1 = self._make_view(model, feat, edge_index, edge_weight=edge_weight)
        h2 = self._make_view(model, feat, edge_index, edge_weight=edge_weight)

        z1 = F.normalize(self.projector(h1), dim=-1)
        z2 = F.normalize(self.projector(h2), dim=-1)

        logits12 = (z1 @ z2.t()) / self.temperature
        logits21 = (z2 @ z1.t()) / self.temperature
        target = torch.arange(z1.size(0), device=z1.device)

        ce12 = F.cross_entropy(logits12, target, reduction="none")
        ce21 = F.cross_entropy(logits21, target, reduction="none")
        pos12 = logits12.diag()
        pos21 = logits21.diag()
        hard12 = self._hard_negative_term(logits12, pos12)
        hard21 = self._hard_negative_term(logits21, pos21)

        return 0.5 * (ce12 + ce21) + self.hard_weight * 0.5 * (hard12 + hard21)

    def compute_loss(self, model, feat: torch.Tensor, edge_index: torch.Tensor, edge_weight=None, **kwargs) -> torch.Tensor:
        return self.compute_node_loss(model, feat, edge_index, edge_weight=edge_weight, **kwargs).mean()
