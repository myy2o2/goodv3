from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import model_embeddings, weighted_neighbor_average


class GraphTTAPseudoContrastiveTask(nn.Module):
    name = "graphtta"

    def __init__(
        self,
        hidden_dim: int,
        num_pseudo_classes: int = 8,
        contrast_temperature: float = 0.1,
        edge_temperature: float = 1.0,
    ):
        super().__init__()
        self.contrast_temperature = float(contrast_temperature)
        self.edge_temperature = float(edge_temperature)
        self.pseudo_head = nn.Linear(hidden_dim, num_pseudo_classes)
        self.projector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def _augmented_view(self, h: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        src, dst = edge_index
        cosine = F.cosine_similarity(h[src], h[dst], dim=-1, eps=1e-8)
        edge_weight = torch.sigmoid(cosine / max(self.edge_temperature, 1e-6))
        return weighted_neighbor_average(h, edge_index, edge_weight)

    def compute_node_loss(self, model, feat: torch.Tensor, edge_index: torch.Tensor, edge_weight=None, **kwargs) -> torch.Tensor:
        h = model_embeddings(model, feat, edge_index, edge_weight)
        aug_h = self._augmented_view(h, edge_index)
        z = F.normalize(self.projector(h), dim=-1)
        z_aug = F.normalize(self.projector(aug_h), dim=-1)

        with torch.no_grad():
            pseudo_labels = self.pseudo_head(h).argmax(dim=-1)
            aug_pseudo_labels = self.pseudo_head(aug_h).argmax(dim=-1)

        similarities = (z @ z_aug.t()) / max(self.contrast_temperature, 1e-6)
        log_probs = similarities - torch.logsumexp(similarities, dim=1, keepdim=True)
        positive_mask = pseudo_labels.unsqueeze(1) == aug_pseudo_labels.unsqueeze(0)
        positive_count = positive_mask.sum(dim=1).clamp_min(1)
        node_loss = -(log_probs * positive_mask).sum(dim=1) / positive_count
        return node_loss

    def compute_loss(self, model, feat: torch.Tensor, edge_index: torch.Tensor, edge_weight=None, **kwargs) -> torch.Tensor:
        return self.compute_node_loss(model, feat, edge_index, edge_weight=edge_weight, **kwargs).mean()
