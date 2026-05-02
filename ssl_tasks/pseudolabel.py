from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import augment


class PseudoLabelConsistencyTask(nn.Module):
    name = "pseudolabel"

    def __init__(
        self,
        weak_edge_drop: float = 0.05,
        strong_edge_drop: float = 0.3,
        strong_node_drop: float = 0.15,
        sharpen_temp: float = 0.5,
        confidence_threshold: float = 0.7,
        entropy_weight: float = 0.05,
    ):
        super().__init__()
        self.weak_edge_drop = float(weak_edge_drop)
        self.strong_edge_drop = float(strong_edge_drop)
        self.strong_node_drop = float(strong_node_drop)
        self.sharpen_temp = float(max(sharpen_temp, 1e-6))
        self.confidence_threshold = float(confidence_threshold)
        self.entropy_weight = float(entropy_weight)

    def _sharpen(self, probs: torch.Tensor) -> torch.Tensor:
        p = probs.pow(1.0 / self.sharpen_temp)
        return p / p.sum(dim=-1, keepdim=True).clamp_min(1e-8)

    def compute_node_loss(self, model, feat: torch.Tensor, edge_index: torch.Tensor, edge_weight=None, **kwargs) -> torch.Tensor:
        with torch.no_grad():
            weak_h = augment(
                model,
                feat,
                strategy="dropedge",
                p=self.weak_edge_drop,
                edge_index=edge_index,
                edge_weight=edge_weight,
            )
            weak_logits = model.classifier(weak_h)
            weak_prob = weak_logits.softmax(dim=-1)
            teacher = self._sharpen(weak_prob)
            conf = teacher.max(dim=-1).values

        strong_edge_h = augment(
            model,
            feat,
            strategy="dropedge",
            p=self.strong_edge_drop,
            edge_index=edge_index,
            edge_weight=edge_weight,
        )
        strong_node_h = augment(
            model,
            feat,
            strategy="dropnode",
            p=self.strong_node_drop,
            edge_index=edge_index,
            edge_weight=edge_weight,
        )
        strong_h = 0.5 * (strong_edge_h + strong_node_h)
        student_logits = model.classifier(strong_h)
        student_logprob = student_logits.log_softmax(dim=-1)

        kl = F.kl_div(student_logprob, teacher, reduction="none").sum(dim=-1)
        conf_mask = (conf >= self.confidence_threshold).float()
        if float(conf_mask.sum()) < 1.0:
            conf_mask = torch.ones_like(conf_mask)

        ent = -(student_logits.softmax(dim=-1) * student_logprob).sum(dim=-1)
        return conf_mask * kl + self.entropy_weight * ent

    def compute_loss(self, model, feat: torch.Tensor, edge_index: torch.Tensor, edge_weight=None, **kwargs) -> torch.Tensor:
        return self.compute_node_loss(model, feat, edge_index, edge_weight=edge_weight, **kwargs).mean()
