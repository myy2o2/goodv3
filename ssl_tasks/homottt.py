from __future__ import annotations

import torch
import torch.nn as nn

from .common import augment, cosine_similarity_per_example, margin_penalty_per_example, negative_cosine_per_example


class HomoTTTConsistencyTask(nn.Module):
    name = "homottt"

    def __init__(self, perturb_ratio: float = 0.05):
        super().__init__()
        self.perturb_ratio = float(perturb_ratio)
        self.strategy = "dropedge"

    def compute_node_loss(
        self,
        model,
        feat: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight=None,
        margin: float = -1.0,
        **kwargs,
    ) -> torch.Tensor:
        output1 = augment(
            model,
            feat,
            strategy=self.strategy,
            p=self.perturb_ratio,
            edge_index=edge_index,
            edge_weight=edge_weight,
        )
        output2 = augment(
            model,
            feat,
            strategy="dropedge",
            p=0.0,
            edge_index=edge_index,
            edge_weight=edge_weight,
        )
        output3 = augment(
            model,
            feat,
            strategy="shuffle",
            p=0.0,
            edge_index=edge_index,
            edge_weight=edge_weight,
        )

        pos_term = negative_cosine_per_example(output1, output2)
        if float(margin) != -1.0:
            neg_term = margin_penalty_per_example(output2, output3, margin=margin)
            return pos_term - neg_term
        return pos_term + cosine_similarity_per_example(output2, output3)

    def compute_loss(self, model, feat: torch.Tensor, edge_index: torch.Tensor, edge_weight=None, **kwargs) -> torch.Tensor:
        return self.compute_node_loss(model, feat, edge_index, edge_weight=edge_weight, **kwargs).mean()


class HomoTTTDropNodeConsistencyTask(HomoTTTConsistencyTask):
    name = "homonode"

    def __init__(self, perturb_ratio: float = 0.05):
        super().__init__(perturb_ratio=perturb_ratio)
        self.strategy = "dropnode"
