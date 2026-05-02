from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

import torch
import torch.nn as nn

from .consistency import EmbeddingConsistencyTask
from .contrastive import NodeInfoNCEContrastiveTask
from .hardcontrast import HardAugmentedContrastiveTask
from .degree import DegreePredictionTask
from .denoise import FeatureDenoisingTask
from .entropy import PredictionEntropyTask
from .bootstrap import BootstrapConsistencyTask
from .generative import EdgeReconstructionTask, NegativeSamplingEdgeReconstructionTask
from .maskedgen import MaskedFeatureGenerationTask
from .graphtta import GraphTTAPseudoContrastiveTask
from .homottt import HomoTTTConsistencyTask, HomoTTTDropNodeConsistencyTask
from .neighbor import NeighborReconstructionTask
from .pseudolabel import PseudoLabelConsistencyTask
from .propagation import PropagationConsistencyTask
from .residual import ResidualFeatureReconstructionTask
from .smoothness import EdgeSmoothnessTask


_TASKS = {
    "consistency": EmbeddingConsistencyTask,
    "contrastive": NodeInfoNCEContrastiveTask,
    "hardcontrast": HardAugmentedContrastiveTask,
    "bootstrap": BootstrapConsistencyTask,
    "degree": DegreePredictionTask,
    "denoise": FeatureDenoisingTask,
    "generative": NegativeSamplingEdgeReconstructionTask,
    "maskedgen": MaskedFeatureGenerationTask,
    "graphtta": GraphTTAPseudoContrastiveTask,
    "homottt": HomoTTTConsistencyTask,
    "homonode": HomoTTTDropNodeConsistencyTask,
    "neighbor": NeighborReconstructionTask,
    "pseudolabel": PseudoLabelConsistencyTask,
    "propagation": PropagationConsistencyTask,
    "recon": EdgeReconstructionTask,
    "residual": ResidualFeatureReconstructionTask,
    "smoothness": EdgeSmoothnessTask,
    "entropy": PredictionEntropyTask,
}


def available_ssl_tasks() -> list[str]:
    return sorted(_TASKS.keys())


def _dedupe_preserve_order(values: Sequence[str]) -> list[str]:
    seen = set()
    ordered = []
    for value in values:
        if value not in seen:
            seen.add(value)
            ordered.append(value)
    return ordered


def _canonicalize_task_name(task_name: str) -> str:
    key = str(task_name).strip().lower()
    if key not in _TASKS:
        raise ValueError(f"Unsupported SSL task: {task_name}. Available tasks: {available_ssl_tasks()}")
    return key


def parse_ssl_task_names(task_spec: str | Sequence[str] | None) -> list[str]:
    if task_spec is None:
        return []
    if isinstance(task_spec, (list, tuple, set)):
        return _dedupe_preserve_order([_canonicalize_task_name(item) for item in task_spec])

    raw = str(task_spec).strip()
    if not raw:
        return []
    lowered = raw.lower()

    tokens = [token for token in re.split(r"[\s,+|/]+", lowered) if token]
    return _dedupe_preserve_order([_canonicalize_task_name(token) for token in tokens])


def build_ssl_tasks(
    task_spec: str | Sequence[str] | None,
    task_cfg: Mapping[str, object] | None = None,
    device: torch.device | str | None = None,
    hidden_dim: int | None = None,
    input_dim: int | None = None,
) -> nn.ModuleList:
    task_cfg = dict(task_cfg or {})
    target_device = None if device is None else torch.device(device)

    def require(name: str, value: int | None, task_name: str) -> int:
        if value is None:
            raise ValueError(f"{task_name} requires {name} to build its task head.")
        return int(value)

    tasks = []
    for task_name in parse_ssl_task_names(task_spec):
        if task_name == "consistency":
            task = EmbeddingConsistencyTask(
                hidden_dim=require("hidden_dim", hidden_dim, task_name),
                proj_dim=int(task_cfg.get("consistency_proj_dim", 0)),
                dropout=float(task_cfg.get("consistency_dropout", 0.2)),
            )
        elif task_name == "contrastive":
            task = NodeInfoNCEContrastiveTask(
                hidden_dim=require("hidden_dim", hidden_dim, task_name),
                temperature=float(task_cfg.get("contrastive_temperature", 0.2)),
                emb_dropout=float(task_cfg.get("contrastive_dropout", 0.1)),
            )
        elif task_name == "hardcontrast":
            task = HardAugmentedContrastiveTask(
                hidden_dim=require("hidden_dim", hidden_dim, task_name),
                temperature=float(task_cfg.get("hardcontrast_temperature", 0.2)),
                edge_drop=float(task_cfg.get("hardcontrast_edge_drop", 0.2)),
                node_drop=float(task_cfg.get("hardcontrast_node_drop", 0.1)),
                hard_k=int(task_cfg.get("hardcontrast_hard_k", 16)),
                hard_margin=float(task_cfg.get("hardcontrast_margin", 0.2)),
                hard_weight=float(task_cfg.get("hardcontrast_weight", 0.5)),
            )
        elif task_name == "bootstrap":
            task = BootstrapConsistencyTask(
                hidden_dim=require("hidden_dim", hidden_dim, task_name),
                edge_drop=float(task_cfg.get("bootstrap_edge_drop", 0.2)),
                node_drop=float(task_cfg.get("bootstrap_node_drop", 0.1)),
                predictor_hidden=int(task_cfg.get("bootstrap_predictor_hidden", 0)),
            )
        elif task_name == "degree":
            task = DegreePredictionTask(
                hidden_dim=require("hidden_dim", hidden_dim, task_name),
                num_bins=int(task_cfg.get("degree_bins", 6)),
            )
        elif task_name == "denoise":
            task = FeatureDenoisingTask(
                hidden_dim=require("hidden_dim", hidden_dim, task_name),
                input_dim=require("input_dim", input_dim, task_name),
            )
        elif task_name == "generative":
            task = NegativeSamplingEdgeReconstructionTask(
                hidden_dim=require("hidden_dim", hidden_dim, task_name),
                neg_ratio=float(task_cfg.get("generative_neg_ratio", 1.0)),
            )
        elif task_name == "maskedgen":
            task = MaskedFeatureGenerationTask(
                hidden_dim=require("hidden_dim", hidden_dim, task_name),
                input_dim=require("input_dim", input_dim, task_name),
                mask_ratio=float(task_cfg.get("maskedgen_mask_ratio", 0.3)),
            )
        elif task_name == "graphtta":
            task = GraphTTAPseudoContrastiveTask(
                hidden_dim=require("hidden_dim", hidden_dim, task_name),
                num_pseudo_classes=int(task_cfg.get("pseudo_classes", 8)),
                contrast_temperature=float(task_cfg.get("graphtta_contrast_temperature", 0.1)),
                edge_temperature=float(task_cfg.get("graphtta_edge_temperature", 1.0)),
            )
        elif task_name == "homottt":
            task = HomoTTTConsistencyTask(
                perturb_ratio=float(task_cfg.get("homottt_perturb_ratio", 0.05))
            )
        elif task_name == "homonode":
            task = HomoTTTDropNodeConsistencyTask(
                perturb_ratio=float(task_cfg.get("homottt_perturb_ratio", 0.05))
            )
        elif task_name == "neighbor":
            task = NeighborReconstructionTask(
                hidden_dim=require("hidden_dim", hidden_dim, task_name),
            )
        elif task_name == "pseudolabel":
            task = PseudoLabelConsistencyTask(
                weak_edge_drop=float(task_cfg.get("pseudolabel_weak_edge_drop", 0.05)),
                strong_edge_drop=float(task_cfg.get("pseudolabel_strong_edge_drop", 0.3)),
                strong_node_drop=float(task_cfg.get("pseudolabel_strong_node_drop", 0.15)),
                sharpen_temp=float(task_cfg.get("pseudolabel_sharpen_temp", 0.5)),
                confidence_threshold=float(task_cfg.get("pseudolabel_confidence_threshold", 0.7)),
                entropy_weight=float(task_cfg.get("pseudolabel_entropy_weight", 0.05)),
            )
        elif task_name == "propagation":
            task = PropagationConsistencyTask(
                hidden_dim=require("hidden_dim", hidden_dim, task_name),
            )
        elif task_name == "recon":
            task = EdgeReconstructionTask()
        elif task_name == "residual":
            task = ResidualFeatureReconstructionTask(
                hidden_dim=require("hidden_dim", hidden_dim, task_name),
                input_dim=require("input_dim", input_dim, task_name),
            )
        elif task_name == "smoothness":
            task = EdgeSmoothnessTask(
                hidden_dim=require("hidden_dim", hidden_dim, task_name),
            )
        elif task_name == "entropy":
            task = PredictionEntropyTask(max_samples=int(task_cfg.get("entropy_batch_size", 1000)))
        else:
            raise ValueError(f"Unsupported SSL task: {task_name}")
        if target_device is not None:
            task = task.to(target_device)
        tasks.append(task)
    return nn.ModuleList(tasks)


def compute_ssl_loss(
    tasks: Sequence[nn.Module],
    model,
    feat: torch.Tensor,
    edge_index: torch.Tensor,
    edge_weight=None,
    **task_cfg,
) -> torch.Tensor:
    loss = torch.tensor(0.0, device=feat.device)
    for task in tasks:
        loss = loss + task.compute_loss(
            model=model,
            feat=feat,
            edge_index=edge_index,
            edge_weight=edge_weight,
            **task_cfg,
        )
    return loss


def compute_task_loss_matrix(
    tasks: Sequence[nn.Module],
    model,
    feat: torch.Tensor,
    edge_index: torch.Tensor,
    edge_weight=None,
    mask: torch.Tensor | None = None,
    **task_cfg,
) -> torch.Tensor:
    per_task = []
    for task in tasks:
        node_loss = task.compute_node_loss(
            model=model,
            feat=feat,
            edge_index=edge_index,
            edge_weight=edge_weight,
            **task_cfg,
        )
        if mask is not None:
            node_loss = node_loss[mask]
        per_task.append(node_loss)
    if not per_task:
        return torch.empty((feat.shape[0], 0), device=feat.device)
    return torch.stack(per_task, dim=-1)
