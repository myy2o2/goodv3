from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

import torch
import torch.nn as nn

from .bootstrap import BootstrapConsistencyTask
from .homottt import HomoTTTConsistencyTask, HomoTTTDropNodeConsistencyTask
from .neighbor import NeighborReconstructionTask
from .propagation import PropagationConsistencyTask
from .pseudolabel import PseudoLabelConsistencyTask


_TASKS = {
    "bootstrap": BootstrapConsistencyTask,
    "homottt": HomoTTTConsistencyTask,
    "homonode": HomoTTTDropNodeConsistencyTask,
    "neighbor": NeighborReconstructionTask,
    "pseudolabel": PseudoLabelConsistencyTask,
    "propagation": PropagationConsistencyTask,
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
        raise ValueError("Unsupported SSL task: {}. Available tasks: {}".format(task_name, available_ssl_tasks()))
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


def _require(name: str, value: int | None, task_name: str) -> int:
    if value is None:
        raise ValueError("{} requires {} to build its task head.".format(task_name, name))
    return int(value)


def build_ssl_tasks(
    task_spec: str | Sequence[str] | None,
    task_cfg: Mapping[str, object] | None = None,
    device: torch.device | str | None = None,
    hidden_dim: int | None = None,
    input_dim: int | None = None,
) -> nn.ModuleList:
    task_cfg = dict(task_cfg or {})
    target_device = None if device is None else torch.device(device)

    tasks = []
    for task_name in parse_ssl_task_names(task_spec):
        if task_name == "bootstrap":
            task = BootstrapConsistencyTask(
                hidden_dim=_require("hidden_dim", hidden_dim, task_name),
                edge_drop=float(task_cfg.get("bootstrap_edge_drop", 0.2)),
                node_drop=float(task_cfg.get("bootstrap_node_drop", 0.1)),
                predictor_hidden=int(task_cfg.get("bootstrap_predictor_hidden", 0)),
            )
        elif task_name == "homottt":
            task = HomoTTTConsistencyTask(
                perturb_ratio=float(task_cfg.get("homottt_perturb_ratio", 0.05)),
            )
        elif task_name == "homonode":
            task = HomoTTTDropNodeConsistencyTask(
                perturb_ratio=float(task_cfg.get("homottt_perturb_ratio", 0.05)),
            )
        elif task_name == "neighbor":
            task = NeighborReconstructionTask(
                hidden_dim=_require("hidden_dim", hidden_dim, task_name),
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
                hidden_dim=_require("hidden_dim", hidden_dim, task_name),
            )
        else:
            raise ValueError("Unsupported SSL task: {}".format(task_name))
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
        size = int(mask.sum().item()) if mask is not None else int(feat.shape[0])
        return torch.empty((size, 0), device=feat.device)
    return torch.stack(per_task, dim=-1)
