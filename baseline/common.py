from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, SAGEConv


DATA_ROOT = Path("./datasets")


def resolve_dataset_path(dataset: str, domain: str, shift: str) -> Path:
    dataset_key = dataset.lower().replace("_", "-")
    dataset_dirs = {
        "citeseer": ["GOODCiteseer"],
        "cora": ["GOODCora"],
        "pubmed": ["GOODPubmed"],
        "wikics": ["GOODWikiCS", "GOODwikics", "GOODWikics"],
        "ogbn-arxiv": ["GOODArxiv"],
        "arxiv": ["GOODArxiv"],
    }.get(dataset_key)
    if dataset_dirs is None:
        raise ValueError("Unsupported GOOD dataset: {}".format(dataset))

    for dataset_dir in dataset_dirs:
        candidate = DATA_ROOT / dataset_dir / domain / "processed" / "{}.pt".format(shift)
        if candidate.exists():
            return candidate

    searched = [
        str(DATA_ROOT / dataset_dir / domain / "processed" / "{}.pt".format(shift))
        for dataset_dir in dataset_dirs
    ]
    raise FileNotFoundError("GOOD dataset file not found. Searched: {}".format(searched))


def safe_torch_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def extract_masks(data) -> Dict[str, Optional[torch.Tensor]]:
    def _get_mask(*names: str) -> Optional[torch.Tensor]:
        for name in names:
            if hasattr(data, name):
                mask = getattr(data, name)
                if mask is not None:
                    return mask.bool()
        return None

    masks = {
        "train": _get_mask("train_mask"),
        "id_val": _get_mask("id_val_mask"),
        "id_test": _get_mask("id_test_mask"),
        "ood_val": _get_mask("ood_val_mask", "val_mask"),
        "ood_test": _get_mask("ood_test_mask", "test_mask"),
    }
    if masks["ood_test"] is None:
        raise ValueError("Dataset does not contain ood_test/test mask")
    return masks


def load_good_data(dataset: str, domain: str, shift: str, device: torch.device):
    path = resolve_dataset_path(dataset, domain, shift)
    obj = safe_torch_load(path)
    data = obj[0] if isinstance(obj, tuple) else obj
    data = data.to(device)
    masks = extract_masks(data)
    return data, masks, path


class GNNNodeClassifier(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        gnn_type: str,
        num_layers: int,
        dropout: float,
        use_bn: bool,
        classifier_layers: int,
    ):
        super().__init__()
        if int(num_layers) < 1:
            raise ValueError("num_layers must be >= 1")
        if int(classifier_layers) < 1:
            raise ValueError("classifier_layers must be >= 1")

        self.gnn_type = str(gnn_type).lower()
        self.num_layers = int(num_layers)
        self.dropout = float(dropout)
        self.use_bn = bool(use_bn)
        self.classifier_layers = int(classifier_layers)

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        prev_dim = int(in_dim)
        for _ in range(self.num_layers):
            self.convs.append(self._build_conv(prev_dim, int(hidden_dim)))
            self.bns.append(nn.BatchNorm1d(int(hidden_dim)))
            prev_dim = int(hidden_dim)

        self.classifier = self._build_classifier(int(hidden_dim), int(out_dim))

    def _build_conv(self, in_dim: int, out_dim: int):
        if self.gnn_type == "gcn":
            return GCNConv(in_dim, out_dim)
        if self.gnn_type == "gat":
            return GATConv(in_dim, out_dim, heads=1, concat=False)
        if self.gnn_type == "sage":
            return SAGEConv(in_dim, out_dim)
        raise ValueError("Unsupported gnn_type: {}".format(self.gnn_type))

    def _build_classifier(self, hidden_dim: int, out_dim: int):
        if self.classifier_layers == 1:
            return nn.Linear(hidden_dim, out_dim)

        layers: List[nn.Module] = []
        for _ in range(self.classifier_layers - 1):
            layers.extend(
                [
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Dropout(self.dropout),
                ]
            )
        layers.append(nn.Linear(hidden_dim, out_dim))
        return nn.Sequential(*layers)

    def get_embed(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight=None):
        h = x
        for idx, conv in enumerate(self.convs):
            if self.gnn_type == "gcn" and edge_weight is not None:
                h = conv(h, edge_index, edge_weight=edge_weight)
            else:
                h = conv(h, edge_index)
            if self.use_bn:
                h = self.bns[idx](h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        return h

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight=None):
        h = self.get_embed(x, edge_index, edge_weight=edge_weight)
        return self.classifier(h)


def set_seed(seed: int):
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


@torch.no_grad()
def accuracy(logits: torch.Tensor, y: torch.Tensor, mask: Optional[torch.Tensor]) -> float:
    if mask is None or int(mask.sum()) == 0:
        return float("nan")
    pred = logits[mask].argmax(dim=-1)
    return float((pred == y[mask]).float().mean().item())


@torch.no_grad()
def evaluate_model(
    model: GNNNodeClassifier,
    data,
    masks,
    edge_weight=None,
    x: Optional[torch.Tensor] = None,
    edge_index: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    model.eval()
    eval_x = data.x if x is None else x
    eval_edge_index = data.edge_index if edge_index is None else edge_index
    logits = model(eval_x, eval_edge_index, edge_weight=edge_weight)
    return {
        "id_val_acc": accuracy(logits, data.y, masks["id_val"]),
        "id_test_acc": accuracy(logits, data.y, masks["id_test"]),
        "ood_val_acc": accuracy(logits, data.y, masks["ood_val"]),
        "ood_test_acc": accuracy(logits, data.y, masks["ood_test"]),
    }


def load_flat_params(params_file: str) -> Dict[str, object]:
    if not params_file:
        return {}
    with Path(params_file).open("r", encoding="utf-8") as f:
        params = json.load(f)
    if not isinstance(params, dict):
        raise ValueError("params file must be a flat JSON object")
    return params


def normalize_pretrain_model_cfg(pre_ckpt: Dict[str, object], pre_ckpt_path: Path) -> Dict[str, object]:
    model_cfg = dict(pre_ckpt.get("model_cfg", {}))
    params_path = pre_ckpt_path.parent / "params.json"
    extra = {}
    if params_path.exists():
        try:
            extra = load_flat_params(str(params_path))
        except Exception:
            extra = {}

    model_cfg["gnn_type"] = str(model_cfg.get("gnn_type", extra.get("gnn_type", "gcn")))
    model_cfg["classifier_layers"] = int(model_cfg.get("classifier_layers", extra.get("classifier_layers", 1)))
    model_cfg["use_bn"] = bool(model_cfg.get("use_bn", extra.get("use_bn", True)))
    return model_cfg


def pretrain_state_dict(pre_ckpt: Dict[str, object]) -> Dict[str, torch.Tensor]:
    state_dict = pre_ckpt.get("state_dict")
    if state_dict is None:
        state_dict = pre_ckpt.get("model_state_dict")
    if state_dict is None:
        raise KeyError("Pretrain checkpoint does not contain state_dict or model_state_dict")
    return state_dict


def build_model_from_checkpoint(pre_ckpt: Dict[str, object], device: torch.device) -> GNNNodeClassifier:
    model_cfg = pre_ckpt["model_cfg"]
    model = GNNNodeClassifier(
        in_dim=int(model_cfg["in_dim"]),
        hidden_dim=int(model_cfg["hidden_dim"]),
        out_dim=int(model_cfg["out_dim"]),
        gnn_type=str(model_cfg["gnn_type"]),
        num_layers=int(model_cfg["num_layers"]),
        dropout=float(model_cfg["dropout"]),
        use_bn=bool(model_cfg.get("use_bn", True)),
        classifier_layers=int(model_cfg.get("classifier_layers", 1)),
    ).to(device)
    model.load_state_dict(pretrain_state_dict(pre_ckpt))
    return model


def clamp_replace_last_k_layers(replace_last_k_layers: int, num_layers: int) -> int:
    if int(num_layers) <= 0:
        raise ValueError("num_layers must be positive")
    k = max(int(replace_last_k_layers), 1)
    return min(k, int(num_layers))


def set_trainable_blocks(
    model: GNNNodeClassifier,
    replace_last_k_layers: int,
    update_bn_only: bool = False,
) -> List[nn.Parameter]:
    for param in model.parameters():
        param.requires_grad = False

    trainable: List[nn.Parameter] = []
    if update_bn_only:
        for bn in model.bns:
            for param in bn.parameters():
                param.requires_grad = True
                trainable.append(param)
        return trainable

    k = clamp_replace_last_k_layers(replace_last_k_layers, model.num_layers)
    split = int(model.num_layers) - k
    for layer_idx in range(model.num_layers):
        trainable_layer = layer_idx >= split
        for param in model.convs[layer_idx].parameters():
            param.requires_grad = trainable_layer
            if trainable_layer:
                trainable.append(param)
        if model.use_bn:
            for param in model.bns[layer_idx].parameters():
                param.requires_grad = trainable_layer
                if trainable_layer:
                    trainable.append(param)
    return trainable


def base_adaptation_train_cfg(args, num_layers: int) -> Dict[str, object]:
    return {
        "encoder_lr": float(args.encoder_lr),
        "ssl_lr": float(args.ssl_lr),
        "weight_decay": float(args.weight_decay),
        "finetune_epochs": int(args.finetune_epochs),
        "replace_last_k_layers": clamp_replace_last_k_layers(int(args.replace_last_k_layers), int(num_layers)),
    }


def base_adaptation_search_space(trial, args, num_layers: int) -> Dict[str, object]:
    cfg = base_adaptation_train_cfg(args, num_layers)
    cfg.update(
        {
            "encoder_lr": trial.suggest_float("encoder_lr", 1e-5, 1e-2, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-8, 1e-3, log=True),
            "replace_last_k_layers": trial.suggest_int("replace_last_k_layers", 1, max(int(num_layers), 1)),
        }
    )
    return cfg
