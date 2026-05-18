from __future__ import annotations

import argparse
import copy
import json
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, SAGEConv

try:
    import optuna
except Exception:
    optuna = None


DATA_ROOT = Path("./datasets")


def resolve_dataset_path(dataset: str, domain: str, shift: str) -> Path:
    dataset_key = dataset.lower().replace("_", "-") # 统一格式，便于匹配
    dataset_dirs = {
        "citeseer": ["GOODCiteseer"],
        "cora": ["GOODCora"],
        "pubmed": ["GOODPubmed"],
        "wikics": ["GOODWikiCS", "GOODwikics"],
        "ogbn-arxiv": ["GOODArxiv"],
        "arxiv": ["GOODArxiv"],
    }.get(dataset_key)
    if dataset_dirs is None:
        raise ValueError("Unsupported GOOD dataset: {}".format(dataset))

    for dataset_dir in dataset_dirs:
        candidate = DATA_ROOT / dataset_dir / domain / "processed" / "{}.pt".format(shift)
        if candidate.exists():
            return candidate
    searched = [str(DATA_ROOT / dataset_dir / domain / "processed" / "{}.pt".format(shift)) for dataset_dir in dataset_dirs]
    raise FileNotFoundError("GOOD dataset file not found. Searched: {}".format(searched))


def safe_torch_load(path: Path): # 新版 torch 可用，torch兼容
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def extract_masks(data) -> Dict[str, torch.Tensor]:
    masks = {}

    def _get_mask(*names: str) -> Optional[torch.Tensor]:
        for name in names:
            if hasattr(data, name):
                m = getattr(data, name)
                if m is not None:
                    return m.bool()
        return None

    train_mask = _get_mask("train_mask")
    if train_mask is None:
        raise ValueError("Dataset does not contain train_mask")

    masks["train"] = train_mask
    masks["id_val"] = _get_mask("id_val_mask")
    masks["id_test"] = _get_mask("id_test_mask")
    masks["ood_val"] = _get_mask("ood_val_mask", "val_mask")  # 优先 ood_val，没有则 val
    masks["ood_test"] = _get_mask("ood_test_mask", "test_mask") # 优先 ood_test，没有则 test
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
        classifier_layers: int, # 分类头 MLP 层数超参
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        self.gnn_type = gnn_type.lower()
        self.num_layers = int(num_layers)
        self.dropout = float(dropout)
        self.use_bn = bool(use_bn)
        self.classifier_layers = int(classifier_layers)
        if self.classifier_layers < 1:
            raise ValueError("classifier_layers must be >= 1")

        self.convs = nn.ModuleList() # GNN 主干
        self.bns = nn.ModuleList() # 每层对应 BN
        prev_dim = in_dim
        for _ in range(self.num_layers):
            self.convs.append(self._build_conv(prev_dim, hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))
            prev_dim = hidden_dim

        self.classifier = self._build_classifier(hidden_dim, out_dim)

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
            return nn.Linear(hidden_dim, out_dim) # 单层线性头
        layers = []
        for _ in range(self.classifier_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(self.dropout))
        layers.append(nn.Linear(hidden_dim, out_dim)) # 输出层
        return nn.Sequential(*layers)

    def get_embed(self, x: torch.Tensor, edge_index: torch.Tensor):
        h = x
        for idx, conv in enumerate(self.convs):
            h = conv(h, edge_index)
            if self.use_bn:
                h = self.bns[idx](h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        return h

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        h = self.get_embed(x, edge_index)
        return self.classifier(h)


@torch.no_grad()
def accuracy(logits: torch.Tensor, y: torch.Tensor, mask: Optional[torch.Tensor]) -> float: # 预测类别
    if mask is None or int(mask.sum()) == 0:
        return float("nan")
    pred = logits[mask].argmax(dim=-1)
    return float((pred == y[mask]).float().mean().item())


def train_once(args, data, masks, params: Dict[str, float], trial=None):
    model = GNNNodeClassifier(
        in_dim=data.x.shape[1],
        hidden_dim=int(params["hidden_dim"]),
        out_dim=int(data.y.max().item()) + 1,
        gnn_type=str(params.get("gnn_type", args.gnn_type)),
        num_layers=int(params["num_layers"]),
        dropout=float(params["dropout"]),
        use_bn=bool(params.get("use_bn", args.use_bn)),
        classifier_layers=int(params.get("classifier_layers", args.classifier_layers)),
    ).to(data.x.device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(params["lr"]),
        weight_decay=float(params["weight_decay"]),
    )

    best_val = -1.0
    best_state = None
    wait = 0
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(data.x, data.edge_index)
        loss = F.cross_entropy(logits[masks["train"]], data.y[masks["train"]]) # 仅 train节点监督，回传误差
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            eval_logits = model(data.x, data.edge_index)
            id_val_acc = accuracy(eval_logits, data.y, masks["id_val"])

        if trial is not None and not np.isnan(id_val_acc):
            trial.report(id_val_acc, step=epoch)

        if not np.isnan(id_val_acc) and id_val_acc > best_val:
            best_val = id_val_acc
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= int(args.patience):
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
    metrics = {
        "train_acc": accuracy(logits, data.y, masks["train"]),
        "id_val_acc": accuracy(logits, data.y, masks["id_val"]),
        "id_test_acc": accuracy(logits, data.y, masks["id_test"]),
        "ood_val_acc": accuracy(logits, data.y, masks["ood_val"]),
        "ood_test_acc": accuracy(logits, data.y, masks["ood_test"]),
    }
    return model, metrics


def build_search_space(args, trial):
    return {
        "hidden_dim": trial.suggest_categorical("hidden_dim", [32,64,128]),
        "num_layers": trial.suggest_int("num_layers", 3,5),
        "dropout": trial.suggest_float("dropout", 0.0, 0.5),
        "lr": trial.suggest_float("lr", 1e-4, 1e-1, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-8, 1e-2, log=True),
    }


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_stage_output_dir(
    output_root: str,
    stage_name: str,
    dataset: str,
    domain: str,
    shift: str,
    timestamp: str = "",
) -> Path:
    ts = timestamp if timestamp else datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = Path(output_root) / stage_name / dataset.lower() / domain.lower() / shift.lower() / ts
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def load_flat_params(params_file: str) -> Dict[str, object]:
    if not params_file:
        return {}
    with Path(params_file).open("r", encoding="utf-8") as f:
        params = json.load(f)
    if not isinstance(params, dict):
        raise ValueError("params file must be a flat JSON object")
    return params


def parse_args():
    parser = argparse.ArgumentParser(description="Stage-1 pretrain for GOOD node classification")
    parser.add_argument("--dataset", type=str, default="cora")
    parser.add_argument("--domain", type=str, default="word", choices=["word", "degree"])
    parser.add_argument("--shift", type=str, default="covariate", choices=["covariate", "concept", "no_shift"])

    parser.add_argument("--gnn-type", type=str, default="gcn", choices=["gcn", "gat", "sage"])
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--classifier-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--use-bn", action="store_true", default=True)
    parser.add_argument("--no-bn", action="store_false", dest="use_bn")

    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=50)

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    parser.add_argument("--use-optuna", action="store_true")
    parser.add_argument("--optuna-trials", type=int, default=30)
    parser.add_argument("--optuna-timeout", type=int, default=0)

    parser.add_argument("--output-root", type=str, default="./outputs")
    parser.add_argument("--timestamp", type=str, default="")
    parser.add_argument("--run-name", type=str, default="pretrain")
    parser.add_argument("--params-file", type=str, default="")

    prelim = parser.parse_known_args()[0]
    if prelim.params_file:
        parser.set_defaults(**load_flat_params(prelim.params_file))
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(int(args.seed))
    device = torch.device(args.device)
    data, masks, data_path = load_good_data(args.dataset, args.domain, args.shift, device)

    if args.use_optuna:
        if optuna is None:
            raise RuntimeError("Optuna is not installed, but --use-optuna is set")

        best_trial_score = float("-inf")
        best_trial_state_dict = None
        best_trial_metrics = None

        def objective(trial):
            nonlocal best_trial_score, best_trial_state_dict, best_trial_metrics

            # Deterministic but distinct seed per trial.
            set_seed(int(args.seed) + int(trial.number))
            searched = build_search_space(args, trial)
            trial_params = {
                "gnn_type": args.gnn_type,
                "classifier_layers": args.classifier_layers,
                "use_bn": args.use_bn,
                **searched,
            }
            model_trial, metrics = train_once(args, data, masks, trial_params, trial=trial)
            score =  (metrics["id_test_acc"]) / 1.0
            if np.isnan(score):
                score = -1.0

            if score > best_trial_score:
                best_trial_score = score
                best_trial_state_dict = copy.deepcopy(model_trial.state_dict())
                best_trial_metrics = dict(metrics)
            return score

        timeout = None if int(args.optuna_timeout) <= 0 else int(args.optuna_timeout)
        study = optuna.create_study(direction="maximize", pruner=optuna.pruners.NopPruner())
        study.optimize(objective, n_trials=int(args.optuna_trials), timeout=timeout)
        searched_params = dict(study.best_params)
        if best_trial_state_dict is None or best_trial_metrics is None:
            raise RuntimeError("Optuna finished without a valid best trial to save.")
        selected_state_dict = best_trial_state_dict
        metrics = best_trial_metrics
    else:
        searched_params = {
            "hidden_dim": args.hidden_dim,
            "num_layers": args.num_layers,
            "dropout": args.dropout,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
        }

    best_params = {
        "gnn_type": args.gnn_type,
        "classifier_layers": args.classifier_layers,
        "use_bn": args.use_bn,
        **searched_params,
    }
    if not args.use_optuna:
        set_seed(int(args.seed))
        model, metrics = train_once(args, data, masks, best_params)
        selected_state_dict = model.state_dict()

    out_dir = make_stage_output_dir(
        args.output_root,
        "stage1",
        args.dataset,
        args.domain,
        args.shift,
        timestamp=args.timestamp,
    )
    ckpt_path = out_dir / "pretrain_model.pt"
    result_path = out_dir / "metrics.json"
    params_path = out_dir / "params.json"

    torch.save(
        {
            "stage": "pretrain",
            "dataset": args.dataset,
            "domain": args.domain,
            "shift": args.shift,
            "data_path": str(data_path),
            "model_cfg": {
                "gnn_type": str(best_params["gnn_type"]),
                "hidden_dim": int(best_params["hidden_dim"]),
                "num_layers": int(best_params["num_layers"]),
                "classifier_layers": int(best_params["classifier_layers"]),
                "dropout": float(best_params["dropout"]),
                "use_bn": bool(best_params["use_bn"]),
                "in_dim": int(data.x.shape[1]),
                "out_dim": int(data.y.max().item()) + 1,
            },
            "optim_cfg": {
                "lr": float(best_params["lr"]),
                "weight_decay": float(best_params["weight_decay"]),
            },
            "state_dict": selected_state_dict,
            "metrics": metrics,
        },
        ckpt_path,
    )

    with result_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    run_params = {
        "params_file": args.params_file,
        "dataset": args.dataset,
        "domain": args.domain,
        "shift": args.shift,
        "gnn_type": str(best_params["gnn_type"]),
        "hidden_dim": int(best_params["hidden_dim"]),
        "num_layers": int(best_params["num_layers"]),
        "classifier_layers": int(best_params["classifier_layers"]),
        "dropout": float(best_params["dropout"]),
        "use_bn": bool(best_params["use_bn"]),
        "lr": float(best_params["lr"]),
        "weight_decay": float(best_params["weight_decay"]),
        "use_optuna": bool(args.use_optuna),
        "optuna_trials": int(args.optuna_trials),
        "optuna_timeout": int(args.optuna_timeout),
        "epochs": int(args.epochs),
        "patience": int(args.patience),
        "seed": int(args.seed),
        "device": str(args.device),
        "output_root": str(args.output_root),
        "timestamp": str(args.timestamp),
        "run_name": str(args.run_name),
        "data_path": str(data_path),
        "output_dir": str(out_dir),
    }
    with params_path.open("w", encoding="utf-8") as f:
        json.dump(run_params, f, indent=2)

    print("Saved checkpoint:", ckpt_path)
    print("Saved params:", params_path)
    print("Metrics:", json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
