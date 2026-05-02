from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, SAGEConv
from torch_geometric.utils import degree

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ssl_tasks import build_ssl_tasks, parse_ssl_task_names

try:
    import optuna
except Exception:
    optuna = None


DATA_ROOT = Path("./datasets")
METHODS = ("stage1", "eerm", "gtrans", "tent")


def resolve_dataset_path(dataset: str, domain: str, shift: str) -> Path:
    dataset_key = dataset.lower().replace("_", "-")
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


def safe_torch_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def extract_masks(data) -> Dict[str, torch.Tensor]:
    def _get_mask(*names: str) -> Optional[torch.Tensor]:
        for name in names:
            if hasattr(data, name):
                m = getattr(data, name)
                if m is not None:
                    return m.bool()
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
        self.gnn_type = gnn_type.lower()
        self.num_layers = int(num_layers)
        self.dropout = float(dropout)
        self.use_bn = bool(use_bn)
        self.classifier_layers = int(classifier_layers)
        if self.classifier_layers < 1:
            raise ValueError("classifier_layers must be >= 1")

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
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
            return nn.Linear(hidden_dim, out_dim)
        layers: List[nn.Module] = []
        for _ in range(self.classifier_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True), nn.Dropout(self.dropout)])
        layers.append(nn.Linear(hidden_dim, out_dim))
        return nn.Sequential(*layers)

    def get_embed(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight=None):
        del edge_weight
        h = x
        for idx, conv in enumerate(self.convs):
            h = conv(h, edge_index)
            if self.use_bn:
                h = self.bns[idx](h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        return h

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight=None):
        return self.classifier(self.get_embed(x, edge_index, edge_weight=edge_weight))


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def accuracy(logits: torch.Tensor, y: torch.Tensor, mask: Optional[torch.Tensor]) -> float:
    if mask is None or int(mask.sum()) == 0:
        return float("nan")
    pred = logits[mask].argmax(dim=-1)
    return float((pred == y[mask]).float().mean().item())


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


def clamp_replace_last_k_layers(replace_last_k_layers: int, num_layers: int) -> int:
    if num_layers <= 0:
        raise ValueError("num_layers must be positive")
    k = int(replace_last_k_layers)
    if k < 1:
        k = 1
    if k > int(num_layers):
        k = int(num_layers)
    return k


def set_trainable_blocks(
    model: GNNNodeClassifier,
    replace_last_k_layers: int,
    update_bn_only: bool,
) -> List[nn.Parameter]:
    trainable: List[nn.Parameter] = []
    k = clamp_replace_last_k_layers(replace_last_k_layers, model.num_layers)
    split = int(model.num_layers) - k

    if update_bn_only:
        for conv in model.convs:
            for p in conv.parameters():
                p.requires_grad = False
        for bn in model.bns:
            for p in bn.parameters():
                p.requires_grad = True
                trainable.append(p)
        return trainable

    for i in range(model.num_layers):
        trainable_flag = i >= split
        for p in model.convs[i].parameters():
            p.requires_grad = trainable_flag
            if trainable_flag:
                trainable.append(p)
        if model.use_bn:
            for p in model.bns[i].parameters():
                p.requires_grad = trainable_flag
                if trainable_flag:
                    trainable.append(p)
    return trainable


def evaluate_model(model: GNNNodeClassifier, data, masks) -> Dict[str, float]:
    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
    return {
        "id_val_acc": accuracy(logits, data.y, masks["id_val"]),
        "id_test_acc": accuracy(logits, data.y, masks["id_test"]),
        "ood_val_acc": accuracy(logits, data.y, masks["ood_val"]),
        "ood_test_acc": accuracy(logits, data.y, masks["ood_test"]),
    }


def entropy_loss(logits: torch.Tensor) -> torch.Tensor:
    p = logits.softmax(dim=-1)
    return -(p * logits.log_softmax(dim=-1)).sum(dim=-1)


def split_env_masks_by_degree(data, base_mask: torch.Tensor, num_envs: int) -> List[torch.Tensor]:
    num_envs = max(int(num_envs), 1)
    deg = degree(data.edge_index[0], num_nodes=data.x.shape[0]).to(base_mask.device)
    selected = torch.where(base_mask)[0]
    if selected.numel() == 0:
        return [base_mask]
    deg_sel = deg[selected]
    order = torch.argsort(deg_sel)
    chunks = torch.chunk(order, num_envs)

    env_masks: List[torch.Tensor] = []
    for chunk in chunks:
        m = torch.zeros_like(base_mask)
        if chunk.numel() > 0:
            node_ids = selected[chunk]
            m[node_ids] = True
        env_masks.append(m)
    env_masks = [m for m in env_masks if int(m.sum()) > 0]
    return env_masks if env_masks else [base_mask]


def build_ssl_tasks_for_gtrans(args, device: torch.device, hidden_dim: int, input_dim: int):
    task_cfg = json.loads(args.task_cfg_json) if args.task_cfg_json else {}
    task_names = parse_ssl_task_names(args.ssl_tasks)
    if not task_names:
        raise ValueError("No SSL tasks selected for gtrans baseline")
    tasks = build_ssl_tasks(
        task_names,
        task_cfg=task_cfg,
        device=device,
        hidden_dim=hidden_dim,
        input_dim=input_dim,
    )
    return tasks, task_names, task_cfg


def compute_method_loss(method: str, model: GNNNodeClassifier, data, masks, ssl_tasks, train_cfg: Dict[str, float]) -> torch.Tensor:
    ood_mask = masks["ood_test"]

    if method == "tent":
        logits = model(data.x, data.edge_index)
        node_loss = entropy_loss(logits)
        return node_loss[ood_mask].mean()

    if method == "gtrans":
        total = torch.tensor(0.0, device=data.x.device)
        for task in ssl_tasks:
            node_loss = task.compute_node_loss(model, data.x, data.edge_index)
            total = total + node_loss[ood_mask].mean()
        return total

    if method == "eerm":
        logits = model(data.x, data.edge_index)
        node_loss = entropy_loss(logits)
        env_masks = split_env_masks_by_degree(data, ood_mask, int(train_cfg["eerm_num_envs"]))
        env_losses = []
        for env_mask in env_masks:
            env_losses.append(node_loss[env_mask].mean())
        env_stack = torch.stack(env_losses)
        mean_loss = env_stack.mean()
        var_loss = env_stack.var(unbiased=False)
        return mean_loss + float(train_cfg["eerm_var_lambda"]) * var_loss

    raise ValueError("Unsupported baseline method: {}".format(method))


def run_method_once(args, method: str, data, masks, pre_ckpt, train_cfg: Dict[str, float], verbose: bool = True):
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
    ).to(data.x.device)
    model.load_state_dict(pre_ckpt["state_dict"])

    if method == "stage1":
        best_metrics = evaluate_model(model, data, masks)
        history = [{"epoch": 0, **best_metrics}]
        if verbose:
            print("[baseline:stage1] ood_test_acc={:.6f}".format(float(best_metrics["ood_test_acc"])))
        artifacts = {
            "method": method,
            "task_names": [],
            "task_cfg": {},
            "history": history,
            "model_cfg": model_cfg,
            "model_state_dict": model.state_dict(),
        }
        return best_metrics, artifacts

    update_bn_only = bool(train_cfg.get("tent_update_bn_only", False) and method == "tent")
    trainable_params = set_trainable_blocks(
        model,
        replace_last_k_layers=int(train_cfg["replace_last_k_layers"]),
        update_bn_only=update_bn_only,
    )
    if len(trainable_params) == 0:
        raise RuntimeError("No trainable parameters selected for method {}".format(method))

    ssl_tasks = None
    task_names: List[str] = []
    task_cfg: Dict[str, object] = {}
    param_groups = [{"params": trainable_params, "lr": float(train_cfg["encoder_lr"])}]

    if method == "gtrans":
        ssl_tasks, task_names, task_cfg = build_ssl_tasks_for_gtrans(
            args=args,
            device=data.x.device,
            hidden_dim=int(model_cfg["hidden_dim"]),
            input_dim=int(model_cfg["in_dim"]),
        )
        ssl_params = list(ssl_tasks.parameters())
        if len(ssl_params) > 0:
            param_groups.append({"params": ssl_params, "lr": float(train_cfg["ssl_lr"])})

    optimizer = torch.optim.Adam(param_groups, weight_decay=float(train_cfg["weight_decay"]))

    best_score = float("-inf")
    best_state = copy.deepcopy(model.state_dict())
    best_metrics = evaluate_model(model, data, masks)
    history = [{"epoch": 0, **best_metrics}]

    for epoch in range(1, int(train_cfg["finetune_epochs"]) + 1):
        model.train()
        if ssl_tasks is not None:
            ssl_tasks.train()
        optimizer.zero_grad()

        loss = compute_method_loss(
            method=method,
            model=model,
            data=data,
            masks=masks,
            ssl_tasks=ssl_tasks,
            train_cfg=train_cfg,
        )
        if loss.requires_grad:
            loss.backward()
            optimizer.step()

        metrics = evaluate_model(model, data, masks)
        history.append({"epoch": int(epoch), **metrics})
        if not np.isnan(metrics["ood_test_acc"]) and metrics["ood_test_acc"] > best_score:
            best_score = float(metrics["ood_test_acc"])
            best_state = copy.deepcopy(model.state_dict())
            best_metrics = dict(metrics)

    model.load_state_dict(best_state)
    if verbose:
        print("[baseline:{}] best_ood_test_acc={:.6f}".format(method, float(best_metrics["ood_test_acc"])))

    artifacts = {
        "method": method,
        "task_names": task_names,
        "task_cfg": task_cfg,
        "history": history,
        "model_cfg": model_cfg,
        "model_state_dict": model.state_dict(),
    }
    return best_metrics, artifacts


def build_search_space(method: str, trial, args, num_layers: int) -> Dict[str, float]:
    max_layers = max(int(num_layers), 1)
    cfg: Dict[str, float] = {
        "encoder_lr": trial.suggest_float("encoder_lr", 1e-5, 1e-2, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-8, 1e-3, log=True),
        "finetune_epochs": int(args.finetune_epochs),
        "replace_last_k_layers": trial.suggest_int("replace_last_k_layers", 1, max_layers),
        "ssl_lr": float(args.ssl_lr),
        "eerm_num_envs": int(args.eerm_num_envs),
        "eerm_var_lambda": float(args.eerm_var_lambda),
        "tent_update_bn_only": bool(args.tent_update_bn_only),
    }

    if method == "gtrans":
        cfg["ssl_lr"] = trial.suggest_float("ssl_lr", 1e-5, 1e-3, log=True)
    if method == "eerm":
        cfg["eerm_num_envs"] = int(trial.suggest_int("eerm_num_envs", 2, 6))
        cfg["eerm_var_lambda"] = trial.suggest_float("eerm_var_lambda", 1e-3, 2.0, log=True)
    if method == "tent":
        cfg["tent_update_bn_only"] = bool(trial.suggest_categorical("tent_update_bn_only", [True, False]))

    return cfg


def default_train_cfg(args, method: str, num_layers: int) -> Dict[str, float]:
    return {
        "encoder_lr": float(args.encoder_lr),
        "weight_decay": float(args.weight_decay),
        "finetune_epochs": int(args.finetune_epochs),
        "replace_last_k_layers": clamp_replace_last_k_layers(int(args.replace_last_k_layers), num_layers),
        "ssl_lr": float(args.ssl_lr),
        "eerm_num_envs": int(args.eerm_num_envs),
        "eerm_var_lambda": float(args.eerm_var_lambda),
        "tent_update_bn_only": bool(args.tent_update_bn_only),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Isolated baseline runner for EERM/GTrans/Tent with Optuna")
    parser.add_argument("--pretrain-ckpt", type=str, default="")
    parser.add_argument("--dataset", type=str, default="")
    parser.add_argument("--domain", type=str, default="")
    parser.add_argument("--shift", type=str, default="")

    parser.add_argument("--method", type=str, default="all", choices=["all", "stage1", "eerm", "gtrans", "tent"])
    parser.add_argument("--include-stage1-baseline", action="store_true")
    parser.add_argument("--ssl-tasks", type=str, default="homottt")
    parser.add_argument("--task-cfg-json", type=str, default="{}")

    parser.add_argument("--ssl-lr", type=float, default=1e-3)
    parser.add_argument("--encoder-lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--finetune-epochs", type=int, default=50)
    parser.add_argument("--replace-last-k-layers", type=int, default=1)

    parser.add_argument("--eerm-num-envs", type=int, default=3)
    parser.add_argument("--eerm-var-lambda", type=float, default=0.1)
    parser.add_argument("--tent-update-bn-only", action="store_true")

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    parser.add_argument("--use-optuna", action="store_true")
    parser.add_argument("--optuna-trials", type=int, default=50)
    parser.add_argument("--optuna-timeout", type=int, default=0)

    parser.add_argument("--output-root", type=str, default="./outputs_baseline")
    parser.add_argument("--timestamp", type=str, default="")
    parser.add_argument("--run-name", type=str, default="baseline_compare")
    parser.add_argument("--params-file", type=str, default="")

    prelim = parser.parse_known_args()[0]
    if prelim.params_file:
        parser.set_defaults(**load_flat_params(prelim.params_file))
    return parser.parse_args()


def run_one_method(args, method: str, data, masks, pre_ckpt, base_out_dir: Path):
    num_layers = int(pre_ckpt["model_cfg"]["num_layers"])

    if method == "stage1":
        best_train_cfg = default_train_cfg(args, method, num_layers)
        metrics, artifacts = run_method_once(
            args=args,
            method=method,
            data=data,
            masks=masks,
            pre_ckpt=pre_ckpt,
            train_cfg=best_train_cfg,
            verbose=True,
        )
    elif args.use_optuna:
        if optuna is None:
            raise RuntimeError("Optuna is not installed, but --use-optuna is set")

        best_trial_score = float("-inf")
        best_trial_metrics = None
        best_trial_artifacts = None
        best_train_cfg = None

        def objective(trial):
            nonlocal best_trial_score, best_trial_metrics, best_trial_artifacts, best_train_cfg
            set_seed(int(args.seed) + int(trial.number))
            train_cfg = build_search_space(method, trial, args, num_layers)
            metrics_tmp, artifacts_tmp = run_method_once(
                args=args,
                method=method,
                data=data,
                masks=masks,
                pre_ckpt=pre_ckpt,
                train_cfg=train_cfg,
                verbose=False,
            )
            score = metrics_tmp["ood_test_acc"]
            if np.isnan(score):
                score = -1.0
            if score > best_trial_score:
                best_trial_score = float(score)
                best_trial_metrics = copy.deepcopy(metrics_tmp)
                best_trial_artifacts = copy.deepcopy(artifacts_tmp)
                best_train_cfg = dict(train_cfg)
            return score

        timeout = None if int(args.optuna_timeout) <= 0 else int(args.optuna_timeout)
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=int(args.optuna_trials), timeout=timeout)
        if best_trial_metrics is None or best_trial_artifacts is None or best_train_cfg is None:
            raise RuntimeError("Optuna finished without a valid best trial result for method {}".format(method))
        metrics = best_trial_metrics
        artifacts = best_trial_artifacts
    else:
        best_train_cfg = default_train_cfg(args, method, num_layers)
        metrics, artifacts = run_method_once(
            args=args,
            method=method,
            data=data,
            masks=masks,
            pre_ckpt=pre_ckpt,
            train_cfg=best_train_cfg,
            verbose=True,
        )

    method_dir = base_out_dir / method
    method_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = method_dir / "baseline_model.pt"
    metrics_path = method_dir / "metrics.json"
    params_path = method_dir / "params.json"

    torch.save(
        {
            "stage": "baseline_compare",
            "method": method,
            "pretrain_ckpt": str(args.pretrain_ckpt),
            "dataset": args.dataset,
            "domain": args.domain,
            "shift": args.shift,
            "task_names": artifacts["task_names"],
            "task_cfg": artifacts["task_cfg"],
            "model_cfg": artifacts["model_cfg"],
            "train_cfg": best_train_cfg,
            "metrics": metrics,
            "history": artifacts["history"],
            "state_dict": artifacts["model_state_dict"],
        },
        ckpt_path,
    )

    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    run_params = {
        "params_file": args.params_file,
        "method": method,
        "dataset": args.dataset,
        "domain": args.domain,
        "shift": args.shift,
        "pretrain_ckpt": str(args.pretrain_ckpt),
        "ssl_tasks": str(args.ssl_tasks),
        "task_cfg_json": str(args.task_cfg_json),
        "encoder_lr": float(best_train_cfg["encoder_lr"]),
        "ssl_lr": float(best_train_cfg["ssl_lr"]),
        "weight_decay": float(best_train_cfg["weight_decay"]),
        "finetune_epochs": int(best_train_cfg["finetune_epochs"]),
        "replace_last_k_layers": int(best_train_cfg["replace_last_k_layers"]),
        "eerm_num_envs": int(best_train_cfg["eerm_num_envs"]),
        "eerm_var_lambda": float(best_train_cfg["eerm_var_lambda"]),
        "tent_update_bn_only": bool(best_train_cfg["tent_update_bn_only"]),
        "use_optuna": bool(args.use_optuna),
        "optuna_trials": int(args.optuna_trials),
        "optuna_timeout": int(args.optuna_timeout),
        "seed": int(args.seed),
        "device": str(args.device),
        "output_root": str(args.output_root),
        "output_dir": str(method_dir),
    }
    with params_path.open("w", encoding="utf-8") as f:
        json.dump(run_params, f, indent=2)

    return {
        "method": method,
        "ood_test_acc": float(metrics["ood_test_acc"]),
        "metrics": metrics,
        "train_cfg": best_train_cfg,
        "ckpt_path": str(ckpt_path),
        "metrics_path": str(metrics_path),
        "params_path": str(params_path),
    }


def main():
    args = parse_args()
    if not args.pretrain_ckpt:
        raise ValueError("--pretrain-ckpt is required (or provide it via --params-file)")

    set_seed(int(args.seed))
    device = torch.device(args.device)

    pre_ckpt_path = Path(args.pretrain_ckpt)
    pre_ckpt = safe_torch_load(pre_ckpt_path)
    pre_ckpt["model_cfg"] = normalize_pretrain_model_cfg(pre_ckpt, pre_ckpt_path)

    dataset = args.dataset or pre_ckpt["dataset"]
    domain = args.domain or pre_ckpt["domain"]
    shift = args.shift or pre_ckpt["shift"]
    args.dataset = dataset
    args.domain = domain
    args.shift = shift

    data, masks, data_path = load_good_data(dataset, domain, shift, device)

    ts = args.timestamp if args.timestamp else datetime.now().strftime("%Y%m%d-%H%M%S")
    base_out_dir = Path(args.output_root) / "baseline_compare" / dataset.lower() / domain.lower() / shift.lower() / ts
    base_out_dir.mkdir(parents=True, exist_ok=True)

    if args.method == "all":
        methods = ["eerm", "gtrans", "tent"]
        if args.include_stage1_baseline:
            methods = ["stage1"] + methods
    else:
        methods = [args.method]
    results = []
    for method in methods:
        print("\n========== Baseline Method: {} ==========".format(method))
        result = run_one_method(
            args=args,
            method=method,
            data=data,
            masks=masks,
            pre_ckpt=pre_ckpt,
            base_out_dir=base_out_dir,
        )
        results.append(result)
        print("[{}] ood_test_acc={:.6f}".format(method, float(result["ood_test_acc"])))

    best = max(results, key=lambda x: float(x["ood_test_acc"]))

    summary = {
        "run_name": str(args.run_name),
        "dataset": dataset,
        "domain": domain,
        "shift": shift,
        "data_path": str(data_path),
        "pretrain_ckpt": str(args.pretrain_ckpt),
        "methods": methods,
        "best_method": best["method"],
        "best_ood_test_acc": float(best["ood_test_acc"]),
        "results": results,
    }

    summary_path = base_out_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nSaved baseline summary:", summary_path)
    print("Best method:", best["method"])
    print("Best ood_test_acc:", float(best["ood_test_acc"]))
    print("Summary:", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
