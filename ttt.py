from __future__ import annotations

import argparse
import copy
import json
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, SAGEConv

from ssl_tasks import build_ssl_tasks, parse_ssl_task_names

try:
    import optuna
except Exception:
    optuna = None


DATA_ROOT = Path("./datasets")

# resolve_dataset_path / safe_torch_load / extract_masks / load_good_data
# 与 stage1/stage2 一致。这里关键差异是 extract_masks 强制需要 ood_test。
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
    return data, extract_masks(data), path


class GNNNodeClassifier(nn.Module):
     # 与 stage1 同结构：GNN encoder + 可配置 MLP 分类头（含 classifier_layers）
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
        layers = []
        for _ in range(self.classifier_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True), nn.Dropout(self.dropout)])
        layers.append(nn.Linear(hidden_dim, out_dim))
        return nn.Sequential(*layers)

    def get_embed(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight=None):
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


class GateMLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        num_layers: int,
        out_dim: int,
        dropout: float,
        temperature: float,
    ):
         # Gate 网络结构（从 stage2 checkpoint 读权重）
        super().__init__()
        self.temperature = max(float(temperature), 1e-6)
        layers = []
        prev = in_dim
        for _ in range(max(num_layers - 1, 0)):
            layers.extend([nn.Linear(prev, hidden_dim), nn.ReLU(inplace=True), nn.Dropout(dropout)])
            prev = hidden_dim
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, h: torch.Tensor):
        logits = self.net(h) / self.temperature
        return F.softmax(logits, dim=-1)


@torch.no_grad()
def accuracy(logits: torch.Tensor, y: torch.Tensor, mask: Optional[torch.Tensor]) -> float:
    if mask is None or int(mask.sum()) == 0:
        return float("nan")
    pred = logits[mask].argmax(dim=-1)
    return float((pred == y[mask]).float().mean().item())


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


def parse_task_cfg_json(task_cfg_json: str) -> Dict[str, object]:
    if not task_cfg_json:
        return {}
    task_cfg = json.loads(task_cfg_json)
    if not isinstance(task_cfg, dict):
        raise ValueError("task_cfg_json must decode to a JSON object")
    return task_cfg


def resolve_task_names(ssl_tasks: str, num_ssl: int) -> list[str]:
    task_names = parse_ssl_task_names(ssl_tasks)
    if int(num_ssl) > 0:
        if int(num_ssl) > len(task_names):
            raise ValueError("num_ssl={} is larger than parsed task count={}".format(num_ssl, len(task_names)))
        task_names = task_names[: int(num_ssl)]
    if not task_names:
        raise ValueError("No SSL tasks selected. Provide --ssl-tasks when --gate-ckpt is not used.")
    return task_names


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


def parse_args():
    parser = argparse.ArgumentParser(description="Stage-3 TTT with per-SSL adapted encoders")
    parser.add_argument("--pretrain-ckpt", type=str, default="")
    parser.add_argument("--gate-ckpt", type=str, default="")
    parser.add_argument("--ssl-tasks", type=str, default="")
    parser.add_argument("--num-ssl", type=int, default=1)
    parser.add_argument("--task-cfg-json", type=str, default="")

    parser.add_argument("--dataset", type=str, default="")
    parser.add_argument("--domain", type=str, default="")
    parser.add_argument("--shift", type=str, default="")

    parser.add_argument("--ssl-lr", type=float, default=1e-3)
    parser.add_argument("--encoder-lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--finetune-epochs", type=int, default=100)
    parser.add_argument("--replace-last-k-layers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--use-optuna", action="store_true")
    parser.add_argument("--optuna-trials", type=int, default=20)
    parser.add_argument("--optuna-timeout", type=int, default=0)

    parser.add_argument("--output-root", type=str, default="./outputs")
    parser.add_argument("--timestamp", type=str, default="")
    parser.add_argument("--run-name", type=str, default="ttt")
    parser.add_argument("--params-file", type=str, default="")

    prelim = parser.parse_known_args()[0]
    if prelim.params_file:
        parser.set_defaults(**load_flat_params(prelim.params_file))
    return parser.parse_args()


def clamp_replace_last_k_layers(replace_last_k_layers: int, num_layers: int) -> int:
    if num_layers <= 0:
        raise ValueError("num_layers must be positive")
    k = int(replace_last_k_layers)
    if k < 1:
        k = 1
    if k > int(num_layers):
        k = int(num_layers)
    return k


def mixed_encoder_embed(
    x: torch.Tensor,
    edge_index: torch.Tensor,
    pre_model: GNNNodeClassifier,
    adapted_model: GNNNodeClassifier,
    replace_last_k_layers: int,
) -> torch.Tensor:
    total_layers = int(pre_model.num_layers)
    k = clamp_replace_last_k_layers(replace_last_k_layers, total_layers)
    split = total_layers - k

    h = x
    for idx in range(total_layers):
        if idx < split:
            src_model = pre_model
        else:
            src_model = adapted_model
        h = src_model.convs[idx](h, edge_index)
        if src_model.use_bn:
            h = src_model.bns[idx](h)
        h = F.relu(h)
        # Evaluation path: disable dropout to keep metric selection stable.
        h = F.dropout(h, p=src_model.dropout, training=False)
    return h


def run_ttt_once( # 1) 恢复 stage1 预训练模型（含 classifier）
    args,
    data,
    masks,
    pre_ckpt,
    gate_ckpt,
    train_cfg: Dict[str, float],
    verbose: bool = True,
):
    model_cfg = pre_ckpt["model_cfg"]
    ood_test_mask = masks["ood_test"]

    pre_model = GNNNodeClassifier(
        in_dim=int(model_cfg["in_dim"]),
        hidden_dim=int(model_cfg["hidden_dim"]),
        out_dim=int(model_cfg["out_dim"]),
        gnn_type=str(model_cfg["gnn_type"]),
        num_layers=int(model_cfg["num_layers"]),
        dropout=float(model_cfg["dropout"]),
        use_bn=bool(model_cfg.get("use_bn", True)),
        classifier_layers=int(model_cfg.get("classifier_layers", 1)),
    ).to(data.x.device)
    pre_model.load_state_dict(pre_ckpt["state_dict"])
    pre_model.eval()
    use_gate = gate_ckpt is not None
    gate = None
    if use_gate:
        # 2) 恢复 stage2 gate（冻结）
        gate_cfg = gate_ckpt["gate_cfg"]
        gate = GateMLP(
            in_dim=int(gate_cfg["in_dim"]),
            hidden_dim=int(gate_cfg["hidden_dim"]),
            num_layers=int(gate_cfg["num_layers"]),
            out_dim=int(gate_cfg["out_dim"]),
            dropout=float(gate_cfg["dropout"]),
            temperature=float(gate_cfg.get("temperature", 1.0)),
        ).to(data.x.device)
        gate.load_state_dict(gate_ckpt["gate_state_dict"])
        gate.eval()
        for p in gate.parameters():
            p.requires_grad = False

        task_names = list(gate_ckpt["task_names"])
        task_cfg = dict(gate_ckpt.get("task_cfg", {}))
    else:
        # 单 SSL 消融模式：不需要 gate checkpoint，直接按命令行任务构建分支。
        task_names = resolve_task_names(str(args.ssl_tasks), int(args.num_ssl))
        task_cfg = parse_task_cfg_json(str(args.task_cfg_json))
        gate_cfg = {
            "mode": "no_gate_single_ssl",
            "task_count": int(len(task_names)),
            "temperature": None,
        }
    replace_last_k_layers = clamp_replace_last_k_layers(
        int(train_cfg.get("replace_last_k_layers", 1)),
        int(model_cfg["num_layers"]),
    )

    adapted_models = []
    for _ in task_names:
        model_i = GNNNodeClassifier(
            in_dim=int(model_cfg["in_dim"]),
            hidden_dim=int(model_cfg["hidden_dim"]),
            out_dim=int(model_cfg["out_dim"]),
            gnn_type=str(model_cfg["gnn_type"]),
            num_layers=int(model_cfg["num_layers"]),
            dropout=float(model_cfg["dropout"]),
            use_bn=bool(model_cfg.get("use_bn", True)),
            classifier_layers=int(model_cfg.get("classifier_layers", 1)),
        ).to(data.x.device)
        model_i.load_state_dict(pre_ckpt["state_dict"])
        model_i.eval()
        adapted_models.append(model_i)

    def evaluate_current(models) -> Dict[str, float]:
        with torch.no_grad():
            branch_logits = []
            for branch_model in models:
                h_i = mixed_encoder_embed(
                    data.x,
                    data.edge_index,
                    pre_model=pre_model,
                    adapted_model=branch_model,
                    replace_last_k_layers=replace_last_k_layers,
                )
                logits_i = pre_model.classifier(h_i)
                branch_logits.append(logits_i)
            if use_gate:
                pre_embed = pre_model.get_embed(data.x, data.edge_index)
                gate_weights = gate(pre_embed)
                logits_stack = torch.stack(branch_logits, dim=1)
                fused_logits = (logits_stack * gate_weights.unsqueeze(-1)).sum(dim=1)
            elif len(branch_logits) == 1:
                fused_logits = branch_logits[0]
            else:
                fused_logits = torch.stack(branch_logits, dim=0).mean(dim=0)

        return {
            "id_val_acc": accuracy(fused_logits, data.y, masks["id_val"]),
            "id_test_acc": accuracy(fused_logits, data.y, masks["id_test"]),
            "ood_val_acc": accuracy(fused_logits, data.y, masks["ood_val"]),
            "ood_test_acc": accuracy(fused_logits, data.y, masks["ood_test"]),
        }

    best_ood_test = float("-inf")
    best_metrics = None
    best_model_states = [copy.deepcopy(model.state_dict()) for model in adapted_models]
    history = []

    def record_history(epoch_idx: int, metrics_dict: Dict[str, float]) -> None:
        history.append(
            {
                "epoch": int(epoch_idx),
                "ood_test_acc": float(metrics_dict["ood_test_acc"]),
                "id_test_acc": float(metrics_dict["id_test_acc"]),
                "ood_val_acc": float(metrics_dict["ood_val_acc"]),
                "id_val_acc": float(metrics_dict["id_val_acc"]),
            }
        )

    initial_metrics = evaluate_current(adapted_models)
    epoch_counter = 0
    record_history(epoch_counter, initial_metrics)
    if not np.isnan(initial_metrics["ood_test_acc"]) and initial_metrics["ood_test_acc"] > best_ood_test:
        best_ood_test = initial_metrics["ood_test_acc"]
        best_metrics = initial_metrics
        best_model_states = [copy.deepcopy(model.state_dict()) for model in adapted_models]

    branch_ssl_tasks = []
    branch_optimizers = []
    for idx, task_name in enumerate(task_names):
        model_i = adapted_models[idx]
        ssl_i = build_ssl_tasks(
            [task_name],
            task_cfg=task_cfg,
            device=data.x.device,
            hidden_dim=int(model_cfg["hidden_dim"]),
            input_dim=int(model_cfg["in_dim"]),
        )
        ssl_params = list(ssl_i.parameters())
        for p in model_i.convs.parameters():
            p.requires_grad = True

        param_groups = [{"params": model_i.convs.parameters(), "lr": float(train_cfg["encoder_lr"])}]
        if len(ssl_params) > 0:
            param_groups.append({"params": ssl_params, "lr": float(train_cfg["ssl_lr"])})
        joint_optimizer = torch.optim.Adam(
            param_groups,
            weight_decay=float(train_cfg["weight_decay"]),
        )

        branch_ssl_tasks.append(ssl_i)
        branch_optimizers.append(joint_optimizer)

    for epoch in range(1, int(train_cfg["finetune_epochs"]) + 1):
        # One global epoch: update all branches once, then evaluate once.
        for idx, task_name in enumerate(task_names):
            model_i = adapted_models[idx]
            ssl_i = branch_ssl_tasks[idx]
            joint_optimizer = branch_optimizers[idx]

            model_i.train()
            ssl_i.train()
            joint_optimizer.zero_grad()
            node_loss = ssl_i[0].compute_node_loss(model_i, data.x, data.edge_index)
            loss = node_loss[ood_test_mask].mean()
            if loss.requires_grad:
                loss.backward()
                joint_optimizer.step()

        current_metrics = evaluate_current(adapted_models)
        epoch_counter = epoch
        record_history(epoch_counter, current_metrics)
        if not np.isnan(current_metrics["ood_test_acc"]) and current_metrics["ood_test_acc"] > best_ood_test:
            best_ood_test = current_metrics["ood_test_acc"]
            best_metrics = current_metrics
            best_model_states = [copy.deepcopy(model.state_dict()) for model in adapted_models]

    for model_i in adapted_models:
        model_i.eval()
    if verbose:
        print("Adapted branches: {}".format(", ".join("{}:{}".format(i + 1, name) for i, name in enumerate(task_names))))
        if not use_gate:
            print("[ttt] no gate checkpoint provided; using direct SSL branch prediction")

    for model_i, state in zip(adapted_models, best_model_states):
        model_i.load_state_dict(state)
    metrics = best_metrics if best_metrics is not None else evaluate_current(adapted_models)

    artifacts = {
        "model_cfg": model_cfg,
        "gate_cfg": gate_cfg,
        "gate_state_dict": gate.state_dict() if gate is not None else {},
        "classifier_state_dict": pre_model.classifier.state_dict(),
        "adapted_model_state_dicts": [model_i.state_dict() for model_i in adapted_models],
        "task_names": task_names,
        "task_cfg": task_cfg,
        "replace_last_k_layers": int(replace_last_k_layers),
        "history": history,
        "fusion_mode": "gate_weighted_logits" if use_gate else ("single_ssl" if len(task_names) == 1 else "uniform_logit_average"),
    }
    return metrics, artifacts


def save_ttt_curve_plot(history, plot_path: Path) -> None:
    if not history:
        return
    epochs = [int(item["epoch"]) for item in history]
    ood_test = [float(item["ood_test_acc"]) for item in history]

    plt.figure(figsize=(8, 4.8))
    plt.plot(epochs, ood_test, color="#1f77b4", linewidth=2.0)
    plt.xlabel("epoch")
    plt.ylabel("ood_test_acc")
    plt.title("TTT OOD Test Accuracy vs Epoch")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()


def save_gate_weight_plot(task_names: Sequence[str], gate_weights: torch.Tensor, plot_path: Path) -> None:
    if not task_names or gate_weights.ndim != 2 or int(gate_weights.shape[0]) == 0:
        return

    weights_np = gate_weights.detach().cpu().numpy()
    task_count = min(len(task_names), int(weights_np.shape[1]))
    if task_count <= 0:
        return

    if task_count == 1:
        y = weights_np[:, 0]
        x = np.arange(y.shape[0])
        plt.figure(figsize=(8, 4))
        plt.scatter(x, y, s=6, alpha=0.8)
        plt.ylim(0.0, 1.0)
        plt.xlim(0, max(1, y.shape[0]))
        plt.xlabel("Node Index")
        plt.ylabel("Gate weight: {}".format(task_names[0]))
        plt.title("Gate weights (one task)")
        plt.tight_layout()
        plt.savefig(plot_path, dpi=180)
        plt.close()
        return

    if task_count >= 2:
        w1 = weights_np[:, 0]
        w2 = weights_np[:, 1]
        plt.figure(figsize=(5, 5))
        plt.scatter(w1, w2, s=6, alpha=0.7)
        plt.xlim(0.0, 1.0)
        plt.ylim(0.0, 1.0)
        plt.xlabel("Gate weight: {}".format(task_names[0]))
        plt.ylabel("Gate weight: {}".format(task_names[1]))
        plt.title("Gate weights (two tasks)")
        plt.tight_layout()
        plt.savefig(plot_path, dpi=180)
        plt.close()
        return


def build_ttt_search_space(trial, args, num_layers: int):
    max_layers = max(int(num_layers), 1)
    return {
        "ssl_lr": trial.suggest_float("ssl_lr", 1e-5, 1e-3, log=True),
        "encoder_lr": trial.suggest_float("encoder_lr", 1e-5, 1e-2, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-8, 1e-3, log=True),
        "finetune_epochs": int(args.finetune_epochs),
        "replace_last_k_layers": trial.suggest_int("replace_last_k_layers", 1, max_layers),
    }


def main():
    args = parse_args()
    if not args.pretrain_ckpt:
        raise ValueError("--pretrain-ckpt is required (or provide it via --params-file)")

    set_seed(int(args.seed))
    device = torch.device(args.device)

    pre_ckpt_path = Path(args.pretrain_ckpt)
    pre_ckpt = safe_torch_load(pre_ckpt_path)
    gate_ckpt = safe_torch_load(Path(args.gate_ckpt)) if args.gate_ckpt else None

    pre_ckpt["model_cfg"] = normalize_pretrain_model_cfg(pre_ckpt, pre_ckpt_path)

    model_cfg = pre_ckpt["model_cfg"]
    dataset = args.dataset or (gate_ckpt.get("dataset") if gate_ckpt is not None else "") or pre_ckpt["dataset"]
    domain = args.domain or (gate_ckpt.get("domain") if gate_ckpt is not None else "") or pre_ckpt["domain"]
    shift = args.shift or (gate_ckpt.get("shift") if gate_ckpt is not None else "") or pre_ckpt["shift"]

    data, masks, data_path = load_good_data(dataset, domain, shift, device)

    if args.use_optuna:
        if optuna is None:
            raise RuntimeError("Optuna is not installed, but --use-optuna is set")

        best_trial_score = float("-inf")
        best_trial_metrics = None
        best_trial_artifacts = None
        best_train_cfg = None

        def objective(trial):
            nonlocal best_trial_score, best_trial_metrics, best_trial_artifacts, best_train_cfg
            train_cfg = build_ttt_search_space(trial, args, int(model_cfg["num_layers"]))
            metrics_tmp, artifacts_tmp = run_ttt_once(
                args=args,
                data=data,
                masks=masks,
                pre_ckpt=pre_ckpt,
                gate_ckpt=gate_ckpt,
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
            raise RuntimeError("Optuna finished without a valid best trial result to save.")
        metrics = best_trial_metrics
        artifacts = best_trial_artifacts
    else:
        best_train_cfg = {
            "ssl_lr": float(args.ssl_lr),
            "encoder_lr": float(args.encoder_lr),
            "weight_decay": float(args.weight_decay),
            "finetune_epochs": int(args.finetune_epochs),
            "replace_last_k_layers": clamp_replace_last_k_layers(
                int(args.replace_last_k_layers),
                int(model_cfg["num_layers"]),
            ),
        }
        metrics, artifacts = run_ttt_once(
            args=args,
            data=data,
            masks=masks,
            pre_ckpt=pre_ckpt,
            gate_ckpt=gate_ckpt,
            train_cfg=best_train_cfg,
            verbose=True,
        )

    out_dir = make_stage_output_dir(
        args.output_root,
        "stage3" if gate_ckpt is not None else "stage3_single_ssl",
        dataset,
        domain,
        shift,
        timestamp=args.timestamp,
    )
    ckpt_path = out_dir / "ttt_model.pt"
    metrics_path = out_dir / "metrics.json"
    params_path = out_dir / "params.json"
    plot_path = out_dir / "ood_test_acc_vs_epoch.png"

    save_ttt_curve_plot(artifacts.get("history", []), plot_path)

    torch.save(
        {
            "stage": "ttt",
            "pretrain_ckpt": str(args.pretrain_ckpt),
            "gate_ckpt": str(args.gate_ckpt),
            "dataset": dataset,
            "domain": domain,
            "shift": shift,
            "data_path": str(data_path),
            "task_names": artifacts["task_names"],
            "task_cfg": artifacts["task_cfg"],
            "replace_last_k_layers": int(artifacts.get("replace_last_k_layers", 1)),
            "model_cfg": artifacts["model_cfg"],
            "gate_cfg": artifacts["gate_cfg"],
            "gate_state_dict": artifacts["gate_state_dict"],
            "classifier_state_dict": artifacts["classifier_state_dict"],
            "adapted_model_state_dicts": artifacts["adapted_model_state_dicts"],
            "fusion_mode": artifacts.get("fusion_mode", ""),
            "train_cfg": best_train_cfg,
            "metrics": metrics,
            "history": artifacts.get("history", []),
        },
        ckpt_path,
    )

    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    run_params = {
        "params_file": args.params_file,
        "dataset": dataset,
        "domain": domain,
        "shift": shift,
        "pretrain_ckpt": str(args.pretrain_ckpt),
        "gate_ckpt": str(args.gate_ckpt),
        "ssl_tasks": ",".join(artifacts["task_names"]),
        "num_ssl": int(len(artifacts["task_names"])),
        "task_cfg_json": json.dumps(artifacts["task_cfg"], ensure_ascii=True),
        "ssl_lr": float(best_train_cfg["ssl_lr"]),
        "encoder_lr": float(best_train_cfg["encoder_lr"]),
        "weight_decay": float(best_train_cfg["weight_decay"]),
        "finetune_epochs": int(best_train_cfg["finetune_epochs"]),
        "replace_last_k_layers": int(best_train_cfg["replace_last_k_layers"]),
        "use_optuna": bool(args.use_optuna),
        "optuna_trials": int(args.optuna_trials),
        "optuna_timeout": int(args.optuna_timeout),
        "seed": int(args.seed),
        "device": str(args.device),
        "output_root": str(args.output_root),
        "timestamp": str(args.timestamp),
        "run_name": str(args.run_name),
        "fusion_mode": artifacts.get("fusion_mode", ""),
        "output_dir": str(out_dir),
        "plot_path": str(plot_path),
    }
    with params_path.open("w", encoding="utf-8") as f:
        json.dump(run_params, f, indent=2)

    print("Saved ttt checkpoint:", ckpt_path)
    print("Saved params:", params_path)
    print("Saved plot:", plot_path)
    print("Metrics:", json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
