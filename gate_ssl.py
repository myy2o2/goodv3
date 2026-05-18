from __future__ import annotations

import argparse
import json
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, SAGEConv

from ssl_tasks import available_ssl_tasks, build_ssl_tasks, compute_task_loss_matrix, parse_ssl_task_names
# available_ssl_tasks: 全部任务名
# build_ssl_tasks: 构建任务模块列表
# compute_task_loss_matrix: 返回 [N, T] 的每节点每任务损失
# parse_ssl_task_names: 解析字符串任务名

DATA_ROOT = Path("./datasets")

# resolve_dataset_path / safe_torch_load / extract_masks / load_good_data
# 与 pretrain 同逻辑：定位数据、兼容 torch.load、提取 split 掩码

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
    masks["ood_val"] = _get_mask("ood_val_mask", "val_mask")
    masks["ood_test"] = _get_mask("ood_test_mask", "test_mask")
    return masks


def load_good_data(dataset: str, domain: str, shift: str, device: torch.device):
    path = resolve_dataset_path(dataset, domain, shift)
    obj = safe_torch_load(path)
    data = obj[0] if isinstance(obj, tuple) else obj
    data = data.to(device)
    masks = extract_masks(data)
    return data, masks, path


class GNNNodeClassifier(nn.Module): # 作用：在 stage2 中恢复 stage1 模型参数，提取固定 embedding
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
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(self.dropout))
        layers.append(nn.Linear(hidden_dim, out_dim))
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


class FrozenEmbeddingModel(nn.Module):
    def __init__(self, embedding: torch.Tensor, classifier: nn.Module):
        super().__init__()
        self.register_buffer("embedding", embedding)
        self.classifier = classifier

    def get_embed(self, feat: torch.Tensor, edge_index: torch.Tensor, edge_weight=None):
        del feat, edge_index, edge_weight
        return self.embedding # 始终返回固定 embedding

    def forward(self, feat: torch.Tensor, edge_index: torch.Tensor, edge_weight=None):
        del feat, edge_index, edge_weight
        return self.classifier(self.embedding)


class GateMLP(nn.Module): # 作用：门控网络，用于对每个任务的损失进行加权
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        num_layers: int,
        out_dim: int,
        dropout: float,
        temperature: float,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError("Gate num_layers must be >= 1")
        self.temperature = max(float(temperature), 1e-6)
        layers: List[nn.Module] = []
        prev = in_dim
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(prev, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))
            prev = hidden_dim
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        logits = self.net(h) / self.temperature
        return F.softmax(logits, dim=-1) # 每节点对 T 个 SSL 任务的权重


def normalize_per_task(loss_mat: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    col_sum = loss_mat.sum(dim=0, keepdim=True) # 每个任务在所有节点上的总损失
    return loss_mat / col_sum.clamp_min(eps) # 节点归一化（每列和约为 1）


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
    ssl_tag: str,
    timestamp: str = "",
) -> Path:
    ts = timestamp if timestamp else datetime.now().strftime("%d%H%M%S")
    run_dir_name = "{}_{}".format(ssl_tag, ts)
    out_dir = Path(output_root) / stage_name / dataset.lower() / domain.lower() / shift.lower() / run_dir_name
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def make_ssl_tag(task_names: List[str]) -> str:
    raw = "-".join(task_names).lower()
    safe = "".join(ch if (ch.isalnum() or ch in "-_") else "-" for ch in raw)
    while "--" in safe:
        safe = safe.replace("--", "-")
    return safe.strip("-") or "ssl"


def load_flat_params(params_file: str) -> Dict[str, object]:
    if not params_file:
        return {}
    with Path(params_file).open("r", encoding="utf-8") as f:
        params = json.load(f)
    if not isinstance(params, dict):
        raise ValueError("params file must be a flat JSON object")
    return params


def normalize_pretrain_model_cfg(pretrain_ckpt: Dict[str, object], pretrain_ckpt_path: Path) -> Dict[str, object]:
    model_cfg = dict(pretrain_ckpt.get("model_cfg", {}))
    params_path = pretrain_ckpt_path.parent / "params.json"
    extra = {}
    if params_path.exists():
        try:
            extra = load_flat_params(str(params_path))
        except Exception:
            extra = {}

    # Backward-compatibility for older stage1 checkpoints missing these fields.
    model_cfg["gnn_type"] = str(model_cfg.get("gnn_type", extra.get("gnn_type", "gcn")))
    model_cfg["classifier_layers"] = int(model_cfg.get("classifier_layers", extra.get("classifier_layers", 1)))
    model_cfg["use_bn"] = bool(model_cfg.get("use_bn", extra.get("use_bn", True)))
    return model_cfg


def parse_args():
    parser = argparse.ArgumentParser(description="Stage-2 train SSL heads + gate")
    parser.add_argument("--pretrain-ckpt", type=str, default="")
    parser.add_argument("--dataset", type=str, default="")
    parser.add_argument("--domain", type=str, default="")
    parser.add_argument("--shift", type=str, default="")

    parser.add_argument("--ssl-tasks", type=str, default="consistency")
    parser.add_argument("--num-ssl", type=int, default=1)
    parser.add_argument("--task-cfg-json", type=str, default="")

    parser.add_argument("--gate-hidden-dim", type=int, default=128)
    parser.add_argument("--gate-num-layers", type=int, default=2)
    parser.add_argument("--gate-dropout", type=float, default=0.1)
    parser.add_argument("--gate-temperature", type=float, default=1.0)

    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    parser.add_argument("--output-root", type=str, default="./outputs")
    parser.add_argument("--timestamp", type=str, default="")
    parser.add_argument("--run-name", type=str, default="gate_ssl")
    parser.add_argument("--params-file", type=str, default="")

    prelim = parser.parse_known_args()[0]
    if prelim.params_file:
        parser.set_defaults(**load_flat_params(prelim.params_file))
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.pretrain_ckpt:
        raise ValueError("--pretrain-ckpt is required (or provide it via --params-file)")

    set_seed(int(args.seed))
    device = torch.device(args.device)

    pretrain_ckpt_path = Path(args.pretrain_ckpt)
    pretrain_ckpt = safe_torch_load(pretrain_ckpt_path)
    model_cfg = normalize_pretrain_model_cfg(pretrain_ckpt, pretrain_ckpt_path)

    dataset = args.dataset or pretrain_ckpt["dataset"]
    domain = args.domain or pretrain_ckpt["domain"]
    shift = args.shift or pretrain_ckpt["shift"]

    data, masks, data_path = load_good_data(dataset, domain, shift, device)

    base_model = GNNNodeClassifier(
        in_dim=int(model_cfg["in_dim"]),
        hidden_dim=int(model_cfg["hidden_dim"]),
        out_dim=int(model_cfg["out_dim"]),
        gnn_type=str(model_cfg["gnn_type"]),
        num_layers=int(model_cfg["num_layers"]),
        dropout=float(model_cfg["dropout"]),
        use_bn=bool(model_cfg.get("use_bn", True)),
        classifier_layers=int(model_cfg.get("classifier_layers", 1)),
    ).to(device)
    base_model.load_state_dict(pretrain_ckpt["state_dict"])
    base_model.eval()
    for p in base_model.parameters():
        p.requires_grad = False

    with torch.no_grad():
        base_embed = base_model.get_embed(data.x, data.edge_index)  #  提取固定 embedding，给 gate/ssl 训练使用

    frozen_model = FrozenEmbeddingModel(base_embed, base_model.classifier).to(device)


     # 3) 解析 SSL 任务
    if args.ssl_tasks.lower() == "all":
        task_names = available_ssl_tasks()
    else:
        task_names = parse_ssl_task_names(args.ssl_tasks)
    if int(args.num_ssl) > 0:
        if int(args.num_ssl) > len(task_names):
            raise ValueError("num_ssl={} is larger than parsed task count={}".format(args.num_ssl, len(task_names)))
        task_names = task_names[: int(args.num_ssl)]
    if not task_names:
        raise ValueError("No SSL tasks selected")

    task_cfg = json.loads(args.task_cfg_json) if args.task_cfg_json else {}
    ssl_tasks = build_ssl_tasks(
        task_names,
        task_cfg=task_cfg,
        device=device,
        hidden_dim=int(model_cfg["hidden_dim"]),
        input_dim=int(model_cfg["in_dim"]),
    )
     # 4) 构建 gate
    gate = GateMLP(
        in_dim=int(model_cfg["hidden_dim"]),
        hidden_dim=int(args.gate_hidden_dim),
        num_layers=int(args.gate_num_layers),
        out_dim=len(task_names),
        dropout=float(args.gate_dropout),
        temperature=float(args.gate_temperature),
    ).to(device)

    ssl_param_list = list(ssl_tasks.parameters())
    optimizer = torch.optim.Adam(
        list(gate.parameters()) + ssl_param_list,
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )

    best_val = float("inf")
    best_state = None
    wait = 0
    train_mask = masks["train"]
    joint_skipped_no_grad = 0

    for _ in range(1, int(args.epochs) + 1):
        gate.train()
        ssl_tasks.train()
        optimizer.zero_grad()

        loss_mat = compute_task_loss_matrix(
            ssl_tasks,
            frozen_model,
            feat=data.x,
            edge_index=data.edge_index,
        )
        norm_loss = normalize_per_task(loss_mat)
        weights = gate(base_embed)

        node_loss = (weights * norm_loss).sum(dim=-1)
        loss = node_loss[train_mask].mean()
        if loss.requires_grad:
            loss.backward()
            optimizer.step()
        else:
            joint_skipped_no_grad += 1

        with torch.no_grad():
            gate.eval()
            ssl_tasks.eval()
            val_mask = masks["id_val"] if masks["id_val"] is not None else train_mask
            val_loss = node_loss[val_mask].mean().item()

        if val_loss < best_val:
            best_val = val_loss
            best_state = {
                "gate": gate.state_dict(),
                "ssl": [task.state_dict() for task in ssl_tasks],
            }
            wait = 0
        else:
            wait += 1
            if wait >= int(args.patience):
                break

    if best_state is not None:
        gate.load_state_dict(best_state["gate"])
        for task, state in zip(ssl_tasks, best_state["ssl"]):
            task.load_state_dict(state)

    if joint_skipped_no_grad > 0:
        print("[stage2] Skip backward when loss has no grad graph: joint_skipped={}".format(joint_skipped_no_grad))
    
     #  最终结果与绘图
    gate.eval()
    ssl_tasks.eval()
    with torch.no_grad():
        final_loss_mat = compute_task_loss_matrix(
            ssl_tasks,
            frozen_model,
            feat=data.x,
            edge_index=data.edge_index,
        )
        final_norm_loss = normalize_per_task(final_loss_mat)
        final_weights = gate(base_embed)

    out_dir = make_stage_output_dir(
        args.output_root,
        "stage2",
        dataset,
        domain,
        shift,
        make_ssl_tag(task_names),
        timestamp=args.timestamp,
    )
    ckpt_path = out_dir / "gate_model.pt"
    plot_path = out_dir / "plot.png"
    metrics_path = out_dir / "metrics.json"
    params_path = out_dir / "params.json"

    torch.save(
        {
            "stage": "gate_ssl",
            "pretrain_ckpt": str(args.pretrain_ckpt),
            "dataset": dataset,
            "domain": domain,
            "shift": shift,
            "data_path": str(data_path),
            "model_cfg": model_cfg,
            "task_names": task_names,
            "task_cfg": task_cfg,
            "gate_cfg": {
                "in_dim": int(model_cfg["hidden_dim"]),
                "hidden_dim": int(args.gate_hidden_dim),
                "num_layers": int(args.gate_num_layers),
                "dropout": float(args.gate_dropout),
                "temperature": float(args.gate_temperature),
                "out_dim": len(task_names),
            },
            "gate_state_dict": gate.state_dict(),
        },
        ckpt_path,
    )

    if len(task_names) == 1:
        y = final_norm_loss[:, 0].detach().cpu().numpy()
        x = np.arange(y.shape[0])
        plt.figure(figsize=(8, 4))
        plt.scatter(x, y, s=6, alpha=0.8)
        plt.ylim(0.0, 1.0)
        plt.xlim(0, max(1, y.shape[0]))
        plt.xlabel("Node Index")
        plt.ylabel("Normalized SSL Loss")
        plt.title("{} node-wise normalized loss".format(task_names[0]))
        plt.tight_layout()
        plt.savefig(plot_path, dpi=180)
        plt.close()
    else:
        w1 = final_weights[:, 0].detach().cpu().numpy()
        w2 = final_weights[:, 1].detach().cpu().numpy()
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

    summary = {
        "task_names": task_names,
        "mean_gate_weight": final_weights.mean(dim=0).detach().cpu().tolist(),
        "plot_path": str(plot_path),
        "ckpt_path": str(ckpt_path),
    }
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    run_params = {
        "params_file": args.params_file,
        "dataset": dataset,
        "domain": domain,
        "shift": shift,
        "pretrain_ckpt": str(args.pretrain_ckpt),
        "ssl_tasks": ",".join(task_names),
        "num_ssl": int(len(task_names)),
        "task_cfg_json": json.dumps(task_cfg, ensure_ascii=True),
        "gate_hidden_dim": int(args.gate_hidden_dim),
        "gate_num_layers": int(args.gate_num_layers),
        "gate_dropout": float(args.gate_dropout),
        "gate_temperature": float(args.gate_temperature),
        "epochs": int(args.epochs),
        "patience": int(args.patience),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "seed": int(args.seed),
        "device": str(args.device),
        "output_root": str(args.output_root),
        "timestamp": str(args.timestamp),
        "run_name": str(args.run_name),
        "output_dir": str(out_dir),
    }
    with params_path.open("w", encoding="utf-8") as f:
        json.dump(run_params, f, indent=2)

    print("Saved checkpoint:", ckpt_path)
    print("Saved params:", params_path)
    print("Saved plot:", plot_path)
    print("Summary:", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
