from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Iterable, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from ttt import (
    GNNNodeClassifier,
    load_good_data,
    mixed_encoder_embed,
    normalize_pretrain_model_cfg,
    safe_torch_load,
    set_seed,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot t-SNE for stage1 pretrain embeddings and post-TTT adapted embeddings."
    )
    parser.add_argument("--dataset", required=True, help="Dataset name, e.g. cora, citeseer, pubmed, wikics, arxiv.")
    parser.add_argument("--domain", required=True, help="GOOD domain, e.g. word, degree, time.")
    parser.add_argument("--shift", required=True, help="GOOD shift, e.g. covariate, concept.")
    parser.add_argument("--stage1-root", default="./outputs/stage1")
    parser.add_argument(
        "--stage3-root",
        default="./outputs/stage3_new",
        help="Root containing stage3_new runs. The script also tries ./outputs/stage3_new as fallback.",
    )
    parser.add_argument("--pretrain-ckpt", default="", help="Optional explicit stage1 checkpoint path.")
    parser.add_argument("--ttt-ckpt", default="", help="Optional explicit TTT checkpoint path.")
    parser.add_argument(
        "--run-regex",
        default="",
        help=(
            "Regex matched against run directory names under stage3-root/dataset/domain/shift. "
            "Example: 'bootstrap_pseudolabel_propagation'."
        ),
    )
    parser.add_argument(
        "--select",
        choices=["latest", "best", "first"],
        default="latest",
        help="Which matching TTT run to use when multiple runs match.",
    )
    parser.add_argument("--list-runs", action="store_true", help="List matching TTT runs and exit.")
    parser.add_argument("--output-dir", default="", help="Output directory for plots and arrays.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-nodes", type=int, default=5000, help="Subsample nodes for t-SNE if graph is larger.")
    parser.add_argument(
        "--sample-mask",
        choices=["all", "train", "id_val", "id_test", "ood_val", "ood_test", "id", "ood", "test"],
        default="all",
        help="Node pool used before optional max-nodes sampling.",
    )
    parser.add_argument("--perplexity", type=float, default=30.0)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--point-size", type=float, default=7.0)
    parser.add_argument("--alpha", type=float, default=0.78)
    parser.add_argument("--no-standardize", action="store_true")
    parser.add_argument(
        "--separate-tsne",
        action="store_true",
        help="Run t-SNE separately for stage1 and TTT. Default fits one shared t-SNE on both embeddings.",
    )
    return parser.parse_args()


def build_model(model_cfg: Dict[str, object], device: torch.device) -> GNNNodeClassifier:
    return GNNNodeClassifier(
        in_dim=int(model_cfg["in_dim"]),
        hidden_dim=int(model_cfg["hidden_dim"]),
        out_dim=int(model_cfg["out_dim"]),
        gnn_type=str(model_cfg["gnn_type"]),
        num_layers=int(model_cfg["num_layers"]),
        dropout=float(model_cfg["dropout"]),
        use_bn=bool(model_cfg.get("use_bn", True)),
        classifier_layers=int(model_cfg.get("classifier_layers", 1)),
    ).to(device)


def default_pretrain_ckpt(stage1_root: str, dataset: str, domain: str, shift: str) -> Path:
    return Path(stage1_root) / dataset.lower() / domain.lower() / shift.lower() / "right" / "pretrain_model.pt"


def candidate_stage3_roots(primary_root: str) -> list[Path]:
    roots = []
    for root in [Path(primary_root), Path("./output/stage3_new"), Path("./outputs/stage3_new")]:
        if root.exists() and root.is_dir() and root not in roots:
            roots.append(root)
    return roots


def run_sort_key(path: Path):
    try:
        mtime = path.stat().st_mtime
    except OSError:
        mtime = 0.0
    return (mtime, path.name)


def load_metrics(run_dir: Path) -> Dict[str, object]:
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        return {}
    with metrics_path.open("r", encoding="utf-8") as f:
        metrics = json.load(f)
    return metrics if isinstance(metrics, dict) else {}


def find_ttt_runs(
    stage3_root: str,
    dataset: str,
    domain: str,
    shift: str,
    run_regex: str,
) -> list[Path]:
    pattern = re.compile(run_regex) if run_regex else None
    for root in candidate_stage3_roots(stage3_root):
        runs = []
        base = root / dataset.lower() / domain.lower() / shift.lower()
        if not base.exists():
            continue
        for ckpt_path in sorted(base.glob("*/ttt_shared_onestage_model.pt")):
            run_dir = ckpt_path.parent
            if pattern is None or pattern.search(run_dir.name):
                runs.append(run_dir)
        if runs:
            return runs
    return []


def select_ttt_run(runs: list[Path], select: str) -> Path:
    if not runs:
        raise FileNotFoundError("No matching TTT runs found.")
    if select == "first":
        return sorted(runs)[0]
    if select == "latest":
        return max(runs, key=run_sort_key)
    if select == "best":
        def score(run_dir: Path) -> float:
            value = load_metrics(run_dir).get("ood_test_acc", float("-inf"))
            try:
                return float(value)
            except (TypeError, ValueError):
                return float("-inf")

        return max(runs, key=score)
    raise ValueError("Unsupported select mode: {}".format(select))


def mask_to_indices(masks: Dict[str, Optional[torch.Tensor]], mode: str, num_nodes: int) -> torch.Tensor:
    if mode == "all":
        return torch.arange(num_nodes)
    if mode in masks:
        mask = masks[mode]
        if mask is None:
            raise ValueError("Mask '{}' is not available.".format(mode))
        return mask.detach().cpu().nonzero(as_tuple=False).view(-1)
    if mode == "id":
        parts = [masks.get("id_val"), masks.get("id_test")]
    elif mode == "ood":
        parts = [masks.get("ood_val"), masks.get("ood_test")]
    elif mode == "test":
        parts = [masks.get("id_test"), masks.get("ood_test")]
    else:
        raise ValueError("Unsupported sample mask: {}".format(mode))

    out = []
    for mask in parts:
        if mask is not None:
            out.append(mask.detach().cpu().nonzero(as_tuple=False).view(-1))
    if not out:
        raise ValueError("No masks are available for mode '{}'.".format(mode))
    return torch.unique(torch.cat(out, dim=0), sorted=True)


def subsample_indices(indices: torch.Tensor, max_nodes: int, seed: int) -> torch.Tensor:
    if max_nodes <= 0 or int(indices.numel()) <= max_nodes:
        return indices
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    perm = torch.randperm(int(indices.numel()), generator=generator)[:max_nodes]
    return indices[perm].sort().values


def build_split_labels(masks: Dict[str, Optional[torch.Tensor]], indices: torch.Tensor) -> list[str]:
    labels = []
    split_order = ["train", "id_val", "id_test", "ood_val", "ood_test"]
    for idx in indices.tolist():
        split_name = "other"
        for name in split_order:
            mask = masks.get(name)
            if mask is not None and bool(mask.detach().cpu()[idx]):
                split_name = name
                break
        labels.append(split_name)
    return labels


@torch.no_grad()
def compute_embeddings(
    data,
    pre_ckpt: Dict[str, object],
    pre_ckpt_path: Path,
    ttt_ckpt: Dict[str, object],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    model_cfg = normalize_pretrain_model_cfg(pre_ckpt, pre_ckpt_path)
    pre_model = build_model(model_cfg, device)
    pre_model.load_state_dict(pre_ckpt["state_dict"])
    pre_model.eval()

    adapted_model = build_model(ttt_ckpt["model_cfg"], device)
    adapted_model.load_state_dict(ttt_ckpt["adapted_model_state_dict"])
    adapted_model.eval()

    pre_embed = pre_model.get_embed(data.x, data.edge_index)
    ttt_embed = mixed_encoder_embed(
        data.x,
        data.edge_index,
        pre_model=pre_model,
        adapted_model=adapted_model,
        replace_last_k_layers=int(ttt_ckpt.get("replace_last_k_layers", 1)),
    )
    return pre_embed.detach().cpu(), ttt_embed.detach().cpu()


def fit_tsne(features: np.ndarray, args) -> np.ndarray:
    if features.shape[0] <= 2:
        raise ValueError("Need at least 3 points for t-SNE.")
    perplexity = min(float(args.perplexity), max(1.0, (features.shape[0] - 1) / 3.0))
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        max_iter=int(args.max_iter),
        init="pca",
        learning_rate="auto",
        random_state=int(args.seed),
    )
    return tsne.fit_transform(features)


def color_values(labels: Iterable[object]):
    labels = list(labels)
    unique = sorted(set(labels), key=lambda x: str(x))
    mapping = {label: i for i, label in enumerate(unique)}
    return np.array([mapping[label] for label in labels]), unique


def scatter_by_labels(ax, coords: np.ndarray, labels: Iterable[object], title: str, args):
    values, unique = color_values(labels)
    cmap = plt.get_cmap("tab20", max(len(unique), 1))
    sc = ax.scatter(
        coords[:, 0],
        coords[:, 1],
        c=values,
        cmap=cmap,
        s=float(args.point_size),
        alpha=float(args.alpha),
        linewidths=0,
    )
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    handles = []
    for i, label in enumerate(unique):
        handles.append(
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label=str(label),
                markerfacecolor=cmap(i),
                markersize=5,
            )
        )
    if handles:
        ax.legend(handles=handles, loc="best", fontsize=8, frameon=False, markerscale=1.0)
    return sc


def save_plots(
    out_dir: Path,
    pre_coords: np.ndarray,
    ttt_coords: np.ndarray,
    y_labels: list[int],
    split_labels: list[str],
    run_name: str,
    args,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), constrained_layout=True)
    scatter_by_labels(axes[0], pre_coords, y_labels, "Stage1 Pretrain Embedding", args)
    scatter_by_labels(axes[1], ttt_coords, y_labels, "TTT Adapted Embedding", args)
    fig.suptitle("{} / {} / {} | {}".format(args.dataset, args.domain, args.shift, run_name))
    fig.savefig(out_dir / "tsne_by_label.png", dpi=220)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), constrained_layout=True)
    scatter_by_labels(axes[0], pre_coords, split_labels, "Stage1 Pretrain Embedding", args)
    scatter_by_labels(axes[1], ttt_coords, split_labels, "TTT Adapted Embedding", args)
    fig.suptitle("{} / {} / {} | {}".format(args.dataset, args.domain, args.shift, run_name))
    fig.savefig(out_dir / "tsne_by_split.png", dpi=220)
    plt.close(fig)


def main():
    args = parse_args()
    set_seed(int(args.seed))
    device = torch.device(args.device)

    pretrain_path = Path(args.pretrain_ckpt) if args.pretrain_ckpt else default_pretrain_ckpt(
        args.stage1_root, args.dataset, args.domain, args.shift
    )
    if not pretrain_path.exists():
        raise FileNotFoundError("Missing pretrain checkpoint: {}".format(pretrain_path))

    if args.ttt_ckpt:
        ttt_path = Path(args.ttt_ckpt)
        if not ttt_path.exists():
            raise FileNotFoundError("Missing TTT checkpoint: {}".format(ttt_path))
        selected_run = ttt_path.parent
        runs = [selected_run]
    else:
        runs = find_ttt_runs(args.stage3_root, args.dataset, args.domain, args.shift, args.run_regex)
        if args.list_runs:
            for run in sorted(runs):
                metrics = load_metrics(run)
                metric_text = ""
                if "ood_test_acc" in metrics:
                    metric_text = " ood_test_acc={:.6f}".format(float(metrics["ood_test_acc"]))
                print("{}{}".format(run, metric_text))
            print("total={}".format(len(runs)))
            return
        selected_run = select_ttt_run(runs, args.select)
        ttt_path = selected_run / "ttt_shared_onestage_model.pt"

    print("Pretrain checkpoint:", pretrain_path)
    print("TTT checkpoint:", ttt_path)
    print("Selected run:", selected_run)

    pre_ckpt = safe_torch_load(pretrain_path)
    ttt_ckpt = safe_torch_load(ttt_path)

    dataset = args.dataset or ttt_ckpt.get("dataset", pre_ckpt.get("dataset", ""))
    domain = args.domain or ttt_ckpt.get("domain", pre_ckpt.get("domain", ""))
    shift = args.shift or ttt_ckpt.get("shift", pre_ckpt.get("shift", ""))
    data, masks, data_path = load_good_data(dataset, domain, shift, device)

    indices = mask_to_indices(masks, args.sample_mask, int(data.x.shape[0]))
    indices = subsample_indices(indices, int(args.max_nodes), int(args.seed))
    print("Data:", data_path)
    print("Nodes used for t-SNE:", int(indices.numel()))

    pre_embed, ttt_embed = compute_embeddings(data, pre_ckpt, pretrain_path, ttt_ckpt, device)
    idx_np = indices.numpy()
    pre_np = pre_embed[idx_np].numpy()
    ttt_np = ttt_embed[idx_np].numpy()

    if not args.no_standardize:
        scaler = StandardScaler()
        if args.separate_tsne:
            pre_np = scaler.fit_transform(pre_np)
            ttt_np = StandardScaler().fit_transform(ttt_np)
        else:
            stacked_for_scaler = np.vstack([pre_np, ttt_np])
            stacked_for_scaler = scaler.fit_transform(stacked_for_scaler)
            pre_np = stacked_for_scaler[: pre_np.shape[0]]
            ttt_np = stacked_for_scaler[pre_np.shape[0] :]

    if args.separate_tsne:
        pre_coords = fit_tsne(pre_np, args)
        ttt_coords = fit_tsne(ttt_np, args)
    else:
        stacked = np.vstack([pre_np, ttt_np])
        coords = fit_tsne(stacked, args)
        pre_coords = coords[: pre_np.shape[0]]
        ttt_coords = coords[pre_np.shape[0] :]

    out_dir = Path(args.output_dir) if args.output_dir else selected_run / "tsne"
    out_dir.mkdir(parents=True, exist_ok=True)

    y_labels = data.y.detach().cpu()[indices].numpy().astype(int).tolist()
    split_labels = build_split_labels(masks, indices)
    save_plots(out_dir, pre_coords, ttt_coords, y_labels, split_labels, selected_run.name, args)

    np.savez(
        out_dir / "tsne_embeddings.npz",
        indices=idx_np,
        y=np.asarray(y_labels),
        split=np.asarray(split_labels),
        pretrain_tsne=pre_coords,
        ttt_tsne=ttt_coords,
        pretrain_embedding=pre_np,
        ttt_embedding=ttt_np,
    )

    meta = {
        "dataset": dataset,
        "domain": domain,
        "shift": shift,
        "data_path": str(data_path),
        "pretrain_ckpt": str(pretrain_path),
        "ttt_ckpt": str(ttt_path),
        "selected_run": str(selected_run),
        "run_regex": args.run_regex,
        "select": args.select,
        "sample_mask": args.sample_mask,
        "max_nodes": int(args.max_nodes),
        "num_nodes_used": int(indices.numel()),
        "perplexity": float(args.perplexity),
        "max_iter": int(args.max_iter),
        "separate_tsne": bool(args.separate_tsne),
        "standardized": not bool(args.no_standardize),
    }
    with (out_dir / "tsne_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("Saved:", out_dir / "tsne_by_label.png")
    print("Saved:", out_dir / "tsne_by_split.png")
    print("Saved:", out_dir / "tsne_embeddings.npz")
    print("Saved:", out_dir / "tsne_meta.json")


if __name__ == "__main__":
    main()
