from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch run stage3 TTT for all gate_model.pt under a stage2 directory"
    )
    parser.add_argument(
        "--stage2-dir",
        type=str,
        required=True,
        help="Stage2 directory to scan, e.g. ./outputs/stage2/citeseer/word/covariate",
    )
    parser.add_argument(
        "--stage3-params",
        type=str,
        default="templates/stage3.json",
        help="Path to stage3 params JSON",
    )
    parser.add_argument(
        "--ttt-script",
        type=str,
        default="ttt.py",
        help="Path to stage3 script",
    )
    parser.add_argument(
        "--gate-file-name",
        type=str,
        default="gate_model.pt",
        help="Checkpoint filename to search under stage2 dir",
    )
    parser.add_argument(
        "--max-runs",
        type=int,
        default=0,
        help="If > 0, only run the first N checkpoints",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only list checkpoints and exit",
    )
    parser.add_argument(
        "--summary-json",
        type=str,
        default="",
        help="Optional path to save summary JSON",
    )
    return parser.parse_args()


def load_json(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError("JSON root must be an object: {}".format(path))
    return obj


def find_gate_ckpts(stage2_dir: Path, gate_file_name: str) -> List[Path]:
    # Search recursively to support many run folders under one stage2 folder.
    ckpts = sorted(stage2_dir.rglob(gate_file_name))
    return [p for p in ckpts if p.is_file()]


def sanitize_tag(text: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_-]+", "-", text)
    safe = re.sub(r"-+", "-", safe).strip("-")
    return safe or "run"


def make_timestamp(idx: int, gate_path: Path) -> str:
    now = datetime.now().strftime("%Y%m%d-%H%M%S")
    parent_tag = sanitize_tag(gate_path.parent.name)
    return "batch{:03d}_{}_{}".format(idx, parent_tag, now)


def read_ood_test_acc(metrics_path: Path) -> Optional[float]:
    if not metrics_path.exists():
        return None
    metrics = load_json(metrics_path)
    value = metrics.get("ood_test_acc")
    if value is None:
        return None
    return float(value)


def main():
    args = parse_args()

    stage2_dir = Path(args.stage2_dir)
    if not stage2_dir.exists() or not stage2_dir.is_dir():
        raise FileNotFoundError("Invalid --stage2-dir: {}".format(stage2_dir))

    stage3_params_path = Path(args.stage3_params)
    if not stage3_params_path.exists():
        raise FileNotFoundError("Missing --stage3-params: {}".format(stage3_params_path))

    ttt_script_path = Path(args.ttt_script)
    if not ttt_script_path.exists():
        raise FileNotFoundError("Missing --ttt-script: {}".format(ttt_script_path))

    stage3_params = load_json(stage3_params_path)
    output_root = str(stage3_params.get("output_root", "./outputs"))
    dataset = str(stage3_params.get("dataset", "")).lower()
    domain = str(stage3_params.get("domain", "")).lower()
    shift = str(stage3_params.get("shift", "")).lower()
    if not dataset or not domain or not shift:
        raise ValueError("stage3 params must include dataset/domain/shift")

    gate_ckpts = find_gate_ckpts(stage2_dir, args.gate_file_name)
    total_found = len(gate_ckpts)
    print("Found {} gate checkpoints under {}".format(total_found, stage2_dir))

    if total_found == 0:
        return

    if args.max_runs > 0:
        gate_ckpts = gate_ckpts[: int(args.max_runs)]
        print("Limited to first {} checkpoints by --max-runs".format(len(gate_ckpts)))

    for i, ckpt in enumerate(gate_ckpts, start=1):
        print("  [{}/{}] {}".format(i, len(gate_ckpts), ckpt))

    if args.dry_run:
        return

    results = []
    success = 0
    failed = 0

    for i, gate_ckpt in enumerate(gate_ckpts, start=1):
        ts = make_timestamp(i, gate_ckpt)
        print("\n[{}/{}] Running gate checkpoint: {}".format(i, len(gate_ckpts), gate_ckpt))
        print("  timestamp: {}".format(ts))

        cmd = [
            sys.executable,
            str(ttt_script_path),
            "--params-file",
            str(stage3_params_path),
            "--gate-ckpt",
            str(gate_ckpt),
            "--timestamp",
            ts,
        ]
        ret = subprocess.run(cmd)

        run_out_dir = Path(output_root) / "stage3" / dataset / domain / shift / ts
        metrics_path = run_out_dir / "metrics.json"
        ood_test_acc = read_ood_test_acc(metrics_path)

        one = {
            "gate_ckpt": str(gate_ckpt),
            "return_code": int(ret.returncode),
            "timestamp": ts,
            "output_dir": str(run_out_dir),
            "metrics_path": str(metrics_path),
            "ood_test_acc": ood_test_acc,
        }
        results.append(one)

        if ret.returncode == 0 and ood_test_acc is not None:
            success += 1
            print("  ood_test_acc={:.6f}".format(ood_test_acc))
        else:
            failed += 1
            print("  run failed or metrics missing (return_code={}, metrics={})".format(ret.returncode, ood_test_acc))

    valid = [r for r in results if r["ood_test_acc"] is not None and r["return_code"] == 0]
    best = None
    if valid:
        best = max(valid, key=lambda x: float(x["ood_test_acc"]))

    print("\nBatch done")
    print("  total_found={}".format(total_found))
    print("  total_ran={}".format(len(gate_ckpts)))
    print("  success_with_metrics={}".format(success))
    print("  failed_or_missing_metrics={}".format(failed))
    if best is None:
        print("  best_ood_test_acc=None")
    else:
        print("  best_ood_test_acc={:.6f}".format(float(best["ood_test_acc"])))
        print("  best_gate_ckpt={}".format(best["gate_ckpt"]))
        print("  best_metrics_path={}".format(best["metrics_path"]))

    summary = {
        "stage2_dir": str(stage2_dir),
        "stage3_params": str(stage3_params_path),
        "total_found": int(total_found),
        "total_ran": int(len(gate_ckpts)),
        "success_with_metrics": int(success),
        "failed_or_missing_metrics": int(failed),
        "best": best,
        "results": results,
    }

    if args.summary_json:
        summary_path = Path(args.summary_json)
    else:
        summary_path = stage2_dir / "ttt_batch_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("Saved summary: {}".format(summary_path))


if __name__ == "__main__":
    main()
