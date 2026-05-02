from __future__ import annotations

import argparse
import itertools
import subprocess
import sys
from pathlib import Path

from ssl_tasks import parse_ssl_task_names





def parse_args():
    parser = argparse.ArgumentParser(description="Run all n-choose-m SSL combinations for gate stage.")
    parser.add_argument("--ssl-pool", type=str, required=True, help="Comma-separated SSL task pool")
    parser.add_argument("--choose", type=int, required=True, help="Select m tasks from pool")
    parser.add_argument("--stage2json", type=str, required=True, help="Path to stage2 JSON file")
    return parser.parse_args()


def main():
    args = parse_args()

    if not Path(args.stage2json).exists():
        raise FileNotFoundError("Missing stage2 params file: {}".format(args.stage2json))

    pool = parse_ssl_task_names(args.ssl_pool)
    if not pool:
        raise ValueError("No valid SSL task parsed from --ssl-pool")

    m = int(args.choose)
    n = len(pool)
    if m <= 0 or m > n:
        raise ValueError("--choose must satisfy 1 <= choose <= {}".format(n))

    combos = list(itertools.combinations(pool, m))
    print("SSL pool:", pool)
    print("Run {} combinations ({} choose {})".format(len(combos), n, m))

    success = 0
    fail = 0
    for idx, combo in enumerate(combos, start=1):
        combo_str = ",".join(combo)
        cmd = [
            sys.executable,
            "gate_ssl.py",
            "--params-file",
            str(args.stage2json),
            "--ssl-tasks",
            combo_str,
            "--num-ssl",
            str(m),
        ]
        print("\n[{}/{}] Running combo: {}".format(idx, len(combos), combo_str))
        result = subprocess.run(cmd)
        if result.returncode == 0:
            success += 1
        else:
            fail += 1
            print("Combo failed with code {}: {}".format(result.returncode, combo_str))

    print("\nDone. success={}, fail={}".format(success, fail))
    if fail > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
