"""
Summarize tau2 results for a multi-domain Olmo-3 eval run.

Given a list of results.json paths (one per domain), loads each, computes
metrics, and writes a summary.txt with pass^1..pass^N for each domain plus
an overall mean.

Usage:
    uv run python scripts/olmo3_summary.py \
        --output summary.txt \
        retail=data/simulations/.../retail/results.json \
        airline=data/simulations/.../airline/results.json \
        ...
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tau2.data_model.simulation import Results
from tau2.metrics.agent_metrics import compute_metrics


def summarize(domain_paths: dict[str, Path], output_path: Path) -> int:
    lines: list[str] = []
    lines.append(f"tau2-bench Olmo-3 evaluation summary")
    lines.append("=" * 60)
    lines.append("")

    overall_pass1: list[float] = []
    overall_avg_reward: list[float] = []
    missing: list[str] = []

    for domain, path in domain_paths.items():
        if not path.exists():
            missing.append(f"{domain}: {path} not found")
            lines.append(f"[{domain}] MISSING — {path}")
            lines.append("")
            continue

        try:
            results = Results.load(path)
            metrics = compute_metrics(results)
        except Exception as e:  # noqa: BLE001
            lines.append(f"[{domain}] ERROR loading {path}: {e}")
            lines.append("")
            missing.append(f"{domain}: load error ({e})")
            continue

        lines.append(f"[{domain}]")
        lines.append(f"  Source:           {path}")
        lines.append(f"  Total tasks:      {metrics.total_tasks}")
        lines.append(f"  Total sims:       {metrics.total_simulations}")
        lines.append(f"  Infra errors:     {metrics.infra_error_count}")
        lines.append(f"  Avg reward:       {metrics.avg_reward:.4f}")
        lines.append(f"  Avg agent cost:   ${metrics.avg_agent_cost:.4f}")
        for k in sorted(metrics.pass_hat_ks):
            v = metrics.pass_hat_ks[k]
            lines.append(f"  pass^{k}:           {v:.4f}")
        lines.append("")

        overall_avg_reward.append(metrics.avg_reward)
        if 1 in metrics.pass_hat_ks:
            overall_pass1.append(metrics.pass_hat_ks[1])

    lines.append("Overall (unweighted mean across loaded domains)")
    lines.append("-" * 60)
    if overall_avg_reward:
        mean_reward = sum(overall_avg_reward) / len(overall_avg_reward)
        lines.append(
            f"  Avg reward:       {mean_reward:.4f}  ({len(overall_avg_reward)} domains)"
        )
    if overall_pass1:
        mean_pass1 = sum(overall_pass1) / len(overall_pass1)
        lines.append(
            f"  pass^1:           {mean_pass1:.4f}  ({len(overall_pass1)} domains)"
        )
    if missing:
        lines.append("")
        lines.append("Missing / errored:")
        for m in missing:
            lines.append(f"  - {m}")

    output = "\n".join(lines) + "\n"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(output)
    print(output, end="")
    return 0 if not missing else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to write the summary.txt to.",
    )
    parser.add_argument(
        "entries",
        nargs="+",
        help="One or more <domain>=<path/to/results.json> entries.",
    )
    args = parser.parse_args()

    domain_paths: dict[str, Path] = {}
    for entry in args.entries:
        if "=" not in entry:
            print(f"Bad entry (expected domain=path): {entry}", file=sys.stderr)
            return 2
        domain, path = entry.split("=", 1)
        domain_paths[domain.strip()] = Path(path.strip())

    return summarize(domain_paths, args.output)


if __name__ == "__main__":
    raise SystemExit(main())
