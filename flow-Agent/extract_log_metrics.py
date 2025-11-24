#!/usr/bin/env python3
"""
Extract wirelength and clock period metrics from result_dump logs and write a markdown report.

Targets:
- 5_1_grt.log: line containing "[INFO GRT-0018] Total wirelength:"
- 1_1_yosys_canonicalize.log: line containing "Setting clock period to"
"""

from __future__ import annotations

import argparse
import datetime as _dt
import re
from pathlib import Path
from typing import Iterable, Optional, Tuple


WIRELENGTH_RE = re.compile(r"\[INFO GRT-0018\]\s*Total wirelength:\s*([0-9.]+)\s*um", re.IGNORECASE)
CLOCK_RE = re.compile(r"Setting clock period to\s*([0-9.]+)", re.IGNORECASE)


def parse_numeric(pattern: re.Pattern[str], path: Path) -> Optional[str]:
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                match = pattern.search(line)
                if match:
                    return match.group(1)
    except FileNotFoundError:
        return None
    return None


def discover_results(base_dir: Path) -> Iterable[Tuple[str, str, Path]]:
    for result_dir in sorted(base_dir.glob("result_dump_*"), key=_numeric_suffix):
        logs_root = result_dir / "logs_dump"
        if not logs_root.is_dir():
            continue
        for base_dir in sorted(logs_root.glob("base_*"), key=_numeric_suffix):
            yield (
                result_dir.name.replace("result_dump_", ""),
                base_dir.name.replace("base_", ""),
                base_dir,
            )


def _numeric_suffix(path: Path) -> int:
    stem = path.name.split("_")[-1]
    try:
        return int(stem)
    except ValueError:
        return 0


def build_markdown(rows: list[dict[str, str]], source_root: Path) -> str:
    timestamp = _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "# Log Metrics Summary",
        "",
        f"- Source root: `{source_root}`",
        f"- Generated at: {timestamp}",
        "",
        "| result_dump | base | clock_period | total_wirelength |",
        "| --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['result']} | {row['base']} | {row['clock']} | {row['wirelength']} |"
        )
    if not rows:
        lines.append("| (none) | (none) | (none) | (none) |")
    return "\n".join(lines) + "\n"


def main() -> None:
    repo_root = Path(__file__).resolve().parent
    default_output = repo_root / "output_results" / "asap7" / "log_metrics.md"

    parser = argparse.ArgumentParser(
        description="Extract wirelength and clock period metrics from result_dump logs."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=default_output,
        help=f"Output markdown file (default: {default_output})",
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=repo_root / "backup_dir" / "asap7" / "aes",
        help="Root directory containing result_dump_* directories.",
    )
    args = parser.parse_args()

    rows: list[dict[str, str]] = []
    for result_id, base_id, base_path in discover_results(args.source):
        clock = parse_numeric(CLOCK_RE, base_path / "1_1_yosys_canonicalize.log")
        wirelength = parse_numeric(WIRELENGTH_RE, base_path / "5_1_grt.log")
        rows.append(
            {
                "result": result_id,
                "base": base_id,
                "clock": clock or "N/A",
                "wirelength": f"{wirelength} um" if wirelength else "N/A",
            }
        )

    output_md = build_markdown(rows, args.source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(output_md, encoding="utf-8")


if __name__ == "__main__":
    main()
