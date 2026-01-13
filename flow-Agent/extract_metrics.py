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
import json
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


def parse_wirelength(base_path: Path) -> Optional[float]:
    """
    Prefer detailed route JSON (5_2_route.json) for wirelength; fall back to GRT log.
    """
    route_json = base_path / "5_2_route.json"
    if route_json.exists():
        try:
            with route_json.open("r", encoding="utf-8", errors="ignore") as fh:
                data = json.load(fh)
            wl = data.get("detailedroute__route__wirelength")
            if wl is not None:
                return float(wl)
        except Exception as e:
            print(f"[WARN] Failed to parse wirelength from {route_json}: {e}")

    # fallback to global route log regex
    wl_str = parse_numeric(WIRELENGTH_RE, base_path / "5_1_grt.log")
    return float(wl_str) if wl_str is not None else None


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

    parser = argparse.ArgumentParser(
        description="Extract wirelength and clock period metrics from result_dump logs."
    )
    parser.add_argument(
        "-i",
        "--input",
        required=True,
        type=Path,
        help="Root directory containing result_dump_* directories (e.g., backup_dir/<platform>/<design>).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output markdown file, e.g., output_results/<platform>_<design>_<ts>.md. If omitted, auto-generate under output_results/.",
    )
    args = parser.parse_args()

    source_root = args.input.resolve()
    # Auto-generate output path when not provided
    if args.output:
        output_path = args.output if args.output.is_absolute() else Path.cwd() / args.output
    else:
        platform = source_root.parent.name if source_root.parent else "unknown"
        design = source_root.name
        ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = repo_root / "output_results" / f"{platform}_{design}_{ts}.md"

    rows: list[dict[str, str]] = []
    for result_id, base_id, base_path in discover_results(source_root):
        clock = parse_numeric(CLOCK_RE, base_path / "1_1_yosys_canonicalize.log")
        wirelength = parse_wirelength(base_path)
        rows.append(
            {
                "result": result_id,
                "base": base_id,
                "clock": clock or "N/A",
                "wirelength": f"{wirelength} um" if wirelength is not None else "N/A",
            }
        )

    output_md = build_markdown(rows, source_root)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(output_md, encoding="utf-8")


if __name__ == "__main__":
    main()

# python extract_metrics.py -i backup_dir/asap7/aes -o output_results/asap7_aes_20250101_1200.md
