#!/usr/bin/env python3
"""Compute auto-release KPIs for lenet/vgg16/minerva runs.

KPIs reported:
1) avg_resident_bytes
2) peak_resident_bytes
3) avg_epoch_runtime
4) avg_page_lifetime

Usage:
    python calc_auto_release_kpis_multi.py \
        --lenet-dir /path/to/lenet \
        --vgg16-dir /path/to/vgg16 \
        --minerva-dir /path/to/minerva

The script auto-detects dry-run vs enforce files based on CSV headers and
prints KPIs for both modes when available.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


EPOCH_RE = re.compile(r"epoch_(\d+)", re.IGNORECASE)


@dataclass
class AutoRow:
    epoch: int
    seq: int
    alloc_time: Optional[float]
    release_time: Optional[float]
    candidate_release_time: Optional[float]
    lifetime: Optional[float]
    candidate_lifetime: Optional[float]
    epoch_runtime: Optional[float]


@dataclass
class AutoFileData:
    path: Path
    mode: str
    rows: List[AutoRow]


def str_to_int(val: str) -> Optional[int]:
    if val is None:
        return None
    v = val.strip()
    if v == "":
        return None
    try:
        return int(v)
    except ValueError:
        return None


def str_to_float(val: str) -> Optional[float]:
    if val is None:
        return None
    v = val.strip()
    if v == "":
        return None
    try:
        return float(v)
    except ValueError:
        return None


def epoch_from_filename(path: Path) -> Optional[int]:
    m = EPOCH_RE.search(path.name)
    if not m:
        return None
    return int(m.group(1))


def load_csv_header(path: Path) -> List[str]:
    with path.open("r", newline="") as f:
        reader = csv.reader(f)
        try:
            return next(reader)
        except StopIteration:
            return []


def classify_csv(path: Path) -> str:
    header = set(load_csv_header(path))
    if not header:
        return "unknown"

    auto_required = {"seq", "epoch"}
    auto_hints = {
        "would_release",
        "post_threshold_access",
        "alloc_time",
        "release_time",
        "candidate_release_time",
    }
    if auto_required.issubset(header) and header.intersection(auto_hints):
        return "auto"

    return "unknown"


def detect_mode(fields: Iterable[str]) -> str:
    field_set = set(fields)
    if "did_release" in field_set or "release_time" in field_set:
        return "enforce"
    if "candidate_release_time" in field_set:
        return "dry_run"
    return "unknown"


def load_auto_file(path: Path) -> AutoFileData:
    rows: List[AutoRow] = []

    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        mode = detect_mode(reader.fieldnames or [])
        for r in reader:
            epoch = str_to_int(r.get("epoch", ""))
            seq = str_to_int(r.get("seq", ""))
            if epoch is None or seq is None:
                guessed_epoch = epoch_from_filename(path)
                if guessed_epoch is None or seq is None:
                    continue
                epoch = guessed_epoch

            rows.append(
                AutoRow(
                    epoch=epoch,
                    seq=seq,
                    alloc_time=str_to_float(r.get("alloc_time", "")),
                    release_time=str_to_float(r.get("release_time", "")),
                    candidate_release_time=str_to_float(r.get("candidate_release_time", "")),
                    lifetime=str_to_float(r.get("lifetime", "")),
                    candidate_lifetime=str_to_float(r.get("candidate_lifetime", "")),
                    epoch_runtime=str_to_float(r.get("epoch_runtime", "")),
                )
            )

    return AutoFileData(path=path, mode=mode, rows=rows)


def iter_csv_files(data_dir: Path) -> Iterable[Path]:
    yield from data_dir.rglob("*.csv")


def choose_release_time(row: AutoRow) -> Optional[float]:
    if row.release_time is not None:
        return row.release_time
    if row.candidate_release_time is not None:
        return row.candidate_release_time
    return None


def choose_lifetime(row: AutoRow) -> Optional[float]:
    if row.lifetime is not None:
        return row.lifetime
    if row.candidate_lifetime is not None:
        return row.candidate_lifetime
    if row.alloc_time is not None:
        rt = choose_release_time(row)
        if rt is not None:
            return rt - row.alloc_time
    return None


def safe_mean(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return sum(values) / len(values)


def compute_resident_metrics(
    rows_by_epoch: Dict[int, List[AutoRow]],
    page_size: int,
) -> Tuple[Optional[float], Optional[float]]:
    avg_resident_per_epoch: List[float] = []
    peak_bytes_global = 0.0

    for epoch, rows in rows_by_epoch.items():
        events: List[Tuple[float, int]] = []
        epoch_runtime: Optional[float] = None

        for row in rows:
            if row.alloc_time is None:
                continue

            rel_t = choose_release_time(row)
            if rel_t is None or rel_t < row.alloc_time:
                continue

            events.append((row.alloc_time, +1))
            events.append((rel_t, -1))

            if epoch_runtime is None and row.epoch_runtime is not None and row.epoch_runtime > 0:
                epoch_runtime = row.epoch_runtime

        if not events or epoch_runtime is None or epoch_runtime <= 0:
            continue

        events.sort(key=lambda x: (x[0], -x[1]))

        resident_pages = 0
        last_t = events[0][0]
        area_pages_time = 0.0
        peak_pages = 0

        for t, delta in events:
            dt = t - last_t
            if dt > 0:
                area_pages_time += resident_pages * dt
            resident_pages += delta
            if resident_pages > peak_pages:
                peak_pages = resident_pages
            last_t = t

        avg_pages = area_pages_time / epoch_runtime
        avg_resident_per_epoch.append(avg_pages * page_size)
        peak_bytes_global = max(peak_bytes_global, peak_pages * page_size)

    return safe_mean(avg_resident_per_epoch), (peak_bytes_global if peak_bytes_global > 0 else None)


def compute_kpis_for_files(auto_files: List[AutoFileData], page_size: int) -> Dict[str, object]:
    rows_by_epoch: Dict[int, List[AutoRow]] = defaultdict(list)
    epoch_runtime_by_epoch: Dict[int, float] = {}
    lifetime_values: List[float] = []

    for af in auto_files:
        for r in af.rows:
            rows_by_epoch[r.epoch].append(r)

            if r.epoch_runtime is not None and r.epoch_runtime > 0:
                if r.epoch not in epoch_runtime_by_epoch:
                    epoch_runtime_by_epoch[r.epoch] = r.epoch_runtime

            lt = choose_lifetime(r)
            if lt is not None and lt >= 0:
                lifetime_values.append(lt)

    avg_resident_bytes, peak_resident_bytes = compute_resident_metrics(rows_by_epoch, page_size)

    return {
        "auto_files": len(auto_files),
        "epochs_found": sorted(rows_by_epoch.keys()),
        "kpis": {
            "avg_resident_bytes": avg_resident_bytes,
            "peak_resident_bytes": peak_resident_bytes,
            "avg_epoch_runtime": safe_mean(list(epoch_runtime_by_epoch.values())),
            "avg_page_lifetime": safe_mean(lifetime_values),
        },
    }


def compute_kpis_for_dir(data_dir: Path, page_size: int) -> Dict[str, object]:
    auto_files_by_mode: Dict[str, List[AutoFileData]] = {
        "dry_run": [],
        "enforce": [],
        "unknown": [],
    }

    for p in iter_csv_files(data_dir):
        if classify_csv(p) == "auto":
            auto_file = load_auto_file(p)
            auto_files_by_mode.setdefault(auto_file.mode, []).append(auto_file)

    return {
        "by_mode": {
            "dry_run": compute_kpis_for_files(auto_files_by_mode.get("dry_run", []), page_size),
            "enforce": compute_kpis_for_files(auto_files_by_mode.get("enforce", []), page_size),
        },
    }


def fmt_val(val: Optional[float], digits: int = 6) -> str:
    if val is None:
        return "N/A"
    return f"{val:.{digits}f}"


def print_table(results: Dict[str, Dict[str, object]]) -> None:
    headers = [
        "benchmark",
        "mode",
        "avg_resident_bytes",
        "peak_resident_bytes",
        "avg_epoch_runtime",
        "avg_page_lifetime",
    ]

    rows: List[List[str]] = []
    for name in ("lenet", "vgg16", "minerva"):
        data = results.get(name, {})
        by_mode = data.get("by_mode", {})
        for mode in ("dry_run", "enforce"):
            kpis = by_mode.get(mode, {}).get("kpis", {})
            rows.append(
                [
                    name,
                    mode,
                    fmt_val(kpis.get("avg_resident_bytes")),
                    fmt_val(kpis.get("peak_resident_bytes")),
                    fmt_val(kpis.get("avg_epoch_runtime"), 9),
                    fmt_val(kpis.get("avg_page_lifetime"), 9),
                ]
            )

    table = [headers] + rows
    col_widths = [max(len(str(row[i])) for row in table) for i in range(len(headers))]

    def fmt_row(row: List[str]) -> str:
        return "  ".join(str(row[i]).ljust(col_widths[i]) for i in range(len(headers)))

    print(fmt_row(headers))
    print("  ".join("-" * w for w in col_widths))
    for row in rows:
        print(fmt_row(row))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute selected auto-release KPIs for lenet/vgg16/minerva."
    )
    parser.add_argument("--lenet-dir", required=True, help="Directory to scan for lenet CSV files.")
    parser.add_argument("--vgg16-dir", required=True, help="Directory to scan for vgg16 CSV files.")
    parser.add_argument("--minerva-dir", required=True, help="Directory to scan for minerva CSV files.")
    parser.add_argument(
        "--page-size",
        type=int,
        default=4096,
        help="Page size in bytes for resident-memory KPIs (default: 4096).",
    )
    parser.add_argument(
        "--output-json",
        action="store_true",
        help="Print JSON output instead of a table.",
    )

    args = parser.parse_args()

    dirs = {
        "lenet": Path(args.lenet_dir).expanduser().resolve(),
        "vgg16": Path(args.vgg16_dir).expanduser().resolve(),
        "minerva": Path(args.minerva_dir).expanduser().resolve(),
    }

    for name, p in dirs.items():
        if not p.exists() or not p.is_dir():
            raise SystemExit(f"{name} data directory not found: {p}")

    results = {name: compute_kpis_for_dir(p, args.page_size) for name, p in dirs.items()}

    if args.output_json:
        payload = {"page_size": args.page_size, "results": results}
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print_table(results)


if __name__ == "__main__":
    main()
