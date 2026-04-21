#!/usr/bin/env python3
"""Compute auto-release KPIs from experiment CSV files.

The script recursively scans a target directory, auto-detects:
- auto-release CSVs (enforce or dry-run)
- page-allocation trace CSVs

Then it computes 6 KPIs:
1) avg_resident_bytes
2) peak_resident_bytes
3) execution_ratio_did_over_would
4) post_threshold_ratio
5) avg_epoch_runtime
6) avg_page_lifetime (new KPI)

Usage:
  python calc_auto_release_kpis.py --data-dir /path/to/minerva_run_dir
  python calc_auto_release_kpis.py --data-dir /path/to/minerva_run_dir --page-size 4096 --output-json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple


EPOCH_RE = re.compile(r"epoch_(\d+)", re.IGNORECASE)


@dataclass
class AutoRow:
    epoch: int
    seq: int
    would_release: Optional[bool]
    did_release: Optional[bool]
    post_threshold_access: Optional[int]
    alloc_time: Optional[float]
    release_time: Optional[float]
    candidate_release_time: Optional[float]
    lifetime: Optional[float]
    candidate_lifetime: Optional[float]
    epoch_runtime: Optional[float]


@dataclass
class AutoFileData:
    path: Path
    mode: str  # enforce or dry_run or unknown
    rows: List[AutoRow]


@dataclass
class PageAllocFileData:
    path: Path
    epoch: int
    seq_set: Set[int]


def str_to_bool(val: str) -> Optional[bool]:
    if val is None:
        return None
    v = val.strip().lower()
    if v in {"true", "1", "yes", "y"}:
        return True
    if v in {"false", "0", "no", "n"}:
        return False
    return None


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

    if {"seq", "epoch", "would_release", "post_threshold_access"}.issubset(header):
        return "auto"

    if {"seq", "epoch", "pid", "cause", "vaddr_hex", "paddr_hex", "page_size"}.issubset(header):
        return "page_alloc"

    return "unknown"


def load_auto_file(path: Path) -> AutoFileData:
    rows: List[AutoRow] = []

    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        fields = set(reader.fieldnames or [])

        mode = "unknown"
        if "did_release" in fields or "release_time" in fields:
            mode = "enforce"
        elif "candidate_release_time" in fields:
            mode = "dry_run"

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
                    would_release=str_to_bool(r.get("would_release", "")),
                    did_release=str_to_bool(r.get("did_release", "")),
                    post_threshold_access=str_to_int(r.get("post_threshold_access", "")),
                    alloc_time=str_to_float(r.get("alloc_time", "")),
                    release_time=str_to_float(r.get("release_time", "")),
                    candidate_release_time=str_to_float(r.get("candidate_release_time", "")),
                    lifetime=str_to_float(r.get("lifetime", "")),
                    candidate_lifetime=str_to_float(r.get("candidate_lifetime", "")),
                    epoch_runtime=str_to_float(r.get("epoch_runtime", "")),
                )
            )

    return AutoFileData(path=path, mode=mode, rows=rows)


def load_page_alloc_file(path: Path) -> Optional[PageAllocFileData]:
    seq_set: Set[int] = set()
    epoch_guess = epoch_from_filename(path)
    epoch_value: Optional[int] = None

    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            seq = str_to_int(r.get("seq", ""))
            if seq is not None:
                seq_set.add(seq)

            if epoch_value is None:
                ep = str_to_int(r.get("epoch", ""))
                if ep is not None:
                    epoch_value = ep

    epoch = epoch_value if epoch_value is not None else epoch_guess
    if epoch is None:
        return None

    return PageAllocFileData(path=path, epoch=epoch, seq_set=seq_set)


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


def detect_auto_release_mode(enforce_rows: int, dry_rows: int, auto_files: int) -> str:
    if auto_files == 0:
        return "unknown"

    if enforce_rows > 0 and dry_rows > 0:
        return "mixed"

    if enforce_rows > 0:
        return "enforce"

    if dry_rows > 0:
        return "dry-run"

    return "unknown"


def compute_resident_metrics(
    rows_by_epoch: Dict[int, List[AutoRow]],
    page_size: int,
) -> Tuple[Optional[float], Optional[float]]:
    # Returns (avg_resident_bytes_over_epochs, peak_resident_bytes_global)
    avg_resident_per_epoch: List[float] = []
    peak_bytes_global = 0.0

    for epoch, rows in rows_by_epoch.items():
        events: List[Tuple[float, int]] = []
        epoch_runtime: Optional[float] = None

        for row in rows:
            if row.alloc_time is None:
                continue

            rel_t = choose_release_time(row)
            if rel_t is None:
                continue

            if rel_t < row.alloc_time:
                continue

            events.append((row.alloc_time, +1))
            events.append((rel_t, -1))

            if epoch_runtime is None and row.epoch_runtime is not None:
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


def compute_kpis(data_dir: Path, page_size: int) -> Dict[str, object]:
    auto_files: List[AutoFileData] = []
    page_alloc_files: List[PageAllocFileData] = []

    for p in iter_csv_files(data_dir):
        typ = classify_csv(p)
        if typ == "auto":
            auto_files.append(load_auto_file(p))
        elif typ == "page_alloc":
            pa = load_page_alloc_file(p)
            if pa is not None:
                page_alloc_files.append(pa)

    rows_by_epoch: Dict[int, List[AutoRow]] = defaultdict(list)
    enforce_rows = 0
    dry_rows = 0

    for af in auto_files:
        if af.mode == "enforce":
            enforce_rows += len(af.rows)
        elif af.mode == "dry_run":
            dry_rows += len(af.rows)

        for r in af.rows:
            rows_by_epoch[r.epoch].append(r)

    page_alloc_by_epoch: Dict[int, Set[int]] = {}
    for paf in page_alloc_files:
        page_alloc_by_epoch[paf.epoch] = set(paf.seq_set)

    # KPI 3: execution_ratio_did_over_would
    would_cnt = 0
    did_cnt = 0
    has_did_release_field = False

    # KPI 5: post_threshold_ratio
    post_nonzero = 0
    post_den = 0

    # KPI 6: avg_epoch_runtime
    epoch_runtime_by_epoch: Dict[int, float] = {}

    # KPI 7: avg_page_lifetime
    lifetime_values: List[float] = []

    for epoch, rows in rows_by_epoch.items():
        for row in rows:
            if row.would_release is True:
                would_cnt += 1
            if row.did_release is not None:
                has_did_release_field = True
                if row.did_release is True:
                    did_cnt += 1

            if row.post_threshold_access is not None:
                post_den += 1
                if row.post_threshold_access > 0:
                    post_nonzero += 1

            if row.epoch_runtime is not None and row.epoch_runtime > 0:
                if epoch not in epoch_runtime_by_epoch:
                    epoch_runtime_by_epoch[epoch] = row.epoch_runtime

            lt = choose_lifetime(row)
            if lt is not None and lt >= 0:
                lifetime_values.append(lt)

    # KPI 1 & 2
    avg_resident_bytes, peak_resident_bytes = compute_resident_metrics(rows_by_epoch, page_size)

    detected_mode = detect_auto_release_mode(
        enforce_rows=enforce_rows,
        dry_rows=dry_rows,
        auto_files=len(auto_files),
    )

    return {
        "data_dir": str(data_dir),
        "detected_mode": detected_mode,
        "files_detected": {
            "auto_files": len(auto_files),
            "page_alloc_files": len(page_alloc_files),
            "enforce_rows": enforce_rows,
            "dry_run_rows": dry_rows,
        },
        "kpis": {
            "avg_resident_bytes": avg_resident_bytes,
            "peak_resident_bytes": peak_resident_bytes,
            "execution_ratio_did_over_would": ((did_cnt / would_cnt) if would_cnt > 0 else None) if has_did_release_field else None,
            "post_threshold_ratio": (post_nonzero / post_den) if post_den > 0 else None,
            "avg_epoch_runtime": safe_mean(list(epoch_runtime_by_epoch.values())),
            "avg_page_lifetime": safe_mean(lifetime_values),
        },
        "supporting": {
            "epochs_found": sorted(rows_by_epoch.keys()),
            "would_release_count": would_cnt,
            "did_release_count": did_cnt,
            "post_threshold_nonzero": post_nonzero,
            "post_threshold_denominator": post_den,
            "epoch_runtime_by_epoch": dict(sorted(epoch_runtime_by_epoch.items())),
            "page_size_bytes_assumed": page_size,
        },
    }


def print_human_report(result: Dict[str, object]) -> None:
    k = result["kpis"]
    s = result["supporting"]
    f = result["files_detected"]

    def fmt(v: Optional[float], digits: int = 6) -> str:
        if v is None:
            return "N/A"
        if isinstance(v, float):
            return f"{v:.{digits}f}"
        return str(v)

    print("=== Auto-Release KPI Report ===")
    print(f"data_dir: {result['data_dir']}")
    print(f"detected_mode: {result['detected_mode']}")
    print(
        "files: auto={auto_files}, page_alloc={page_alloc}, enforce_rows={enforce_rows}, dry_run_rows={dry_rows}".format(
            auto_files=f["auto_files"],
            page_alloc=f["page_alloc_files"],
            enforce_rows=f["enforce_rows"],
            dry_rows=f["dry_run_rows"],
        )
    )
    print(f"epochs: {s['epochs_found']}")
    print()
    print(f"1) avg_resident_bytes: {fmt(k['avg_resident_bytes'])}")
    print(f"2) peak_resident_bytes: {fmt(k['peak_resident_bytes'])}")
    print(f"3) execution_ratio_did_over_would: {fmt(k['execution_ratio_did_over_would'])}")
    print(f"4) post_threshold_ratio: {fmt(k['post_threshold_ratio'])}")
    print(f"5) avg_epoch_runtime: {fmt(k['avg_epoch_runtime'], 9)}")
    print(f"6) avg_page_lifetime: {fmt(k['avg_page_lifetime'], 9)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute auto-release KPIs from CSV experiments.")
    parser.add_argument("--data-dir", required=True, help="Directory to recursively scan for CSV files.")
    parser.add_argument(
        "--page-size",
        type=int,
        default=4096,
        help="Page size in bytes for resident-memory KPIs (default: 4096).",
    )
    parser.add_argument(
        "--output-json",
        action="store_true",
        help="Print JSON output instead of human-readable summary.",
    )

    args = parser.parse_args()
    data_dir = Path(args.data_dir).expanduser().resolve()
    if not data_dir.exists() or not data_dir.is_dir():
        raise SystemExit(f"data directory not found: {data_dir}")

    result = compute_kpis(data_dir, page_size=args.page_size)

    if args.output_json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print_human_report(result)


if __name__ == "__main__":
    main()
