#!/usr/bin/env python3
"""Run auto-release overhead benchmarks and fit c_access, c_release, c_page.

This script runs Go benchmarks in amd/driver and parses ns/op results.
It reports:
- c_access: ObservePageAccess overhead (ns/op), baseline-subtracted
- c_release: fixed cost per release batch (ns)
- c_page: per-page release cost (ns/page)

Usage:
  python fit_auto_release_overhead.py
  python fit_auto_release_overhead.py --driver-dir /path/to/amd/driver
  python fit_auto_release_overhead.py --input bench.txt
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

BENCH_PATTERN = re.compile(r"^(BenchmarkAutoRelease\S+)\s+\d+\s+([0-9.]+)\s+ns/op")
SIZE_PATTERN = re.compile(r"BenchmarkAutoReleaseProcessDueReleases/size_(\d+)")


def run_benchmarks(driver_dir: Path) -> str:
    env = os.environ.copy()
    env.setdefault("GOMAXPROCS", "1")

    cmd = [
        "go",
        "test",
        "-run",
        "^$",
        "-bench",
        "BenchmarkAutoRelease",
        "-benchmem",
    ]

    result = subprocess.run(
        cmd,
        cwd=str(driver_dir),
        env=env,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    return result.stdout


def parse_ns_per_op(output: str) -> Dict[str, float]:
    results: Dict[str, float] = {}
    for line in output.splitlines():
        m = BENCH_PATTERN.match(line.strip())
        if not m:
            continue
        name = m.group(1)
        ns = float(m.group(2))
        results[name] = ns
    return results


def fit_linear(x: List[float], y: List[float]) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    if len(x) < 2:
        return None, None, None

    x_mean = sum(x) / len(x)
    y_mean = sum(y) / len(y)
    denom = sum((xi - x_mean) ** 2 for xi in x)
    if denom == 0:
        return None, None, None

    slope = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(x, y)) / denom
    intercept = y_mean - slope * x_mean

    ss_tot = sum((yi - y_mean) ** 2 for yi in y)
    ss_res = sum((yi - (intercept + slope * xi)) ** 2 for xi, yi in zip(x, y))
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else None

    return intercept, slope, r2


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit auto-release overhead coefficients.")
    parser.add_argument("--driver-dir", help="Path to amd/driver directory.")
    parser.add_argument("--input", help="Optional benchmark output file to parse.")
    args = parser.parse_args()

    driver_dir = Path(args.driver_dir).resolve() if args.driver_dir else Path(__file__).resolve().parent

    if args.input:
        output = Path(args.input).read_text()
    else:
        output = run_benchmarks(driver_dir)

    results = parse_ns_per_op(output)

    access = results.get("BenchmarkAutoReleaseObserveAccess")
    noop = results.get("BenchmarkAutoReleaseObserveAccessNoop")
    if access is None or noop is None:
        raise SystemExit("missing access benchmark results")

    c_access = access - noop

    sizes: List[float] = []
    times: List[float] = []
    for name, ns in results.items():
        m = SIZE_PATTERN.match(name)
        if not m:
            continue
        sizes.append(float(m.group(1)))
        times.append(ns)

    intercept, slope, r2 = fit_linear(sizes, times)

    print("=== Auto-Release Overhead Fit (ns) ===")
    print(f"c_access: {c_access:.6f} ns/op")

    if intercept is None or slope is None:
        print("c_release: N/A")
        print("c_page: N/A")
    else:
        print(f"c_release: {intercept:.6f} ns/batch")
        print(f"c_page: {slope:.6f} ns/page")
        if r2 is not None:
            print(f"fit_r2: {r2:.6f}")


if __name__ == "__main__":
    main()
