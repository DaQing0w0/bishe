#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path


def parse_hex(value: str) -> int:
    return int(value.strip(), 16)


def resolve_path(base_dir: Path, p: Path) -> Path:
    return p if p.is_absolute() else (base_dir / p)


def load_mem_counts(mem_csv: Path) -> dict[int, Counter[int]]:
    counts: dict[int, Counter[int]] = defaultdict(Counter)

    with mem_csv.open("r", newline="") as f:
        reader = csv.DictReader(f)
        required = {"epoch", "page_vnum"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{mem_csv} 缺少列: {sorted(missing)}")

        for row in reader:
            epoch = int(row["epoch"])
            page_vnum = parse_hex(row["page_vnum"])
            counts[epoch][page_vnum] += 1

    return counts


def process_epoch_alloc(
    alloc_csv: Path,
    counts_by_epoch: dict[int, Counter[int]],
    output_csv: Path,
) -> None:
    with alloc_csv.open("r", newline="") as f:
        reader = csv.DictReader(f)
        required = {"seq", "epoch", "vaddr_hex"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{alloc_csv} 缺少列: {sorted(missing)}")

        output_fields = list(reader.fieldnames or []) + ["access_count"]
        rows = []

        for row in reader:
            epoch = int(row["epoch"])
            page_vnum = parse_hex(row["vaddr_hex"])
            row["access_count"] = str(counts_by_epoch.get(epoch, Counter()).get(page_vnum, 0))
            rows.append(row)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=output_fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="统计每个 epoch 中每个 seq 对应页面的访问次数"
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="包含 mem.csv 的目录，默认当前脚本目录",
    )
    parser.add_argument(
        "--mem-csv",
        type=Path,
        default=Path("mem.csv"),
        help="mem.csv 路径（可相对 base-dir）",
    )
    parser.add_argument(
        "--alloc-dir",
        type=Path,
        default=Path("minerva_page_alloc_trace"),
        help="包含 epoch_xxxx_page_alloc.csv 的目录（可相对 base-dir）",
    )
    parser.add_argument(
        "--alloc-csvs",
        type=Path,
        nargs="+",
        default=None,
        help="页面分配 CSV 路径列表（可相对 base-dir）；若不传则按默认 0/1/2 三个文件",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("minerva_page_alloc_trace"),
        help="输出目录（可相对 base-dir）",
    )
    parser.add_argument(
        "--output-csvs",
        type=Path,
        nargs="*",
        default=None,
        help="输出 CSV 路径列表（可相对 base-dir），数量需与 --alloc-csvs 一致",
    )
    parser.add_argument(
        "--keep-default-outputs",
        action="store_true",
        help="使用 --output-csvs 时，保留同名默认输出文件（默认会删除以避免重复）",
    )
    args = parser.parse_args()

    base_dir = args.base_dir.resolve()
    alloc_dir = args.alloc_dir if args.alloc_dir.is_absolute() else base_dir / args.alloc_dir
    output_dir = args.output_dir if args.output_dir.is_absolute() else base_dir / args.output_dir

    mem_csv = resolve_path(base_dir, args.mem_csv)
    counts_by_epoch = load_mem_counts(mem_csv)

    if args.alloc_csvs:
        alloc_csvs = [resolve_path(base_dir, p) for p in args.alloc_csvs]
    else:
        alloc_csvs = sorted(alloc_dir.glob("epoch_*_page_alloc.csv"))
        if not alloc_csvs:
            raise ValueError(f"在目录 {alloc_dir} 下未找到 epoch_*_page_alloc.csv")

    default_output_csvs = []
    for alloc_csv in alloc_csvs:
        out_name = alloc_csv.name.replace("_page_alloc.csv", "_seq_access_count.csv")
        if out_name == alloc_csv.name:
            out_name = alloc_csv.stem + "_seq_access_count.csv"
        default_output_csvs.append(output_dir / out_name)

    if args.output_csvs is not None and len(args.output_csvs) > 0:
        if len(args.output_csvs) != len(alloc_csvs):
            raise ValueError("--output-csvs 的数量必须与 --alloc-csvs 数量一致")
        output_csvs = [resolve_path(base_dir, p) for p in args.output_csvs]
    else:
        output_csvs = default_output_csvs

    if args.output_csvs is not None and len(args.output_csvs) > 0 and not args.keep_default_outputs:
        for custom_output, default_output in zip(output_csvs, default_output_csvs):
            if custom_output.resolve() != default_output.resolve() and default_output.exists():
                default_output.unlink()
                print(f"已删除默认输出以避免重复: {default_output}")

    for alloc_csv, output_csv in zip(alloc_csvs, output_csvs):
        process_epoch_alloc(alloc_csv, counts_by_epoch, output_csv)
        print(f"已生成: {output_csv}")


if __name__ == "__main__":
    main()
