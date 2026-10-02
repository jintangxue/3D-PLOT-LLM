#!/usr/bin/env python3
"""
Filter an existing PartVerse Stage2 part JSON by meta.union_iou (no rebuild).

build_stage2_part_json only drops rows when union_iou < min_union_iou, so:

  filter(json_iou04, min_union_iou=0.5)  ==  rebuild with --min_union_iou 0.5

(Up to row order: this script preserves the original order of kept rows.)

Usage:
  python -m data_tools.partverse_qa.filter_stage2_part_json \\
    --in outputs/stage2_part_k16_iou04_parttok.json \\
    --out outputs/stage2_part_k16_iou05_parttok_filtered.json \\
    --min_union_iou 0.5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List


def _row_iou(row: Dict[str, Any]) -> float:
    meta = row.get("meta")
    if not isinstance(meta, dict):
        return float("-inf")
    v = meta.get("union_iou", 0.0)
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("-inf")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--in", dest="inp", type=str, required=True)
    p.add_argument("--out", type=str, required=True)
    p.add_argument(
        "--min_union_iou",
        type=float,
        required=True,
        help="Keep rows with meta.union_iou >= this value.",
    )
    args = p.parse_args()

    in_path = Path(args.inp)
    with open(in_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise SystemExit(f"Expected JSON array in {in_path}")

    thr = args.min_union_iou
    kept: List[Dict[str, Any]] = []
    dropped = 0
    for row in data:
        if not isinstance(row, dict):
            dropped += 1
            continue
        if _row_iou(row) >= thr:
            kept.append(row)
        else:
            dropped += 1

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(kept, f, ensure_ascii=False, indent=2)

    print(
        f"Kept {len(kept)} / {len(data)} rows (dropped {dropped}) "
        f"with union_iou >= {thr} -> {out_path}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
