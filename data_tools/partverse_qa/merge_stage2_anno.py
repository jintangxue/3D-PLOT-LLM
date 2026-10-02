#!/usr/bin/env python3
"""
Merge a base PointLLM Stage2 JSON array with PartVerse Stage2 part JSON.

Base and part files must each be a JSON list of objects (same top-level shape as
PointLLM_complex_instruction_70K.json and build_stage2_part_json output).

Part subsampling:
  - Default: keep all part rows, then concatenate base + part.
  - --part_target_fraction R  (0<R<1): after merge, about R of rows are part.
    Keeps all base; takes Np = floor(R * Nb / (1-R)) part rows from the pool.
  - --max_part_samples M (M>0): cap part count to min(M, pool size).
  If both are set, Np = min(computed_from_R, M, len(part)).
  - --part_sample_strategy union_iou_desc (default) or random: how to choose
    which part rows when Np < len(part). union_iou_desc prefers higher meta.union_iou.
    --seed affects random strategy and final shuffle only.

Training: pass merged JSON as --anno_path and extend conversation_types with
partverse_caption2slots, partverse_slots2caption.

Usage:
  python -m data_tools.partverse_qa.merge_stage2_anno \\
    --base_json /path/PointLLM_complex_instruction_70K.json \\
    --part_json outputs/stage2_part_k16_iou05.json \\
    --output outputs/stage2_merged_70k_part25.json \\
    --part_target_fraction 0.25 --seed 42
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, List, Optional, Tuple


def _load_array(path: Path) -> List[Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON array in {path}, got {type(data).__name__}")
    return data


def _compute_part_take(
    nb: int,
    len_part: int,
    part_target_fraction: Optional[float],
    max_part_samples: int,
) -> int:
    """How many part rows to include after subsampling."""
    n = len_part
    if part_target_fraction is not None:
        r = part_target_fraction
        if not (0.0 < r < 1.0):
            raise ValueError("--part_target_fraction must be strictly between 0 and 1")
        target = int(nb * r / (1.0 - r))
        n = min(n, max(target, 0))
    if max_part_samples > 0:
        n = min(n, max_part_samples)
    return n


def _row_union_iou(row: Any) -> float:
    if not isinstance(row, dict):
        return float("-inf")
    meta = row.get("meta")
    if not isinstance(meta, dict):
        return float("-inf")
    v = meta.get("union_iou", 0.0)
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("-inf")


def _select_part_rows(
    part: List[Any],
    take: int,
    strategy: str,
    rng: random.Random,
) -> List[Any]:
    if take >= len(part):
        return list(part)
    if strategy == "random":
        return rng.sample(part, take)
    if strategy == "union_iou_desc":
        indexed: List[Tuple[int, Any]] = list(enumerate(part))
        indexed.sort(
            key=lambda t: (
                -_row_union_iou(t[1]),
                str((t[1] or {}).get("object_id", "")) if isinstance(t[1], dict) else "",
                t[0],
            )
        )
        return [t[1] for t in indexed[:take]]
    raise ValueError(f"Unknown part_sample_strategy: {strategy!r}")


def merge_stage2_arrays(
    base: List[Any],
    part: List[Any],
    *,
    part_target_fraction: Optional[float] = None,
    max_part_samples: int = 0,
    part_sample_strategy: str = "union_iou_desc",
    seed: int = 42,
    shuffle_merged: bool = True,
) -> tuple[List[Any], dict]:
    """
    Merge in memory. Use part_target_fraction=None and max_part_samples=0 to append
    the full part list. Returns (merged_rows, stats dict).
    """
    nb = len(base)
    np_ = len(part)
    take = _compute_part_take(nb, np_, part_target_fraction, max_part_samples)
    rng = random.Random(seed)
    part_sel = _select_part_rows(part, take, part_sample_strategy, rng)
    merged = base + part_sel
    if shuffle_merged:
        rng.shuffle(merged)
    stats = {
        "base_rows": nb,
        "part_pool_rows": np_,
        "part_rows_used": len(part_sel),
        "merged_rows": len(merged),
        "part_share": (len(part_sel) / len(merged)) if merged else 0.0,
    }
    return merged, stats


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base_json", type=str, required=True)
    p.add_argument("--part_json", type=str, required=True)
    p.add_argument("--output", type=str, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--part_target_fraction",
        type=float,
        default=None,
        help="Target fraction of part rows in final list (0,1). All base kept; part subsampled.",
    )
    p.add_argument(
        "--max_part_samples",
        type=int,
        default=0,
        help="Hard cap on part rows (0 = no cap beyond other limits).",
    )
    p.add_argument(
        "--no_shuffle",
        action="store_true",
        help="Keep base block first, then part block (default: shuffle merged list).",
    )
    p.add_argument(
        "--part_sample_strategy",
        type=str,
        choices=("union_iou_desc", "random"),
        default="union_iou_desc",
        help="How to pick part rows when subsampling (default: highest union_iou first).",
    )
    args = p.parse_args()

    base_path = Path(args.base_json)
    part_path = Path(args.part_json)
    out_path = Path(args.output)

    base = _load_array(base_path)
    part = _load_array(part_path)
    merged, st = merge_stage2_arrays(
        base,
        part,
        part_target_fraction=args.part_target_fraction,
        max_part_samples=args.max_part_samples,
        part_sample_strategy=args.part_sample_strategy,
        seed=args.seed,
        shuffle_merged=not args.no_shuffle,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    print(
        f"Wrote {st['merged_rows']} rows to {out_path} "
        f"(base {st['base_rows']}, part used {st['part_rows_used']} / pool {st['part_pool_rows']}; "
        f"part share {st['part_share']:.4f})",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
