#!/usr/bin/env python3
"""
Select one query per held-out object from an evaluation json (used by build_splits.py).

Full holdout files are often ~5+ rows per object (short/long captions, etc.).
This script keeps the highest-meta.union_iou rows per object by default.

Example (200 objects -> 200 rows, one slot task per object):
  python -m data_tools.partverse_qa.subset_partverse_eval \\
    --in eval_c2s.json --out eval_c2s_one_per_object.json \\
    --per_object 1

Use the --out JSON with:
  python pointllm/eval/eval_partverse_caption2slots.py --anno_path <out> ...

Modes:
  --per_object K   keep top K rows per object_id by union_iou (default 1)
  --max_rows N     optional hard cap after per-object selection (random tie-break, seed)
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional


def _iou(row: Dict[str, Any]) -> float:
    m = row.get("meta")
    if not isinstance(m, dict):
        return float("-inf")
    try:
        return float(m.get("union_iou", 0.0))
    except (TypeError, ValueError):
        return float("-inf")


def subset_partverse_rows(
    rows: List[Dict[str, Any]],
    *,
    per_object: int = 1,
    max_rows: Optional[int] = None,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    by_oid: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        if not isinstance(r, dict):
            continue
        oid = str(r.get("object_id", ""))
        if oid:
            by_oid[oid].append(r)

    out: List[Dict[str, Any]] = []
    for oid in sorted(by_oid.keys()):
        grp = by_oid[oid]
        grp.sort(key=lambda r: (-_iou(r), str(r.get("meta", {}).get("caption_source", ""))))
        out.extend(grp[:per_object])

    if max_rows is not None and len(out) > max_rows:
        rng = random.Random(seed)
        rng.shuffle(out)
        out = out[:max_rows]
        out.sort(key=lambda r: str(r.get("object_id", "")))
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--in", dest="inp", type=str, required=True)
    p.add_argument("--out", type=str, required=True)
    p.add_argument(
        "--per_object",
        type=int,
        default=1,
        help="Keep this many rows per object_id (highest union_iou first).",
    )
    p.add_argument(
        "--max_rows",
        type=int,
        default=None,
        help="If set, randomly subsample to at most N rows after per-object selection.",
    )
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    in_path = Path(args.inp)
    with open(in_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise SystemExit("Input must be a JSON array")

    rows = [r for r in data if isinstance(r, dict)]
    sel = subset_partverse_rows(
        rows,
        per_object=args.per_object,
        max_rows=args.max_rows,
        seed=args.seed,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(sel, f, ensure_ascii=False, indent=2)

    n_obj = len({str(r["object_id"]) for r in sel if r.get("object_id")})
    print(
        f"Wrote {len(sel)} rows ({n_obj} objects) from {len(rows)} rows "
        f"({len({str(r['object_id']) for r in rows if r.get('object_id')})} objects) -> {out_path}"
    )


if __name__ == "__main__":
    main()
