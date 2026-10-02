#!/usr/bin/env python3
"""
Build the PartVerse-QA split and the merged Stage 2 training files.

Inputs
  --base_json   PointLLM_complex_instruction_70K.json (Stage 2 base data)
  --part_json   PartVerse-QA pair pool from build_stage2_part_json.py / filter_stage2_part_json.py
  --part_holdout_exclude_brief_json
                PointLLM_brief_description_660K_filtered.json (Stage 1 data, read only)

Held-out objects are sampled from the PartVerse objects that appear in neither the 70K file
nor the Stage 1 brief-description file, so evaluation objects receive no object-level
supervision in either training stage. All pairs of the held-out objects are removed from
the training pool.

Outputs (under --out_dir)
  heldout_object_ids.json         sorted list of held-out object ids
  train_pairs.json                PartVerse-QA training pairs (held-out objects removed)
  eval_c2s.json / eval_s2c.json   caption-to-slots / slots-to-caption evaluation queries
  *_one_per_object.json           one query per held-out object (highest union IoU)
  train_stage2.json               70K + all PartVerse-QA training pairs
  train_stage2_pv<percent>.json   70K + a subset of the pairs; <percent> is the share of the pool used
  split_manifest.json             seed, counts and file paths of this build

Partial training files (data-scaling study) can be requested in two ways. --fractions gives the
target share of PartVerse-QA pairs in the merged file; --pool_fractions gives the fraction of
the PartVerse-QA training pool. Both take the pairs with the highest union IoU. Pool-fraction
files merge the pairs into the shuffled 70K rows (the order of train_stage2_pv0.json) and
shuffle again with the same seed. The released PartVerse-QA files were built with
  --filter_min_union_iou 0.5 --merge_seed 42 --fractions 0,0.15,0.25,full --pool_fractions 0.5,0.75
which writes train_stage2_pv0/16/30/50/75.json and train_stage2.json.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from .merge_stage2_anno import merge_stage2_arrays
from .subset_partverse_eval import subset_partverse_rows


def _row_iou(row: Any) -> float:
    if not isinstance(row, dict):
        return float("-inf")
    m = row.get("meta")
    if not isinstance(m, dict):
        return float("-inf")
    try:
        return float(m.get("union_iou", 0.0))
    except (TypeError, ValueError):
        return float("-inf")


def _filter_part_iou(rows: List[Any], thr: Optional[float]) -> List[Any]:
    if thr is None:
        return rows
    return [r for r in rows if isinstance(r, dict) and _row_iou(r) >= thr]


def _unique_object_ids(rows: List[Any]) -> Set[str]:
    ids: Set[str] = set()
    for r in rows:
        if isinstance(r, dict) and r.get("object_id"):
            ids.add(str(r["object_id"]))
    return ids


def _exclude_objects(rows: List[Any], banned: Set[str]) -> List[Any]:
    return [
        r
        for r in rows
        if isinstance(r, dict) and str(r.get("object_id", "")) not in banned
    ]


def _load_json_array(path: Path, what: str) -> List[Any]:
    print(f"Loading {what} {path} ...", file=sys.stderr)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise SystemExit(f"{what} must be a JSON array: {path}")
    return data


def _write_json(path: Path, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _sample_holdout_from_pool(
    eligible: List[str],
    n_objects: int,
    fraction: Optional[float],
    seed: int,
) -> List[str]:
    if not eligible:
        raise ValueError("Empty holdout pool.")
    rng = random.Random(seed)
    pool = list(eligible)
    rng.shuffle(pool)
    if fraction is not None:
        n_h = max(1, int(round(len(pool) * fraction)))
    else:
        n_h = int(n_objects)
    n_h = max(1, min(n_h, len(pool)))
    return sorted(pool[:n_h])


def _parse_fractions(s: str) -> List[Optional[float]]:
    out: List[Optional[float]] = []
    for tok in s.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if tok.lower() == "full":
            out.append(None)
            continue
        v = float(tok)
        if v == 0.0:
            out.append(0.0)
        elif 0.0 < v < 1.0:
            out.append(v)
        else:
            raise ValueError(f"Invalid fraction token {tok!r}; use 0, a value in (0,1), or full")
    if not out:
        raise ValueError("Empty --fractions")
    return out


def _holdout_eval_for_conv_type(
    part_f: List[Any],
    holdout: Set[str],
    conversation_type: str,
) -> List[Dict[str, Any]]:
    return [
        r
        for r in part_f
        if isinstance(r, dict)
        and str(r.get("object_id", "")) in holdout
        and r.get("conversation_type") == conversation_type
    ]


def _write_eval_split(
    rows: List[Dict[str, Any]],
    out_dir: Path,
    basename: str,
    seed: int,
) -> Tuple[Path, Path, List[Dict[str, Any]]]:
    """Write <basename>.json (all queries) and <basename>_one_per_object.json."""
    full_path = out_dir / f"{basename}.json"
    one_path = out_dir / f"{basename}_one_per_object.json"
    _write_json(full_path, rows)
    one_rows = subset_partverse_rows(rows, per_object=1, max_rows=None, seed=seed)
    _write_json(one_path, one_rows)
    return full_path, one_path, one_rows


def main() -> None:
    pointllm_data = os.environ.get("POINTLLM_DATA", "data/pointllm")
    partverse_qa_dir = os.environ.get("PARTVERSE_QA_DIR", "data/partverse_qa")

    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument(
        "--base_json",
        type=str,
        default=os.path.join(pointllm_data, "anno_data/PointLLM_complex_instruction_70K.json"),
    )
    p.add_argument(
        "--part_json",
        type=str,
        default=os.path.join(partverse_qa_dir, "stage2_part_k16_iou04_parttok.json"),
        help="PartVerse-QA pair pool (output of build_stage2_part_json.py / filter_stage2_part_json.py).",
    )
    p.add_argument(
        "--part_holdout_exclude_brief_json",
        type=str,
        default=os.path.join(pointllm_data, "anno_data/PointLLM_brief_description_660K_filtered.json"),
        help="Stage 1 brief-description file; its object ids are excluded from the holdout pool (read only).",
    )
    p.add_argument(
        "--filter_min_union_iou",
        type=float,
        default=None,
        help="If set, drop part rows with meta.union_iou below this value (e.g. 0.5).",
    )
    p.add_argument(
        "--holdout_n_objects",
        type=int,
        default=200,
        help="Number of held-out objects sampled from the eligible pool "
        "(PartVerse objects outside the 70K and Stage 1 files). Ignored if --holdout_object_fraction is set.",
    )
    p.add_argument(
        "--holdout_object_fraction",
        type=float,
        default=None,
        help="If set, overrides --holdout_n_objects: n = round(fraction * |eligible pool|).",
    )
    p.add_argument(
        "--holdout_object_ids_file",
        type=str,
        default="",
        help="JSON array of object ids to use as the exact holdout list instead of sampling "
        "(every id must lie in the eligible pool).",
    )
    p.add_argument(
        "--fractions",
        type=str,
        default="full",
        help="Comma-separated target shares of PartVerse-QA pairs in the merged training file: 0, "
        "values in (0,1), or full. 'full' writes train_stage2.json; the others write train_stage2_pv<percent>.json.",
    )
    p.add_argument(
        "--pool_fractions",
        type=str,
        default="",
        help="Comma-separated fractions of the PartVerse-QA training pool to merge with the 70K rows "
        "(values in (0,1)); each writes train_stage2_pv<percent>.json.",
    )
    p.add_argument("--merge_seed", type=int, default=42)
    p.add_argument(
        "--part_sample_strategy",
        type=str,
        choices=("union_iou_desc", "random"),
        default="union_iou_desc",
        help="How partial shares pick pairs from the training pool.",
    )
    p.add_argument("--no_shuffle_merged", action="store_true")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    part_path = Path(args.part_json)
    base_path = Path(args.base_json)
    brief_path = Path(args.part_holdout_exclude_brief_json)

    part_all = _load_json_array(part_path, "part pool")
    part_f = _filter_part_iou(part_all, args.filter_min_union_iou)
    if args.filter_min_union_iou is not None:
        print(
            f"Filtered part rows by union_iou >= {args.filter_min_union_iou}: "
            f"{len(part_all)} -> {len(part_f)}",
            file=sys.stderr,
        )
    base = _load_json_array(base_path, "Stage 2 base")
    brief_train_ids = _unique_object_ids(_load_json_array(brief_path, "Stage 1 brief"))

    part_ids = _unique_object_ids(part_f)
    base_ids = _unique_object_ids(base)
    eligible = sorted(part_ids - (base_ids | brief_train_ids))
    if not eligible:
        raise ValueError(
            "Empty holdout pool: no PartVerse object ids outside the 70K and Stage 1 files."
        )

    fixed_holdout_path = (args.holdout_object_ids_file or "").strip()
    if fixed_holdout_path:
        hp = Path(fixed_holdout_path)
        if not hp.is_file():
            raise SystemExit(f"--holdout_object_ids_file not found: {hp}")
        holdout_list = [str(x) for x in _load_json_array(hp, "holdout ids")]
        elig_set = set(eligible)
        bad = [x for x in holdout_list if x not in elig_set]
        if bad:
            raise SystemExit(
                f"holdout_object_ids_file: {len(bad)} ids are not in the eligible pool. "
                f"Examples: {bad[:8]}"
            )
        print(f"Using fixed holdout list from {hp.resolve()} ({len(holdout_list)} objects)", file=sys.stderr)
    else:
        if args.holdout_object_fraction is None and len(eligible) < args.holdout_n_objects:
            print(
                f"WARNING: eligible pool has {len(eligible)} objects < "
                f"holdout_n_objects={args.holdout_n_objects}; using all of them.",
                file=sys.stderr,
            )
        holdout_list = _sample_holdout_from_pool(
            eligible, args.holdout_n_objects, args.holdout_object_fraction, args.merge_seed
        )
    holdout = set(holdout_list)
    assert not (holdout & base_ids) and not (holdout & brief_train_ids)
    print(
        f"Holdout: {len(holdout_list)} objects from an eligible pool of {len(eligible)} "
        f"(PartVerse objects {len(part_ids)}, 70K objects {len(base_ids)}, "
        f"Stage 1 objects {len(brief_train_ids)}).",
        file=sys.stderr,
    )

    train_pool = _exclude_objects(part_f, holdout)
    eval_c2s_rows = _holdout_eval_for_conv_type(part_f, holdout, "partverse_caption2slots")
    eval_s2c_rows = _holdout_eval_for_conv_type(part_f, holdout, "partverse_slots2caption")

    holdout_path = out_dir / "heldout_object_ids.json"
    _write_json(holdout_path, holdout_list)
    pool_path = out_dir / "train_pairs.json"
    _write_json(pool_path, train_pool)

    c2s_path, c2s_one_path, c2s_one = _write_eval_split(eval_c2s_rows, out_dir, "eval_c2s", args.merge_seed)
    s2c_path, s2c_one_path, s2c_one = _write_eval_split(eval_s2c_rows, out_dir, "eval_s2c", args.merge_seed)
    print(
        f"Eval: caption-to-slots {len(eval_c2s_rows)} queries ({len(c2s_one)} one-per-object); "
        f"slots-to-caption {len(eval_s2c_rows)} queries ({len(s2c_one)} one-per-object). "
        f"Training pool: {len(train_pool)} PartVerse-QA pairs, {len(base)} 70K rows.",
        file=sys.stderr,
    )

    jobs: List[Tuple[str, Optional[float], int]] = [
        ("share", frac, 0) for frac in _parse_fractions(args.fractions)
    ]
    for frac in (_parse_fractions(args.pool_fractions) if args.pool_fractions.strip() else []):
        if frac is None or frac == 0.0:
            raise ValueError("--pool_fractions takes values strictly between 0 and 1")
        jobs.append(("pool", frac, int(round(frac * len(train_pool)))))

    base_shuffled = list(base)
    if not args.no_shuffle_merged:
        random.Random(args.merge_seed).shuffle(base_shuffled)

    merged_entries: List[Dict[str, Any]] = []
    for kind, frac, cap in jobs:
        if frac == 0.0:
            merged = list(base_shuffled)
            stats = {
                "base_rows": len(base),
                "part_pool_rows": len(train_pool),
                "part_rows_used": 0,
                "merged_rows": len(merged),
                "part_share": 0.0,
            }
        else:
            merged, stats = merge_stage2_arrays(
                base if kind == "share" else base_shuffled,
                train_pool,
                part_target_fraction=frac if kind == "share" else None,
                max_part_samples=cap,
                part_sample_strategy=args.part_sample_strategy,
                seed=args.merge_seed,
                shuffle_merged=not args.no_shuffle_merged,
            )
        pv_pct = round(100.0 * stats["part_rows_used"] / max(stats["part_pool_rows"], 1))
        out_merged = out_dir / ("train_stage2.json" if pv_pct == 100 else f"train_stage2_pv{pv_pct}.json")
        _write_json(out_merged, merged)
        merged_entries.append(
            {
                ("part_target_share" if kind == "share" else "pool_fraction"): frac,
                "path": str(out_merged.resolve()),
                **stats,
            }
        )
        print(f"Wrote {out_merged} ({stats})", file=sys.stderr)

    manifest: Dict[str, Any] = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "merge_seed": args.merge_seed,
        "base_json": str(base_path.resolve()),
        "part_json": str(part_path.resolve()),
        "stage1_brief_json": str(brief_path.resolve()),
        "filter_min_union_iou": args.filter_min_union_iou,
        "part_sample_strategy": args.part_sample_strategy,
        "holdout_source": str(Path(fixed_holdout_path).resolve()) if fixed_holdout_path else "sampled",
        "holdout_count": len(holdout_list),
        "eligible_pool_count": len(eligible),
        "heldout_object_ids": str(holdout_path.resolve()),
        "train_pairs": str(pool_path.resolve()),
        "train_pairs_count": len(train_pool),
        "eval_c2s": str(c2s_path.resolve()),
        "eval_c2s_one_per_object": str(c2s_one_path.resolve()),
        "eval_s2c": str(s2c_path.resolve()),
        "eval_s2c_one_per_object": str(s2c_one_path.resolve()),
        "eval_counts": {
            "c2s": len(eval_c2s_rows),
            "c2s_one_per_object": len(c2s_one),
            "s2c": len(eval_s2c_rows),
            "s2c_one_per_object": len(s2c_one),
        },
        "merged_training_files": merged_entries,
    }
    man_path = out_dir / "split_manifest.json"
    _write_json(man_path, manifest)
    print(f"Wrote {man_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
