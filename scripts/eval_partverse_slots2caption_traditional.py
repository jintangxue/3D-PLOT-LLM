#!/usr/bin/env python3
"""
Compute traditional captioning metrics (BLEU-1..4 / ROUGE-1/2/L / METEOR /
SBERT / SimCSE) on PartVerse slots2caption pred JSONs by adapting them to
the input format expected by `pointllm.eval.traditional_evaluator`.

Source files: `<ckpt>/evaluation*/*_partverse_slots2caption_pred*.json`
Output: `<orig>_evaluated_traditional.json` saved next to the source.

Run with the `pointllm_eval` env (has sentence-transformers / scipy / nltk).

Usage:
  PY_EVAL=$PLOT_PYTHON_EVAL
  PYTHONPATH=$PLOT_ROOT \\
    $PY_EVAL scripts/eval_partverse_slots2caption_traditional.py \\
      --pattern '$PLOT_ROOT/outputs/PointLLM_train_stage2/*/evaluation*/eval_partverse_slots2caption_*pred*.json'
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Any, Dict, List


def _adapt(in_path: str) -> Dict[str, Any]:
    with open(in_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    src = data.get("results") or []
    out: List[Dict[str, Any]] = []
    for r in src:
        oid = r.get("object_id", "")
        gt = r.get("ground_truth")
        mo = r.get("model_output")
        if gt is None:
            gt = r.get("gold", "")
        if mo is None:
            mo = r.get("prediction", "")
        out.append({"object_id": oid, "ground_truth": gt, "model_output": mo})
    summary = data.get("summary") or {}
    prompt_str = (
        f"partverse_slots2caption (anno={summary.get('anno_path','?')})"
    )
    return {"prompt": prompt_str, "results": out}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--pattern",
        required=True,
        help="Glob for slots2caption pred JSONs.",
    )
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-run even if *_evaluated_traditional.json already exists.",
    )
    ap.add_argument(
        "--dry_run",
        action="store_true",
        help="Print what would be processed and exit.",
    )
    args = ap.parse_args()

    paths = sorted(glob.glob(args.pattern))
    if not paths:
        print(f"[partverse-trad] No files matched: {args.pattern}", file=sys.stderr)
        sys.exit(1)

    todo: List[str] = []
    for p in paths:
        if "_evaluated_traditional" in p:
            continue
        out_dir = os.path.dirname(p)
        out_file = os.path.basename(p).replace(".json", "_evaluated_traditional.json")
        out_path = os.path.join(out_dir, out_file)
        if (not args.overwrite) and os.path.isfile(out_path):
            print(f"[skip] already exists: {out_path}")
            continue
        todo.append(p)

    print(f"[partverse-trad] {len(todo)} files to process out of {len(paths)} matched")
    if args.dry_run:
        for p in todo:
            print(f"  would process: {p}")
        return

    # Lazy import: traditional_evaluator pulls heavy deps (SBERT/SimCSE).
    from pointllm.eval.traditional_evaluator import start_evaluation

    for i, p in enumerate(todo, 1):
        out_dir = os.path.dirname(p)
        out_file = os.path.basename(p).replace(".json", "_evaluated_traditional.json")
        print(f"[{i}/{len(todo)}] {p}")
        adapted = _adapt(p)
        if not adapted["results"]:
            print(f"  [warn] no results in source; skipping")
            continue
        try:
            start_evaluation(results=adapted, output_dir=out_dir, output_file=out_file)
        except Exception as e:
            print(f"  [error] {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
