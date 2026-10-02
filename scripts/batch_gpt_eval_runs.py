#!/usr/bin/env python3
"""
Batch GPT eval across multiple run json files in a single Python process.
Uses the SAME pipeline as `python -m pointllm.eval.evaluator` (same prompt,
same model params, same parse logic) — just amortizes the import cost so NFS
torch-import slowness is paid once instead of N times.

Usage:
    python scripts/batch_gpt_eval_runs.py \
        --ckpt_dir outputs/PointLLM_train_stage2/train_stage2 \
        --runs 1 2 3 4 5 \
        --judge gpt-4o-2024-08-06 \
        --num_workers 15
"""
import argparse
import os
import sys

# Force this repo's pointllm to win over any other installed / sibling pointllm
# (e.g. the PointLLM baseline project in a sibling dir). Must happen before
# `from pointllm.eval.evaluator import ...`.
_THIS_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repository root
if _THIS_REPO not in sys.path:
    sys.path.insert(0, _THIS_REPO)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_dir", required=True, help="ckpt directory containing evaluation/")
    p.add_argument("--runs", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    p.add_argument("--judge", default="gpt-4o-2024-08-06")
    p.add_argument("--num_workers", type=int, default=15)
    p.add_argument("--eval_type", default="object-captioning",
                   choices=["object-captioning", "modelnet-close-set-classification", "open-free-form-classification"])
    p.add_argument("--pred_prefix", default="PointLLM_brief_description_val_200_GT_Objaverse_captioning_prompt2")
    args = p.parse_args()

    # Import once (the slow bit on cold NFS)
    print("[init] importing pointllm.eval.evaluator (slow on cold NFS, ~3-10 min)...", flush=True)
    from pointllm.eval.evaluator import start_evaluation
    print("[init] done. running evals...\n", flush=True)

    eval_dir = os.path.join(args.ckpt_dir, "evaluation")
    assert os.path.isdir(eval_dir), f"no evaluation/ in {args.ckpt_dir}"

    for run in args.runs:
        pred = os.path.join(eval_dir, f"{args.pred_prefix}_run{run}.json")
        if not os.path.exists(pred):
            print(f"[SKIP run{run}] pred not found: {pred}")
            continue
        out_file = os.path.basename(pred).replace(".json", f"_evaluated_{args.judge}.json")
        out_path = os.path.join(eval_dir, out_file)
        if os.path.exists(out_path):
            print(f"[SKIP run{run}] already scored: {out_path}")
            continue

        print(f"=== run{run} ===", flush=True)
        start_evaluation(
            results=pred,
            output_dir=eval_dir,
            output_file=out_file,
            eval_type=args.eval_type,
            model_type=args.judge,
            parallel=True,
            num_workers=args.num_workers,
        )
        print()


if __name__ == "__main__":
    main()
