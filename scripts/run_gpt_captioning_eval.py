#!/usr/bin/env python3
"""
Standalone GPT-4o captioning judge — bypasses the pointllm package to avoid
importing torch (which is slow on NFS). Only depends on `openai` (>=1.0) and
`tqdm`.

Runs the same object-captioning prompt as pointllm.eval.evaluator:
OpenAIObjectCaptioningEvaluator. Output format is identical so eval_tools/aggregate.py
can consume it.

Usage:
    python scripts/run_gpt_captioning_eval.py PRED_JSON [PRED_JSON ...] \
        --judge gpt-4o-2024-08-06 --num_workers 15
"""
import argparse
import json
import os
import random
import re
import sys
import time
from multiprocessing.pool import ThreadPool as Pool

from openai import (
    OpenAI,
    RateLimitError,
    APITimeoutError,
    APIConnectionError,
    InternalServerError,
)
from tqdm import tqdm

# ------------------------------------------------------------------
# Copied verbatim from pointllm/eval/evaluator.py — must stay in sync
GPT_PROMPT = """Evaluate a model-generated caption against a human-generated caption (ground truth) for a 3D model. Identify the aspects mentioned in the human caption and calculate the percentage of these aspects correctly mentioned or partially matched in the model caption. Score from 0 to 100, where each aspect contributes equally to the score. Consider similar concepts for partial score.
Provide your score (0-100) and a short justification (less than 15 words) in the format of 'score#reason'

Human-generated caption: {ground_truth}
Model-generated caption: {model_output}

Your response:"""

GPT_PRICES = {
    "gpt-4o-2024-08-06":      {"in": 0.0025, "out": 0.010},
    "gpt-4o-2024-11-20":      {"in": 0.0025, "out": 0.010},
    "gpt-4o-mini-2024-07-18": {"in": 0.00015, "out": 0.00060},
    "gpt-4-turbo-2024-04-09": {"in": 0.010,  "out": 0.030},
    "gpt-4-0613":             {"in": 0.030,  "out": 0.060},
}
# ------------------------------------------------------------------


def retry_bo(func, max_retries=40, max_delay=30, errors=(RateLimitError, APITimeoutError, APIConnectionError, InternalServerError)):
    def wrapper(*args, **kwargs):
        delay = 1.0
        for i in range(max_retries):
            try:
                return func(*args, **kwargs)
            except errors as e:
                delay = min(delay * 2 * (1 + random.random()), max_delay)
                time.sleep(delay)
        raise RuntimeError(f"Max retries ({max_retries}) exceeded")
    return wrapper


def parse_score(resp):
    match = re.search(r"(\d+)#(.*)", resp.strip())
    if match:
        try:
            s = int(match.group(1))
            if 0 <= s <= 100:
                return s, match.group(2).strip()
        except ValueError:
            pass
    return -1, resp.strip()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("results_paths", nargs="+", help="pred_run*.json files to judge")
    p.add_argument("--judge", default="gpt-4o-2024-08-06", choices=list(GPT_PRICES.keys()))
    p.add_argument("--num_workers", type=int, default=15)
    p.add_argument("--force", action="store_true", help="re-run even if output already exists")
    args = p.parse_args()

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    price = GPT_PRICES[args.judge]

    @retry_bo
    def judge_one(inp):
        gt, mo = inp["ground_truth"], inp["model_output"]
        msg = [{"role": "user", "content": GPT_PROMPT.format(ground_truth=gt, model_output=mo)}]
        r = client.chat.completions.create(
            model=args.judge, messages=msg, temperature=1, top_p=1, max_tokens=2048,
        ).model_dump()
        content = r["choices"][0]["message"]["content"]
        sc, reason = parse_score(content)
        u = r["usage"]
        return {
            "object_id": inp.get("object_id", -1),
            "ground_truth": gt,
            "model_output": mo,
            "gpt_score": sc,
            "gpt_reason": reason,
            "prompt_tokens": u["prompt_tokens"],
            "completion_tokens": u["completion_tokens"],
        }

    for results_path in args.results_paths:
        out_path = results_path.replace(".json", f"_evaluated_{args.judge}.json")
        if os.path.exists(out_path) and not args.force:
            print(f"[SKIP] {out_path} already exists")
            continue
        print(f"\n>>> {results_path}")
        data = json.load(open(results_path))
        results = data["results"]
        scored = []
        total_score = 0
        invalid = 0
        pt = ct = 0
        with Pool(args.num_workers) as pool, tqdm(total=len(results)) as bar:
            for r in pool.imap_unordered(judge_one, results):
                scored.append(r)
                if r["gpt_score"] == -1:
                    invalid += 1
                else:
                    total_score += r["gpt_score"]
                pt += r["prompt_tokens"]
                ct += r["completion_tokens"]
                bar.update()

        valid = len(scored) - invalid
        avg = total_score / valid if valid else 0
        cost = pt / 1000 * price["in"] + ct / 1000 * price["out"]
        out = {
            "inference_prompt": data.get("prompt"),
            "gpt_prompt": GPT_PROMPT,
            "average_score": f"{avg:.2f}",
            "total_score": f"{total_score:.2f}",
            "total_predictions": len(scored),
            "invalid_responses": invalid,
            "prompt_tokens": pt,
            "completion_tokens": ct,
            "GPT_cost": f"{cost:.2f}",
            "results": scored,
        }
        json.dump(out, open(out_path, "w"), indent=2)
        print(f"    avg_score={avg:.2f}  invalid={invalid}  cost=${cost:.2f}")
        print(f"    saved -> {out_path}")


if __name__ == "__main__":
    main()
