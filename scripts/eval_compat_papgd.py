#!/usr/bin/env python
"""
Step 4: Run model inference on 3DCoMPaT-GrIn PaPGD samples and emit a results
JSON in PointLLM's `traditional_evaluator.py` format so we can reuse the
existing BLEU / ROUGE / METEOR / SBERT / SimCSE pipeline.

Output format (per run):
  <eval_dir>/<anno_basename>_compat_papgd_pred_run{i}.json
    {
      "inference_prompt": "(per-sample, varies)",
      "results": [{"object_id", "ground_truth", "model_output"}, ...],
      "summary": {"num_samples", "seed", "run_id", ...}
    }

After this, run on the same JSON:
    pointllm/eval/traditional_evaluator.py --results_path <out_json>

Usage (on A100):
  cd $PLOT_ROOT
  python scripts/eval_compat_papgd.py \\
      --model_name outputs/PointLLM_train_stage2/finetune_grin_plotllm \\
      --anno_path  $COMPAT_DATA/anno/compat_valid_multiple_parts.json \\
      --data_path  $COMPAT_DATA/npy \\
      --bcp_pack_dir $COMPAT_DATA/packs_k16 \\
      --num_runs 5
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

# Prepend the repository root to sys.path so the local pointllm package wins.
# (Caller normally `cd $PLOT_ROOT` first;
#  this is a defensive fallback when invoked with absolute path.)
_HERE = os.path.dirname(os.path.abspath(__file__))
for _candidate in [
    os.path.dirname(_HERE),                                     # if script lives inside the project
    os.environ.get("PLOT_ROOT", os.getcwd()),
]:
    if os.path.isdir(os.path.join(_candidate, "pointllm")) and _candidate not in sys.path:
        sys.path.insert(0, _candidate)
        break

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from pointllm.conversation import SeparatorStyle, conv_templates
from pointllm.data.utils import preprocess_multimodal_point_cloud, pc_norm
from pointllm.model import PointLLMLlamaForCausalLM
from pointllm.model.marker_context_utils import resolve_marker_context_mode
from pointllm.model.utils import KeywordsStoppingCriteria


def disable_torch_init():
    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)


def _gold_text(row: Dict[str, Any]) -> str:
    conv = row.get("conversations") or []
    if len(conv) >= 2 and conv[1].get("from") == "gpt":
        return str(conv[1].get("value", ""))
    return ""


def _human_text(row: Dict[str, Any]) -> str:
    conv = row.get("conversations") or []
    if conv and conv[0].get("from") == "human":
        return str(conv[0].get("value", ""))
    return ""


def _load_pc(data_path: str, object_id: str, pointnum: int, use_color: bool) -> torch.Tensor:
    fn = f"{object_id}_{pointnum}.npy"
    pc = np.load(os.path.join(data_path, fn))
    pc = pc_norm(pc)
    if not use_color:
        pc = pc[:, :3]
    return torch.from_numpy(pc.astype("float32"))


def _build_prompt_ids(human_raw: str, tokenizer, point_backbone_config: Dict[str, Any], conv,
                      point_token_len: Optional[int] = None) -> torch.Tensor:
    pbc = dict(point_backbone_config)
    if point_token_len is not None:
        pbc["point_token_len"] = int(point_token_len)
    sources = [[{"from": "human", "value": human_raw},
                {"from": "gpt", "value": "."}]]
    preprocess_multimodal_point_cloud(sources, pbc)
    human_expanded = sources[0][0]["value"]
    conv = conv.copy()
    conv.messages = []
    conv.append_message(conv.roles[0], human_expanded)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    return tokenizer([prompt], return_tensors="pt").input_ids.cuda()


def _candidate_point_token_lens(pbc: Dict[str, Any]) -> List[int]:
    base = int(pbc.get("point_token_len", 512))
    k = int(pbc.get("bcp_num_groups", 16))
    cands = [base, 1 + 2 * k + 512, 1 + k + 512, 1 + 3 * k + 512, 513, 512]
    out, seen = [], set()
    for c in cands:
        if c > 0 and c not in seen:
            out.append(c); seen.add(c)
    return out


def _load_bcp(bcp_pack_dir: Optional[str], object_id: str, pbc: Dict[str, Any], dtype):
    """Return dict of bcp tensors; empty dict if no pack available."""
    if not bcp_pack_dir:
        return {}
    pack_path = os.path.join(bcp_pack_dir, f"{object_id}.pack.npz")
    if not os.path.isfile(pack_path):
        return {}
    out: Dict[str, Optional[torch.Tensor]] = dict(
        bcp_patch_feat=None, bcp_cls_feat=None, bcp_group_ids=None,
        bcp_region_adj=None, bcp_region_stats=None,
    )
    with np.load(pack_path) as pack:
        out["bcp_patch_feat"] = torch.from_numpy(pack["patch_feat"].astype("float32")).unsqueeze(0).cuda().to(dtype)
        if "cls_feat" in pack:
            out["bcp_cls_feat"] = torch.from_numpy(pack["cls_feat"].astype("float32")).unsqueeze(0).cuda().to(dtype)
        elif pbc.get("bcp_part_vocab_token", False) or pbc.get("bcp_part_interleave", False):
            raise RuntimeError(f"Missing cls_feat in pack: {pack_path}")

        k = pbc.get("bcp_num_groups", 16)
        key_r = f"patch_r{k}"
        key_map = f"map64_to_{k}"
        if key_r in pack:
            g = pack[key_r].astype("int32")
        elif "patch_r64" in pack and key_map in pack:
            g = pack[key_map][pack["patch_r64"]].astype("int32")
        else:
            g = np.zeros(512, dtype="int32")
        out["bcp_group_ids"] = torch.from_numpy(g).unsqueeze(0).cuda()

        _mctx = resolve_marker_context_mode(
            pbc.get("bcp_marker_context_mode"),
            pbc.get("bcp_marker_context_stats", False),
            pbc.get("bcp_marker_context_graph", False),
        )
        if pbc.get("bcp_graph_proj_d", False) or _mctx != "none":
            patch_xyz = pack["patch_xyz"]
            edges = pack["patch_adj_edges"]
            region_adj = torch.zeros((k, k), dtype=torch.float32)
            for u, v in edges:
                gu, gv = int(g[u]), int(g[v])
                if gu != gv:
                    region_adj[gu, gv] = 1.0
                    region_adj[gv, gu] = 1.0
            region_stats = torch.zeros((k, 7), dtype=torch.float32)
            for i in range(k):
                mask = g == i
                pts = patch_xyz[mask]
                region_stats[i, 3] = float(mask.sum()) / 512.0
                if len(pts) > 0:
                    region_stats[i, 0:3] = torch.from_numpy(pts.mean(axis=0))
                    region_stats[i, 4:7] = torch.from_numpy(pts.max(axis=0) - pts.min(axis=0))
            out["bcp_region_adj"] = region_adj.unsqueeze(0).cuda()
            out["bcp_region_stats"] = region_stats.unsqueeze(0).cuda()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", required=True)
    ap.add_argument("--tokenizer_name", default=None)
    ap.add_argument("--anno_path", required=True,
                    help="Converted Compat anno JSON (PointLLM format).")
    ap.add_argument("--data_path", required=True, help="Dir of {object_id}_{pointnum}.npy")
    ap.add_argument("--bcp_pack_dir", default=None,
                    help="If model uses BCP packs, point to {object_id}.pack.npz dir.")
    ap.add_argument("--pointnum", type=int, default=8192)
    ap.add_argument("--use_color", action="store_true", default=True)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=1.0,
                    help="0.0 = greedy. Open-ended generation should use 1.0 + sampling.")
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--top_k", type=int, default=50)
    ap.add_argument("--num_runs", type=int, default=1,
                    help="Number of independent sampling runs (each saved as a separate file).")
    ap.add_argument("--run_start", type=int, default=1)
    ap.add_argument("--seed", type=int, default=None,
                    help="Optional. If set, fixes torch RNG (use with --num_runs=1 for reproducibility).")
    ap.add_argument("--limit", type=int, default=0,
                    help="Limit number of samples (testing).")
    ap.add_argument("--conversation_type_prefix", default="compat_papgd_",
                    help="Filter rows whose conversation_type starts with this prefix.")
    args = ap.parse_args()

    if args.seed is not None:
        import random
        random.seed(args.seed); np.random.seed(args.seed)
        torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)

    disable_torch_init()
    model_name = os.path.expanduser(args.model_name)
    tokenizer_name = args.tokenizer_name or model_name
    print(f"[compat-eval] Loading model {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=False)
    model = PointLLMLlamaForCausalLM.from_pretrained(
        model_name, low_cpu_mem_usage=False, use_cache=True, torch_dtype=torch.bfloat16
    ).cuda()
    model.initialize_tokenizer_point_backbone_config_wo_embedding(tokenizer)

    conv_template = conv_templates["vicuna_v1_1"]
    conv = conv_template.copy()
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    pbc = model.get_model().point_backbone_config

    with open(args.anno_path, "r") as f:
        rows: List[Dict[str, Any]] = json.load(f)
    rows = [r for r in rows if str(r.get("conversation_type", "")).startswith(args.conversation_type_prefix)]
    if args.limit > 0:
        rows = rows[: args.limit]
    print(f"[compat-eval] {len(rows)} samples after filter `{args.conversation_type_prefix}*`")

    out_dir = os.path.join(model_name, "evaluation")
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(args.anno_path))[0]

    do_sample = args.temperature > 0
    active_len = int(pbc.get("point_token_len", 512))

    for run_idx in range(args.run_start, args.run_start + args.num_runs):
        out_file = f"{base}_compat_papgd_pred_run{run_idx}.json"
        out_path = os.path.join(out_dir, out_file)
        if os.path.exists(out_path):
            print(f"[compat-eval] run {run_idx} exists, skipping: {out_path}")
            continue

        print(f"\n[compat-eval] === Run {run_idx} ===")
        results: List[Dict[str, Any]] = []
        n = 0

        for row in tqdm(rows, desc=f"run{run_idx}"):
            oid = row["object_id"]
            human = _human_text(row)
            gold = _gold_text(row)

            try:
                pc = _load_pc(args.data_path, oid, args.pointnum, args.use_color
                              ).unsqueeze(0).cuda().to(model.dtype)
            except FileNotFoundError:
                continue

            try:
                bcp = _load_bcp(args.bcp_pack_dir, oid, pbc, model.dtype)
            except RuntimeError as e:
                print(f"  [skip] bcp load failed for {oid}: {e}")
                continue

            # Try active_len first; on token-len mismatch, search candidates
            tried_lens: List[int] = []
            cand_lens = [active_len] + [c for c in _candidate_point_token_lens(pbc) if c != active_len]
            success = False
            for cand in cand_lens:
                tried_lens.append(cand)
                try:
                    input_ids = _build_prompt_ids(human, tokenizer, pbc, conv, point_token_len=cand)
                    stopping = KeywordsStoppingCriteria([stop_str], tokenizer, input_ids)
                    max_len = min(input_ids.shape[1] + args.max_new_tokens, 2048)
                    gen_kwargs = dict(
                        point_clouds=pc, do_sample=do_sample,
                        temperature=max(args.temperature, 1e-5),
                        top_p=args.top_p, max_length=max_len,
                        stopping_criteria=[stopping], **bcp,
                    )
                    if args.top_k > 0 and do_sample:
                        gen_kwargs["top_k"] = args.top_k
                    with torch.inference_mode():
                        out_ids = model.generate(input_ids, **gen_kwargs)
                    in_len = input_ids.shape[1]
                    pred_text = tokenizer.decode(out_ids[0, in_len:], skip_special_tokens=True).strip()
                    if cand != active_len:
                        active_len = cand
                        print(f"  [info] switched active point_token_len → {active_len}")
                    success = True
                    break
                except (ValueError, IndexError) as e:
                    msg = str(e)
                    if "point end token should follow" not in msg and "point patch tokens should be" not in msg:
                        raise
            if not success:
                print(f"  [skip] no working point_token_len for {oid} (tried {tried_lens})")
                continue

            results.append({
                "object_id": oid,
                "ground_truth": gold,
                "model_output": pred_text,
                "human_prompt": human,
                "conversation_type": row.get("conversation_type"),
            })
            n += 1

        payload = {
            "inference_prompt": "(per-sample human prompts; see results[*].human_prompt)",
            "results": results,
            "summary": {
                "num_samples": n,
                "seed": args.seed,
                "run_id": run_idx,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "top_k": args.top_k,
                "anno_path": args.anno_path,
                "model_name": model_name,
                "active_point_token_len": active_len,
            },
        }
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"[compat-eval] saved {n} predictions → {out_path}")
        print(f"[compat-eval] next: traditional_evaluator.py --results_path {out_path}")


if __name__ == "__main__":
    main()
