"""
Inference + local metrics for PartVerse slots -> short caption (Stage2).

Expects anno JSON rows with conversation_type == partverse_slots2caption.

Usage:
  export PYTHONPATH=/path/to/3D-PLOT-LLM
  python pointllm/eval/eval_partverse_slots2caption.py \\
    --model_name outputs/PointLLM_train_stage2/your_run \\
    --anno_path /path/to/eval_partverse_slots2caption_*.json \\
    --data_path /path/to/PointLLM/data/objaverse_data \\
    --bcp_pack_dir /path/to/packs_k16
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from typing import Any, Dict, List

import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from pointllm.conversation import SeparatorStyle, conv_templates
from pointllm.data.utils import preprocess_multimodal_point_cloud
from pointllm.eval.eval_objaverse import disable_torch_init
from pointllm.model import PointLLMLlamaForCausalLM
from pointllm.model.marker_context_utils import resolve_marker_context_mode
from pointllm.model.utils import KeywordsStoppingCriteria


def _norm_text(s: str) -> str:
    return " ".join(s.strip().lower().split())


def _word_f1(pred: str, gold: str) -> float:
    pw = _norm_text(pred).split()
    gw = _norm_text(gold).split()
    if not gw:
        return 0.0
    if not pw:
        return 0.0
    pc, gc = Counter(pw), Counter(gw)
    inter = sum((pc & gc).values())
    prec = inter / len(pw)
    rec = inter / len(gw)
    return (2.0 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0


def _gold_caption(row: Dict[str, Any]) -> str:
    conv = row.get("conversations") or []
    if len(conv) >= 2 and conv[1].get("from") == "gpt":
        return str(conv[1].get("value", ""))
    return ""


def _load_pc(data_path: str, object_id: str, pointnum: int, use_color: bool):
    import numpy as np

    from pointllm.data.utils import pc_norm

    fn = f"{object_id}_{pointnum}.npy"
    pc = np.load(os.path.join(data_path, fn))
    pc = pc_norm(pc)
    if not use_color:
        pc = pc[:, :3]
    return torch.from_numpy(pc.astype("float32"))


def _build_prompt_ids(
    human_raw: str,
    tokenizer,
    point_backbone_config: Dict[str, Any],
    conv,
) -> torch.Tensor:
    sources = [
        [
            {"from": "human", "value": human_raw},
            {"from": "gpt", "value": "."},
        ]
    ]
    preprocess_multimodal_point_cloud(sources, point_backbone_config)
    human_expanded = sources[0][0]["value"]

    conv = conv.copy()
    conv.messages = []
    conv.append_message(conv.roles[0], human_expanded)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    inputs = tokenizer([prompt], return_tensors="pt")
    return inputs.input_ids.cuda()


def _build_prompt_ids_with_point_token_len(
    human_raw: str,
    tokenizer,
    point_backbone_config: Dict[str, Any],
    conv,
    point_token_len: int,
) -> torch.Tensor:
    pbc = dict(point_backbone_config)
    pbc["point_token_len"] = int(point_token_len)
    return _build_prompt_ids(human_raw, tokenizer, pbc, conv)


def _fallback_point_token_lens(point_backbone_config: Dict[str, Any]) -> List[int]:
    base = int(point_backbone_config.get("point_token_len", 512))
    k = int(point_backbone_config.get("bcp_num_groups", 16))
    candidates = [
        base,
        1 + k + 512,      # vocab-token path
        1 + 2 * k + 512,  # vocab-token + graph mean tokens
        512,
    ]
    out: List[int] = []
    for x in candidates:
        if x > 0 and x not in out:
            out.append(x)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--tokenizer_name", type=str, default=None)
    parser.add_argument("--anno_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--bcp_pack_dir", type=str, default=None)
    parser.add_argument("--pointnum", type=int, default=8192)
    parser.add_argument("--use_color", action="store_true", default=True)
    parser.add_argument("--batch_size", type=int, default=1, help="Must stay 1 (variable prompts).")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=0, help="0 disables top-k. Used only when temperature>0.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Optional. If set, seeds torch/cuda for sampling reproducibility.")
    parser.add_argument("--run_id", type=int, default=None,
                        help="Optional. If set, output filename gets a _run{run_id} suffix.")
    args = parser.parse_args()
    if args.batch_size != 1:
        raise SystemExit("Only batch_size=1 supported (per-row human prompts).")

    if args.seed is not None:
        import random

        import numpy as _np
        random.seed(args.seed)
        _np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    disable_torch_init()
    model_name = os.path.expanduser(args.model_name)
    tokenizer_name = args.tokenizer_name or model_name
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=False)
    model = PointLLMLlamaForCausalLM.from_pretrained(
        model_name, low_cpu_mem_usage=False, use_cache=True, torch_dtype=torch.bfloat16
    ).cuda()
    model.initialize_tokenizer_point_backbone_config_wo_embedding(tokenizer)

    conv_template = conv_templates["vicuna_v1_1"]
    conv = conv_template.copy()
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2

    pbc = model.get_model().point_backbone_config

    with open(args.anno_path, "r", encoding="utf-8") as f:
        rows: List[Dict[str, Any]] = json.load(f)

    rows = [r for r in rows if r.get("conversation_type") == "partverse_slots2caption"]
    out_dir = os.path.join(model_name, "evaluation")
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(args.anno_path))[0]
    if args.run_id is not None:
        out_file = f"{base}_partverse_slots2caption_pred_run{args.run_id}.json"
    else:
        out_file = f"{base}_partverse_slots2caption_pred.json"
    out_path = os.path.join(out_dir, out_file)

    results: List[Dict[str, Any]] = []
    f1_sum = 0.0
    ex = 0
    n = 0
    active_point_token_len = int(pbc.get("point_token_len", 512))

    for row in tqdm(rows):
        oid = row["object_id"]
        human = row["conversations"][0]["value"]
        gold_s = _gold_caption(row)

        pc = _load_pc(args.data_path, oid, args.pointnum, args.use_color).unsqueeze(0).cuda().to(model.dtype)

        input_ids = _build_prompt_ids_with_point_token_len(
            human, tokenizer, pbc, conv, active_point_token_len
        )
        stopping = KeywordsStoppingCriteria([stop_str], tokenizer, input_ids)

        bcp_patch = bcp_cls = bcp_gid = None
        bcp_adj = bcp_stats = None
        if args.bcp_pack_dir:
            pack_path = os.path.join(args.bcp_pack_dir, f"{oid}.pack.npz")
            if os.path.isfile(pack_path):
                import numpy as np

                with np.load(pack_path) as pack:
                    patch_np = pack["patch_feat"].astype("float32")
                    bcp_patch = torch.from_numpy(patch_np).unsqueeze(0).cuda().to(model.dtype)
                    if "cls_feat" in pack:
                        cls_np = pack["cls_feat"].astype("float32")
                    else:
                        if pbc.get("bcp_part_vocab_token", False) or pbc.get("bcp_part_interleave", False):
                            raise RuntimeError(
                                f"Missing cls_feat in pack for oid={oid}: {pack_path}. "
                                "This pack is incompatible with vocab/interleave checkpoint layout."
                            )
                        cls_np = None
                    if cls_np is not None:
                        bcp_cls = torch.from_numpy(cls_np).unsqueeze(0).cuda().to(model.dtype)
                    k = pbc.get("bcp_num_groups", 16)
                    key_r = f"patch_r{k}"
                    key_map = f"map64_to_{k}"
                    if key_r in pack:
                        g = pack[key_r].astype("int32")
                    elif "patch_r64" in pack and key_map in pack:
                        g = pack[key_map][pack["patch_r64"]].astype("int32")
                    else:
                        g = np.zeros(512, dtype="int32")
                    bcp_gid = torch.from_numpy(g).unsqueeze(0).cuda()
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
                        bcp_adj = region_adj.unsqueeze(0).cuda()
                        bcp_stats = region_stats.unsqueeze(0).cuda()

        do_sample = args.temperature > 0
        max_len = min(input_ids.shape[1] + args.max_new_tokens, 2048)
        gen_kwargs = dict(
            point_clouds=pc,
            bcp_patch_feat=bcp_patch,
            bcp_cls_feat=bcp_cls,
            bcp_group_ids=bcp_gid,
            bcp_region_adj=bcp_adj,
            bcp_region_stats=bcp_stats,
            do_sample=do_sample,
            temperature=max(args.temperature, 1e-5),
            top_p=args.top_p,
            max_length=max_len,
            stopping_criteria=[stopping],
        )
        if args.top_k > 0:
            gen_kwargs["top_k"] = args.top_k
        try:
            with torch.inference_mode():
                out_ids = model.generate(input_ids, **gen_kwargs)
        except (ValueError, IndexError) as e:
            msg = str(e)
            if "point end token should follow the point start token" not in msg:
                raise
            out_ids = None
            for cand_len in _fallback_point_token_lens(pbc):
                if cand_len == active_point_token_len:
                    continue
                try:
                    input_ids_try = _build_prompt_ids_with_point_token_len(human, tokenizer, pbc, conv, cand_len)
                    stopping_try = KeywordsStoppingCriteria([stop_str], tokenizer, input_ids_try)
                    max_len_try = min(input_ids_try.shape[1] + args.max_new_tokens, 2048)
                    gen_kwargs_try = dict(gen_kwargs)
                    gen_kwargs_try["stopping_criteria"] = [stopping_try]
                    gen_kwargs_try["max_length"] = max_len_try
                    with torch.inference_mode():
                        out_ids = model.generate(input_ids_try, **gen_kwargs_try)
                    input_ids = input_ids_try
                    active_point_token_len = cand_len
                    print(f"[INFO] Switched point_token_len to {active_point_token_len} for subsequent rows.")
                    break
                except (ValueError, IndexError) as e2:
                    if "point end token should follow the point start token" not in str(e2):
                        raise
            if out_ids is None:
                raise
        in_len = input_ids.shape[1]
        pred_text = tokenizer.decode(out_ids[0, in_len:], skip_special_tokens=True).strip()

        wf = _word_f1(pred_text, gold_s) if gold_s else 0.0
        f1_sum += wf
        if gold_s and _norm_text(pred_text) == _norm_text(gold_s):
            ex += 1
        n += 1

        results.append(
            {
                "object_id": oid,
                "gold": gold_s,
                "prediction": pred_text,
                "word_f1": wf,
                "exact_norm_match": bool(gold_s and _norm_text(pred_text) == _norm_text(gold_s)),
            }
        )

    summary = {
        "num_samples": n,
        "mean_word_f1": (f1_sum / n) if n else 0.0,
        "exact_norm_match_rate": (ex / n) if n else 0.0,
        "anno_path": args.anno_path,
        "model_name": model_name,
        "seed": args.seed,
        "run_id": args.run_id,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
    }
    payload = {"summary": summary, "results": results}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"Saved details to {out_path}")


if __name__ == "__main__":
    main()
