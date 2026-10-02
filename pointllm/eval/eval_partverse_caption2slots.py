"""
Inference + local metrics for PartVerse caption -> BCP slot prediction (Stage2).

Expects anno JSON rows with conversation_type == partverse_caption2slots and
<point> in the first human turn (the PartVerse-QA format).

Usage:
  export PYTHONPATH=/path/to/3D-PLOT-LLM
  python pointllm/eval/eval_partverse_caption2slots.py \\
    --model_name outputs/PointLLM_train_stage2/your_run \\
    --anno_path /path/to/eval_partverse_caption2slots_holdout.json \\
    --data_path /path/to/PointLLM/data/objaverse_data \\
    --bcp_pack_dir /path/to/packs_k16

Geometric IoU (default on):
  - Default `--point_patch_assign pointbert_group`: matches the
    `data_tools/partverse_qa/bcp_semantic_mapping.py` default (`--point_patch_assign pointbert_group`): inverse PointBERT
    ``Group`` lift (``group_size`` NN per patch, primary patch by closest center among claimants).
    Use this after regenerating mappings / stage2 gold with the same defaults.
  - `--point_patch_assign voronoi`: nearest `patch_xyz` center per point — only matches **legacy**
    JSON built before the mapping script switched to Group-consistent lift.
Patch-level IoU uses the 512 patch group labels directly. Use `--no_geometric_iou` to skip.

Standard Objaverse captioning/classification stays on PointLLM_brief_description_val_*;
use this script only for the PartVerse slot task.

Pipelines:
  - Pack generation: e.g. `data_tools/partition_packs/launch_bcp_stage1_v3_cls.py`
    loads `{object_id}_8192.npy`, normalizes like training, runs PointBERT `group_divider` once
    (cached) so `patch_xyz` matches `patch_feat`, then `build_k64_pack.build_object_pack` adds BCP
    `patch_r64` / `patch_r16` etc.
  - PartVerse ↔ BCP labels: `data_tools/partverse_qa/bcp_semantic_mapping.py`. It aligns mesh-sampled PartVerse points to Objaverse 8192 via ICP
    and defines which BCP groups supervise each caption; per-point group masks use the same
    point→patch rule as `--point_patch_assign` in that module (default: PointBERT Group–consistent).
"""

from __future__ import annotations

import argparse
import json
import os
import re
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from pointllm.conversation import SeparatorStyle, conv_templates
from pointllm.data.utils import preprocess_multimodal_point_cloud
from pointllm.eval.eval_objaverse import disable_torch_init
from pointllm.model import PointLLMLlamaForCausalLM
from pointllm.model.marker_context_utils import resolve_marker_context_mode
from pointllm.model.utils import KeywordsStoppingCriteria


def _masks_iou(a: np.ndarray, b: np.ndarray) -> float:
    """Boolean-mask IoU on equal-length 1D masks (same definition as `masks_iou` in bcp_semantic_mapping.py)."""
    inter = int(np.logical_and(a, b).sum())
    union = int(np.logical_or(a, b).sum())
    return 0.0 if union == 0 else float(inter) / float(union)


def _slot_strings_to_group_ids(slot_set: Set[str], part_index_base: int, k_groups: int) -> Set[int]:
    """
    Map _parse_slot_set() output to internal BCP group ids in [0, k_groups).
    <part_n> means group (n - part_index_base); plain integers are already internal ids.
    """
    out: Set[int] = set()
    for s in slot_set:
        m = re.fullmatch(r"<part_(\d+)>", s, flags=re.IGNORECASE)
        if m:
            gid = int(m.group(1)) - part_index_base
        else:
            if not s.isdigit():
                continue
            gid = int(s)
        if 0 <= gid < k_groups:
            out.add(gid)
    return out


def _point_group_ids_voronoi(pc_xyz: np.ndarray, patch_xyz: np.ndarray, patch_group: np.ndarray) -> np.ndarray:
    """One patch per point = nearest FPS center (Voronoi). Same lifting as bcp_semantic_mapping."""
    d2 = np.sum((pc_xyz[:, None, :] - patch_xyz[None, :, :]) ** 2, axis=-1)
    nn = np.argmin(d2, axis=1).astype(np.int64)
    return patch_group[nn].astype(np.int32)


def _point_group_ids_pointbert_groups(
    pc_xyz: np.ndarray,
    patch_xyz: np.ndarray,
    patch_group: np.ndarray,
    group_size: int,
) -> np.ndarray:
    """
    Align with PointBERT `Group.forward`: per patch g, neighborhood = `group_size` closest points
    to center patch_xyz[g]. Inverse map: point i -> primary patch among those that include i in their
    neighborhood, breaking ties by smallest d2 to that patch's center; if none, Voronoi fallback.
    """
    n_pts, _ = pc_xyz.shape
    n_patch = int(patch_xyz.shape[0])
    if patch_group.shape[0] != n_patch:
        raise ValueError("patch_xyz and patch_group length mismatch")
    k_nn = min(int(group_size), n_pts)
    d2 = np.sum((pc_xyz[:, None, :] - patch_xyz[None, :, :]) ** 2, axis=-1)

    owners: List[List[int]] = [[] for _ in range(n_pts)]
    for g in range(n_patch):
        col = d2[:, g]
        idx = np.argpartition(col, k_nn - 1)[:k_nn]
        for pi in idx:
            owners[int(pi)].append(g)

    primary = np.empty(n_pts, dtype=np.int64)
    for i in range(n_pts):
        cand = owners[i]
        if cand:
            primary[i] = min(cand, key=lambda g: float(d2[i, g]))
        else:
            primary[i] = int(np.argmin(d2[i]))
    return patch_group[primary].astype(np.int32)


def _geometric_group_iou(
    pc_xyz: np.ndarray,
    patch_xyz: np.ndarray,
    patch_group: np.ndarray,
    gold_groups: Set[int],
    pred_groups: Set[int],
    assign_mode: str,
    pointbert_group_size: int,
) -> float:
    """Point-level IoU over BCP group masks on points (see assign_mode in module docstring)."""
    if pc_xyz.size == 0 or patch_xyz.size == 0 or patch_group.size == 0:
        return 0.0
    if assign_mode == "voronoi":
        g_per_pt = _point_group_ids_voronoi(pc_xyz, patch_xyz, patch_group)
    elif assign_mode == "pointbert_group":
        g_per_pt = _point_group_ids_pointbert_groups(pc_xyz, patch_xyz, patch_group, pointbert_group_size)
    else:
        raise ValueError(f"Unknown assign_mode: {assign_mode}")
    gold_m = np.isin(g_per_pt, list(gold_groups)) if gold_groups else np.zeros(len(g_per_pt), dtype=bool)
    pred_m = np.isin(g_per_pt, list(pred_groups)) if pred_groups else np.zeros(len(g_per_pt), dtype=bool)
    return _masks_iou(gold_m, pred_m)


def _patch_count_iou(patch_group: np.ndarray, gold_groups: Set[int], pred_groups: Set[int]) -> float:
    """Discrete IoU over 512 patches: patch belongs to region iff its group is in the set."""
    if patch_group.size == 0:
        return 0.0
    gold_m = np.isin(patch_group, list(gold_groups)) if gold_groups else np.zeros_like(patch_group, dtype=bool)
    pred_m = np.isin(patch_group, list(pred_groups)) if pred_groups else np.zeros_like(patch_group, dtype=bool)
    return _masks_iou(gold_m, pred_m)


def _parse_slot_set(text: str) -> Tuple[Set[str], str]:
    """
    Normalize prediction or gold into a canonical set string key.
    Prefers <part_k> tokens; falls back to comma-separated integers.
    """
    t = text.strip()
    toks = re.findall(r"<part_\d+>", t, flags=re.IGNORECASE)
    if toks:
        norm = sorted({x.lower() for x in toks})
        return set(norm), ",".join(norm)
    nums = re.findall(r"\b\d+\b", t)
    if nums:
        xs = sorted({int(x) for x in nums})
        return {str(x) for x in xs}, ",".join(str(x) for x in xs)
    return set(), ""


def _gold_from_row(row: Dict[str, Any]) -> str:
    conv = row.get("conversations") or []
    if len(conv) >= 2 and conv[1].get("from") == "gpt":
        return str(conv[1].get("value", ""))
    meta = row.get("meta") or {}
    gids = meta.get("bcp_group_ids")
    if isinstance(gids, list) and gids:
        return ",".join(str(int(x)) for x in sorted(set(gids)))
    return ""


def _load_pc_numpy(data_path: str, object_id: str, pointnum: int) -> np.ndarray:
    from pointllm.data.utils import pc_norm

    fn = f"{object_id}_{pointnum}.npy"
    pc = np.load(os.path.join(data_path, fn))
    return pc_norm(pc).astype(np.float32)


def _load_pc_torch(pc_np: np.ndarray, use_color: bool) -> torch.Tensor:
    pc = pc_np[:, :3] if not use_color else pc_np
    return torch.from_numpy(np.ascontiguousarray(pc))


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
    parser.add_argument(
        "--part_index_base",
        type=int,
        default=None,
        help="If set, overrides row meta['part_index_base'] when mapping <part_k> -> group id.",
    )
    parser.add_argument(
        "--no_geometric_iou",
        action="store_true",
        help="Skip point/patch geometric IoU (faster; token Jaccard still computed).",
    )
    parser.add_argument(
        "--point_patch_assign",
        type=str,
        choices=("pointbert_group", "voronoi"),
        default="pointbert_group",
        help="How each point gets a BCP group for geometric IoU (see module docstring).",
    )
    parser.add_argument(
        "--pointbert_group_size",
        type=int,
        default=32,
        help="Must match the Point-BERT group size used when building the packs (32).",
    )
    parser.add_argument("--pointnum", type=int, default=8192)
    parser.add_argument("--use_color", action="store_true", default=True)
    parser.add_argument("--batch_size", type=int, default=1, help="Must stay 1 (variable prompts).")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    args = parser.parse_args()
    if args.batch_size != 1:
        raise SystemExit("Only batch_size=1 supported (per-row human prompts).")
    do_geo = not bool(args.no_geometric_iou)

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

    rows = [r for r in rows if r.get("conversation_type") == "partverse_caption2slots"]
    out_dir = os.path.join(model_name, "evaluation")
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(args.anno_path))[0]
    out_file = f"{base}_partverse_caption2slots_pred.json"
    out_path = os.path.join(out_dir, out_file)

    results: List[Dict[str, Any]] = []
    j_sum = 0.0
    ex = 0
    n = 0
    geo_point_sum = 0.0
    geo_patch_sum = 0.0
    geo_n = 0
    active_point_token_len = int(pbc.get("point_token_len", 512))

    for row in tqdm(rows):
        oid = row["object_id"]
        human = row["conversations"][0]["value"]
        gold_s = _gold_from_row(row)
        gold_set, _ = _parse_slot_set(gold_s)

        pc_np = _load_pc_numpy(args.data_path, oid, args.pointnum)
        pc = _load_pc_torch(pc_np, args.use_color).unsqueeze(0).cuda().to(model.dtype)

        input_ids = _build_prompt_ids_with_point_token_len(
            human, tokenizer, pbc, conv, active_point_token_len
        )
        stopping = KeywordsStoppingCriteria([stop_str], tokenizer, input_ids)

        bcp_patch = bcp_cls = bcp_gid = None
        bcp_adj = bcp_stats = None
        g_np: Optional[np.ndarray] = None
        patch_xyz_np: Optional[np.ndarray] = None
        if args.bcp_pack_dir:
            pack_path = os.path.join(args.bcp_pack_dir, f"{oid}.pack.npz")
            if os.path.isfile(pack_path):
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
                    g_np = np.asarray(g, dtype=np.int32)
                    if do_geo and "patch_xyz" in pack:
                        patch_xyz_np = np.asarray(pack["patch_xyz"], dtype=np.float32)
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
        pred_text = tokenizer.decode(out_ids[0, in_len:], skip_special_tokens=False).strip()

        pred_set, _ = _parse_slot_set(pred_text)
        if gold_set and pred_set:
            inter = len(gold_set & pred_set)
            union = len(gold_set | pred_set)
            j = inter / union if union else 0.0
        else:
            j = 0.0
        j_sum += j
        if pred_set == gold_set and gold_set:
            ex += 1
        n += 1

        meta = row.get("meta") or {}
        k_groups = int(pbc.get("bcp_num_groups", 16))
        pib = args.part_index_base if args.part_index_base is not None else int(meta.get("part_index_base", 0))
        gold_gid = _slot_strings_to_group_ids(gold_set, pib, k_groups)
        pred_gid = _slot_strings_to_group_ids(pred_set, pib, k_groups)

        geo_point: Optional[float] = None
        geo_patch: Optional[float] = None
        if (
            do_geo
            and g_np is not None
            and patch_xyz_np is not None
            and gold_gid
        ):
            pc_xyz = np.ascontiguousarray(pc_np[:, :3])
            geo_point = _geometric_group_iou(
                pc_xyz,
                patch_xyz_np,
                g_np,
                gold_gid,
                pred_gid,
                args.point_patch_assign,
                args.pointbert_group_size,
            )
            geo_patch = _patch_count_iou(g_np, gold_gid, pred_gid)
            geo_point_sum += geo_point
            geo_patch_sum += geo_patch
            geo_n += 1

        row_out: Dict[str, Any] = {
            "object_id": oid,
            "gold": gold_s,
            "prediction": pred_text,
            "jaccard_tokens": j,
            "exact_set_match": bool(pred_set == gold_set and gold_set),
        }
        if do_geo:
            row_out["geometric_iou_points"] = geo_point
            row_out["geometric_iou_patches"] = geo_patch
            row_out["point_patch_assign"] = args.point_patch_assign
        results.append(row_out)

    summary = {
        "num_samples": n,
        "mean_jaccard": (j_sum / n) if n else 0.0,
        "exact_set_match_rate": (ex / n) if n else 0.0,
        "mean_geometric_iou_points": (geo_point_sum / geo_n) if geo_n else None,
        "mean_geometric_iou_patches": (geo_patch_sum / geo_n) if geo_n else None,
        "num_geometric_iou_samples": geo_n,
        "point_patch_assign": args.point_patch_assign,
        "pointbert_group_size": args.pointbert_group_size,
        "anno_path": args.anno_path,
        "model_name": model_name,
    }
    payload = {"summary": summary, "results": results}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"Saved details to {out_path}")


if __name__ == "__main__":
    main()
