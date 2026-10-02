#!/usr/bin/env python3
"""
Build PointLLM-compatible Stage2 JSON from:
  - mappings JSONL (bcp_semantic_mapping.build)
  - PartVerse text_captions.json

Rule-based templates; K=16 default; short caption primary, optional long duplicate;
optional slots2caption (gpt = short).

Default GPT labels for caption2slots use special tokens <part_0>,... (see --slot_answer_format).

Usage:
  python -m data_tools.partverse_qa.build_stage2_part_json \\
    --out outputs/stage2_part_k16_iou04_parttok.json \\
    --min_union_iou 0.4 --long_ratio 1.0
  # Legacy numeric answers: --slot_answer_format numeric
  # If vocab is <part_1>..<part_16>: --part_index_base 1

Training: add conversation_types e.g. partverse_caption2slots, partverse_slots2caption
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from .default_paths import OUTPUT_DIR, PARTVERSE_CAPTIONS_JSON
except ImportError:
    from default_paths import OUTPUT_DIR, PARTVERSE_CAPTIONS_JSON  # type: ignore


def _part_literal(group_id: int, index_base: int) -> str:
    """Slot token <part_{g}> with g = group_id + index_base (default base 0)."""
    return f"<part_{group_id + index_base}>"


def _slots_answer_numeric(group_ids: List[int]) -> str:
    return ",".join(str(x) for x in sorted(group_ids))


def _slots_answer_part_tokens(group_ids: List[int], index_base: int) -> str:
    return ",".join(_part_literal(g, index_base) for g in sorted(group_ids))


def _slots_phrase_numeric(group_ids: List[int]) -> str:
    s = sorted(group_ids)
    if len(s) == 1:
        return str(s[0])
    if len(s) == 2:
        return f"{s[0]} and {s[1]}"
    return ", ".join(str(x) for x in s[:-1]) + f", and {s[-1]}"


def _slots_phrase_part_tokens(group_ids: List[int], index_base: int) -> str:
    s = sorted(group_ids)
    toks = [_part_literal(g, index_base) for g in s]
    if len(toks) == 1:
        return toks[0]
    if len(toks) == 2:
        return f"{toks[0]} and {toks[1]}"
    return ", ".join(toks[:-1]) + f", and {toks[-1]}"


def _human_caption2slots(
    caption: str, bcp_k: int, answer_format: str, index_base: int, prompt_style: str
) -> str:
    hi = bcp_k - 1
    cap = caption.strip()
    if prompt_style == "short":
        if answer_format == "part_token":
            return (
                f"<point>\nText: \"{cap}\" "
                f"Output only matching <part_n> tokens, comma-separated."
            )
        return (
            f"<point>\nText: \"{cap}\" "
            f"Output only matching region indices 0-{hi}, comma-separated."
        )

    if answer_format == "part_token":
        lo_tok = _part_literal(0, index_base)
        hi_tok = _part_literal(hi, index_base)
        return (
            f"<point>\nThe following text describes ONE component of this 3D object:\n"
            f"\"{cap}\"\n"
            f"Which pseudo-part tokens ({lo_tok} through {hi_tok}) best correspond to this component? "
            f"Answer with comma-separated tokens only (e.g. \"{lo_tok},{hi_tok}\"), with no other text."
        )
    return (
        f"<point>\nThe following text describes ONE component of this 3D object:\n"
        f"\"{cap}\"\n"
        f"Which pseudo-part region indices (0-{hi}) best correspond to this component? "
        f"Answer with comma-separated integers only, with no other text."
    )


def _human_slots2caption(
    group_ids: List[int], bcp_k: int, answer_format: str, index_base: int, prompt_style: str
) -> str:
    _ = bcp_k
    if answer_format == "part_token":
        sp = _slots_phrase_part_tokens(group_ids, index_base)
    else:
        sp = _slots_phrase_numeric(group_ids)
    if prompt_style == "short":
        return f"<point>\nDescribe region {sp} in one short sentence."
    return (
        f"<point>\nDescribe in one concise sentence the region formed together by "
        f"pseudo-part slot(s) {sp} of this 3D object."
    )


def _get_caption(
    caps_obj: Optional[Dict[str, Any]], view_key: str, source: str
) -> Optional[str]:
    if caps_obj is None:
        return None
    entry = caps_obj.get(view_key)
    if not isinstance(entry, list) or len(entry) < 1:
        return None
    if source == "short":
        t = entry[0]
    else:
        t = entry[1] if len(entry) > 1 else entry[0]
    if not isinstance(t, str) or not t.strip():
        return None
    return t.strip()


def load_mappings(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_entries_for_object(
    row: Dict[str, Any],
    caps_obj: Optional[Dict[str, Any]],
    bcp_k: int,
    min_union_iou: float,
    long_ratio: float,
    include_slots2caption: bool,
    rng: random.Random,
    slot_answer_format: str,
    part_index_base: int,
    prompt_style: str,
) -> List[Dict[str, Any]]:
    oid = row["object_id"]
    k = int(row.get("k", bcp_k))
    if k != bcp_k:
        return []

    out: List[Dict[str, Any]] = []
    for ps in row.get("per_semantic", []):
        gids = ps.get("group_ids") or []
        if not gids:
            continue
        ui = float(ps.get("union_iou", 0.0))
        if ui < min_union_iou:
            continue
        sem = int(ps["semantic_id"])
        view_key = str(sem)

        short_cap = _get_caption(caps_obj, view_key, "short")
        long_cap = _get_caption(caps_obj, view_key, "long")
        if short_cap is None:
            continue

        base_meta = {
            "bcp_k": bcp_k,
            "semantic_part_id": sem,
            "bcp_group_ids": sorted(int(x) for x in gids),
            "union_iou": ui,
            "caption_view_key": view_key,
            "slot_answer_format": slot_answer_format,
            "part_index_base": part_index_base,
        }
        if "point_patch_assign" in row:
            base_meta["point_patch_assign"] = row["point_patch_assign"]
        if "pointbert_group_size" in row:
            base_meta["pointbert_group_size"] = row["pointbert_group_size"]
        if slot_answer_format == "part_token":
            ans = _slots_answer_part_tokens(gids, part_index_base)
        else:
            ans = _slots_answer_numeric(gids)

        # caption -> slots (short)
        out.append(
            {
                "object_id": oid,
                "conversation_type": "partverse_caption2slots",
                "conversations": [
                    {
                        "from": "human",
                        "value": _human_caption2slots(
                            short_cap, bcp_k, slot_answer_format, part_index_base, prompt_style
                        ),
                    },
                    {"from": "gpt", "value": ans},
                ],
                "meta": {**base_meta, "caption_source": "short", "task": "caption2slots"},
            }
        )

        # caption -> slots (long) — same label, optional by ratio
        if long_cap is not None and long_ratio > 0 and (long_ratio >= 1.0 or rng.random() < long_ratio):
            out.append(
                {
                    "object_id": oid,
                    "conversation_type": "partverse_caption2slots",
                    "conversations": [
                        {
                            "from": "human",
                            "value": _human_caption2slots(
                                long_cap, bcp_k, slot_answer_format, part_index_base, prompt_style
                            ),
                        },
                        {"from": "gpt", "value": ans},
                    ],
                    "meta": {**base_meta, "caption_source": "long", "task": "caption2slots"},
                }
            )

        if include_slots2caption:
            out.append(
                {
                    "object_id": oid,
                    "conversation_type": "partverse_slots2caption",
                    "conversations": [
                        {
                            "from": "human",
                            "value": _human_slots2caption(
                                gids, bcp_k, slot_answer_format, part_index_base, prompt_style
                            ),
                        },
                        {"from": "gpt", "value": short_cap},
                    ],
                    "meta": {**base_meta, "caption_source": "short", "task": "slots2caption"},
                }
            )

    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Build Stage2 part JSON for PointLLM")
    p.add_argument(
        "--mappings",
        type=str,
        default=str(Path(OUTPUT_DIR) / "mappings_k16_full_seed42.jsonl"),
        help="Input mappings JSONL",
    )
    p.add_argument(
        "--captions",
        type=str,
        default=PARTVERSE_CAPTIONS_JSON,
        help="PartVerse text_captions.json",
    )
    p.add_argument(
        "--out",
        type=str,
        required=True,
        help="Output JSON array path (PointLLM anno format)",
    )
    p.add_argument("--bcp_k", type=int, default=16)
    p.add_argument(
        "--min_union_iou",
        type=float,
        default=0.4,
        help="Canonical full pool uses 0.4; stricter subsets via filter_stage2_part_json.py or --filter_min_union_iou in build_splits.py.",
    )
    p.add_argument(
        "--long_ratio",
        type=float,
        default=1.0,
        help="Probability (or 1.0=always) to add caption2slots with long caption",
    )
    p.add_argument(
        "--no_slots2caption",
        action="store_true",
        help="Disable slots -> caption tasks",
    )
    p.add_argument(
        "--slot_answer_format",
        type=str,
        choices=("part_token", "numeric"),
        default="part_token",
        help="GPT answer for caption2slots: part_token=<part_0>,<part_3> (default, used by the released model); numeric=0,3",
    )
    p.add_argument(
        "--part_index_base",
        type=int,
        default=0,
        help="Token name is <part_{group_id + base}>. The released model uses base=0 (<part_0>..). Use 1 if your vocabulary is <part_1>..<part_K>.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_objects", type=int, default=0, help="0 = all")
    p.add_argument(
        "--prompt_style",
        type=str,
        choices=("verbose", "short"),
        default="verbose",
        help="Human prompt template style. Use 'short' to reduce prompt length shift vs PointLLM 70K.",
    )
    args = p.parse_args()

    rng = random.Random(args.seed)
    cap_path = Path(args.captions)
    print(f"Loading captions from {cap_path} ...", file=sys.stderr)
    with open(cap_path, "r", encoding="utf-8") as f:
        all_caps: Dict[str, Any] = json.load(f)

    rows = load_mappings(Path(args.mappings))
    if args.max_objects > 0:
        rows = rows[: args.max_objects]

    entries: List[Dict[str, Any]] = []
    missing_caps_objects = sum(1 for row in rows if row["object_id"] not in all_caps)
    for row in rows:
        oid = row["object_id"]
        caps_obj = all_caps.get(oid)
        entries.extend(
            build_entries_for_object(
                row,
                caps_obj,
                args.bcp_k,
                args.min_union_iou,
                args.long_ratio,
                not args.no_slots2caption,
                rng,
                args.slot_answer_format,
                args.part_index_base,
                args.prompt_style,
            )
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)

    print(
        f"Wrote {len(entries)} conversations to {out_path} "
        f"(mappings rows {len(rows)}, objects without text_captions entry: {missing_caps_objects})",
        file=sys.stderr,
    )

    # Count by type
    from collections import Counter

    c = Counter(e["conversation_type"] for e in entries)
    for k in sorted(c.keys()):
        print(f"  {k}: {c[k]}", file=sys.stderr)


if __name__ == "__main__":
    main()
