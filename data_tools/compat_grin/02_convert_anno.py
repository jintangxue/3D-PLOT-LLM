#!/usr/bin/env python
"""
Step 2: Convert 3DCoMPaT-GrIn JSONs to PointLLM-compatible conversation format.

Input format (3DCoMPaT-GrIn):
  [{"id": "05_0b1_0",
    "conversations": [{"from": "human", "value": "<point>\\n..."},
                      {"from": "gpt",  "value": "...with <p>seat frame</p>[SEG]..."}],
    "parts": ["seat frame", ...]}, ...]

Output format (PointLLM-compatible):
  [{"object_id": "05_0b1_0",
    "conversation_type": "compat_papgd_<split>",
    "conversations": [{"from": "human", "value": "<point>\\n..."},
                      {"from": "gpt",  "value": "...with seat frame..."}]}]

Key transforms:
  1. `id` → `object_id`
  2. Strip `<p>...</p>` markup and `[SEG]` tokens from gpt outputs
     (We do language-only eval — no segmentation head. Markup is only used
      by Kestrel for downstream Mask3D.)
  3. Add `conversation_type` for filtering downstream
  4. Drop `parts` field (kept in `meta` for reference if needed)

Usage:
  python 02_convert_anno.py \\
      --input  /tmp/.../valid_multiple_parts.json \\
      --output $COMPAT_DATA/anno/compat_valid_multiple_parts.json \\
      --conversation_type compat_papgd_multi
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


# Match <p>...</p> with optional [SEG] token after — keep only inner text
# Also handles whitespace variants like `<p>cover holder</p>[SEG]` and `<p> hour hand</p>[SEG]`
PAR_RE = re.compile(r"<p>\s*(.*?)\s*</p>(?:\[SEG\])?", re.IGNORECASE)
SEG_TOKEN_RE = re.compile(r"\[SEG\]", re.IGNORECASE)


def strip_seg_markup(text: str) -> str:
    """Replace `<p>X</p>[SEG]` with `X`, and remove any standalone `[SEG]` tokens.
    Applies iteratively to handle nested tags like `<p><p>X</p>[SEG]</p>[SEG]`."""
    prev = None
    out = text
    while out != prev:
        prev = out
        out = PAR_RE.sub(r"\1", out)
        out = SEG_TOKEN_RE.sub("", out)
    return out


def convert_row(row: dict, conversation_type: str) -> dict:
    new_convs = []
    for turn in row.get("conversations", []):
        v = turn.get("value", "")
        # Strip markup from gpt outputs; also strip from human (rare, but safe)
        v_clean = strip_seg_markup(v)
        new_convs.append({"from": turn["from"], "value": v_clean})
    return {
        "object_id": row["id"],
        "conversation_type": conversation_type,
        "conversations": new_convs,
        "meta": {
            "parts": row.get("parts", []),
            "source_id": row["id"],
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="3DCoMPaT-GrIn JSON file")
    ap.add_argument("--output", required=True, help="Output PointLLM-format JSON")
    ap.add_argument("--conversation_type", required=True,
                    help="Tag for filtering, e.g. compat_papgd_multi, compat_papgd_single, compat_papgd_embodied")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    with open(args.input, "r") as f:
        rows = json.load(f)
    print(f"[02] Loaded {len(rows)} rows from {args.input}")

    if args.limit > 0:
        rows = rows[: args.limit]

    converted = [convert_row(r, args.conversation_type) for r in rows]
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(converted, f, indent=2)

    print(f"[02] Wrote {len(converted)} rows → {args.output}")

    # Show a sample
    print("\n[02] Sample (first row):")
    print(json.dumps(converted[0], indent=2)[:1200])


if __name__ == "__main__":
    main()
