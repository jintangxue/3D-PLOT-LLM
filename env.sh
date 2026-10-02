#!/bin/bash
# Central path configuration for 3D-PLOT-LLM. Every training / evaluation script sources this file.
# Override any variable in your shell (export VAR=...) or edit the defaults below.

# Repository root (auto-detected from this file's location).
export PLOT_ROOT="${PLOT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
export PYTHONPATH="${PLOT_ROOT}${PYTHONPATH:+:$PYTHONPATH}"

# Python interpreters. PLOT_PYTHON runs training / inference / GPT judge; PLOT_PYTHON_EVAL runs the
# traditional caption metrics (needs sentence-transformers). They may be the same interpreter.
export PLOT_PYTHON="${PLOT_PYTHON:-python}"
export PLOT_PYTHON_EVAL="${PLOT_PYTHON_EVAL:-$PLOT_PYTHON}"

# Data roots (see README "Data preparation").
export POINTLLM_DATA="${POINTLLM_DATA:-$PLOT_ROOT/data/pointllm}"   # contains objaverse_data/ (the 8192-point .npy files) and anno_data/
export POINTLLM_CKPT_DIR="${POINTLLM_CKPT_DIR:-$PLOT_ROOT/checkpoints}"       # PointLLM_7B_v1.1_init/, PointLLM_7B_v1.2/
export POINT_BERT_CKPT="${POINT_BERT_CKPT:-$POINTLLM_CKPT_DIR/PointLLM_7B_v1.1_init/point_bert_v1.2.pt}"
export BCP_PACK_DIR="${BCP_PACK_DIR:-$PLOT_ROOT/data/packs_k16}"                 # K=16 partition packs (packs_k16/ from the dataset tars)
export PARTVERSE_QA_DIR="${PARTVERSE_QA_DIR:-$PLOT_ROOT/data/partverse_qa}"   # PartVerse-QA jsons (HF dataset)
export PARTVERSE_ROOT="${PARTVERSE_ROOT:-$PLOT_ROOT/data/partverse}"                 # PartVerse download (only to rebuild the dataset)
export PARTVERSE_CACHE="${PARTVERSE_CACHE:-$PARTVERSE_ROOT/cache_global_rgb_nn_partid}"  # per-point part ids
export COMPAT_DATA="${COMPAT_DATA:-$PLOT_ROOT/data/3dcompat}"                 # 3DCoMPaT / GrIn
