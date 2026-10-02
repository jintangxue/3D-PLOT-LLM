"""Default absolute paths (edit here or override via CLI)."""

import os
PARTVERSE_ROOT = os.environ.get("PARTVERSE_ROOT", "data/partverse")
PARTVERSE_CAPTIONS_JSON = os.path.join(PARTVERSE_ROOT, "text_captions.json")
PARTVERSE_POINT_CACHE = os.environ.get("PARTVERSE_CACHE", os.path.join(PARTVERSE_ROOT, "cache_global_rgb_nn_partid"))
OBJAVERSE_8192_DIR = os.path.join(os.environ.get("POINTLLM_DATA", "data/pointllm"), "objaverse_data")
BCP_PACK_DIR = os.environ.get("BCP_PACK_DIR", "data/packs_k16")
OUTPUT_DIR = os.environ.get("PARTVERSE_QA_DIR", "data/partverse_qa")
