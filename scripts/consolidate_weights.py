import os
import sys
from deepspeed.utils.zero_to_fp32 import convert_zero_checkpoint_to_fp32_state_dict

def consolidate(ckpt_dir, output_file):
    print(f"Consolidating shards from {ckpt_dir} -> {output_file}...")
    import shutil
    try:
        if os.path.isdir(output_file):
            print(f"Removing existing directory: {output_file}")
            shutil.rmtree(output_file)
        elif os.path.exists(output_file):
            print(f"Removing existing file: {output_file}")
            os.remove(output_file)
        convert_zero_checkpoint_to_fp32_state_dict(ckpt_dir, output_file)
        print("Done!")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python scripts/consolidate_weights.py <ckpt_dir> <output_bin>")
        sys.exit(1)
    consolidate(sys.argv[1], sys.argv[2])
