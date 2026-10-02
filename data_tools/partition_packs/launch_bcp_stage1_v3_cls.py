"""
Offline K=16 partition-pack generator.

Runs the frozen Point-BERT encoder once per object and calls the group divider a single time, so the
stored patch centers and patch features come from the same FPS sample; patch_feat and cls_feat are
stored in float32.
"""
import os
import sys
import argparse
import json
import traceback
import time
import numpy as np
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from build_k64_pack import Config, build_object_pack

# ────────────────────────────────────────────────────────────────────────────────
# Monkey-patch helper: make group_divider deterministic across two calls per batch
# ────────────────────────────────────────────────────────────────────────────────

class _CachedGroupDivider(torch.nn.Module):
    """
    Wraps a Group object for ONE batch. The first call executes the real
    group_divider and caches the result; the second call (inside model.forward)
    returns the cache. This guarantees that the same FPS sample is used for
    both extracting centers AND computing patch embeddings.
    """
    def __init__(self, real_divider):
        super().__init__()
        self._real = real_divider
        self._cache = None

    def forward(self, pts):
        if self._cache is None:
            self._cache = self._real(pts)
        return self._cache

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    # Forward attribute access to the real object so nothing else breaks
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self._real, name)



# ────────────────────────────────────────────────────────────────────────────────
# Dataset
# ────────────────────────────────────────────────────────────────────────────────

class PointCloudDataset(Dataset):
    def __init__(self, file_paths):
        self.file_paths = file_paths

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        fpath = self.file_paths[idx]
        obj_id = Path(fpath).stem.replace("_8192", "")

        try:
            points_raw = np.load(fpath)
            points_6d = np.zeros((8192, 6), dtype=np.float32)
            if points_raw.shape[1] == 3:
                points_6d[:, :3] = points_raw
            else:
                points_6d[:, :6] = points_raw[:, :6]

            # Normalise (same as online pc_norm used during training)
            centroid = points_6d[:, :3].mean(axis=0, keepdims=True)
            points_6d[:, :3] -= centroid
            m = np.max(np.sqrt(np.sum(points_6d[:, :3] ** 2, axis=1, keepdims=True)))
            points_6d[:, :3] /= max(m, 1e-6)

            return points_6d, obj_id, fpath
        except Exception as e:
            return None, obj_id, str(e)


def custom_collate(batch):
    batch = [b for b in batch if b[0] is not None]
    if len(batch) == 0:
        return None, [], []
    points = torch.tensor(np.stack([b[0] for b in batch]), dtype=torch.float32)
    obj_ids = [b[1] for b in batch]
    fpaths = [b[2] for b in batch]
    return points, obj_ids, fpaths


# ────────────────────────────────────────────────────────────────────────────────
# Worker
# ────────────────────────────────────────────────────────────────────────────────

def process_worker(rank, args, files_to_process):
    device = torch.device("cuda")

    # Load model from the OURS path (same as v2)
    sys.path.append(os.environ.get("PLOT_ROOT", os.getcwd()))
    from pointllm.model.pointbert.point_encoder import PointTransformer

    config = Config()
    model = PointTransformer(config, use_max_pool=False)
    if os.path.exists(args.ckpt):
        model.load_checkpoint(args.ckpt)
    else:
        print(f"WARN Worker {rank}: Checkpoint {args.ckpt} not found! Exiting.")
        return
    model.to(device).eval()

    done_manifest = os.path.join(args.output_dir, f"done_manifest_v3_w{rank}.jsonl")
    fail_manifest = os.path.join(args.output_dir, f"fail_manifest_v3_w{rank}.jsonl")

    dataset = PointCloudDataset(files_to_process)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.dataloader_workers,
        collate_fn=custom_collate,
        drop_last=False,
    )

    processed_files = 0
    total_files = len(files_to_process)
    start_time = time.time()
    success_count = 0
    fail_count = 0

    with torch.no_grad():
        for batch_points, obj_ids, fpaths in dataloader:
            if batch_points is None:
                continue

            batch_points = batch_points.to(device)
            B = batch_points.shape[0]

            try:
                # Single FPS call via _CachedGroupDivider.
                # Replace model.group_divider with cached wrapper for this batch.
                cached_divider = _CachedGroupDivider(model.group_divider)
                model.group_divider = cached_divider

                # First call: populate cache and get centers
                neighborhood, centers, group_idx = cached_divider(batch_points)
                # (centers: B, num_group, 3)

                # Second call happens INSIDE model.forward(); it will hit the cache
                # so features and centers are guaranteed to match.
                x, _ = model(batch_points)  # x: (B, 513, 384), uses cached divider

                # Restore real group_divider for next batch
                model.group_divider = cached_divider._real

                cls_tokens = x[:, 0, :]       # (B, 384)
                patch_features_all = x[:, 1:, :]  # (B, 512, 384)

                for b_idx in range(B):
                    obj_id = obj_ids[b_idx]
                    final_path = os.path.join(args.output_dir, f"{obj_id}.pack.npz")

                    if os.path.exists(final_path):
                        continue

                    single_cls = cls_tokens[b_idx]            # (384,)
                    single_patch = patch_features_all[b_idx:b_idx+1]  # (1, 512, 384)
                    single_centers = centers[b_idx:b_idx+1]   # (1, 512, 3)

                    # build_object_pack expects features_raw shaped (1, 513, 384)
                    # where [0, 1:, :] are patch features.
                    dummy_cls_slot = torch.zeros(1, 1, 384, device=device)
                    features_raw = torch.cat([dummy_cls_slot, single_patch], dim=1)

                    pack = build_object_pack(features_raw, single_centers, cls_feat=single_cls)
                    # Store float32 features (build_object_pack keeps fp16 internally).
                    pack["patch_feat"] = single_patch[0].cpu().float().numpy()  # (512, 384)
                    pack["patch_xyz"] = single_centers[0].cpu().float().numpy()  # (512, 3)
                    if "cls_feat" in pack:
                        pack["cls_feat"] = single_cls.cpu().float().numpy()      # (384,)


                    tmp_path = os.path.join(args.output_dir, f"{obj_id}.pack.tmp.npz")
                    np.savez_compressed(tmp_path, **pack)
                    os.replace(tmp_path, final_path)

                    with open(done_manifest, "a") as df:
                        df.write(json.dumps({"obj_id": obj_id, "status": "success"}) + "\n")
                    success_count += 1

            except Exception as e:
                # Restore real divider if exception occurs mid-batch
                if hasattr(model.group_divider, "_real"):
                    model.group_divider = model.group_divider._real
                err_str = str(e)
                tb_str = traceback.format_exc()
                for obj_id in obj_ids:
                    with open(fail_manifest, "a") as ff:
                        ff.write(
                            json.dumps({"obj_id": obj_id, "error": f"BATCH ERROR: {err_str}", "traceback": tb_str}) + "\n"
                        )
                    fail_count += 1

            processed_files += B
            if processed_files % 100 < B:
                elapsed = time.time() - start_time
                print(
                    f"Worker {rank:02d} | [{processed_files}/{total_files}] "
                    f"| Speed: {processed_files/elapsed:.2f} it/s | Fails: {fail_count}"
                )

    print(f"Worker {rank:02d} DONE. Success: {success_count}, Fail: {fail_count}")


# ────────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Fixed Stage 1 + CLS Generator (v3)")
    parser.add_argument("--data_dir", type=str, default=os.path.join(os.environ.get("POINTLLM_DATA", "data/pointllm"), "objaverse_data"))
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.environ.get("BCP_PACK_DIR", "data/packs_k16"),
        help="Output directory for fixed packs (use a NEW dir to avoid mixing with v2 packs)",
    )
    parser.add_argument("--ckpt", type=str, default=os.environ.get("POINT_BERT_CKPT", "checkpoints/PointLLM_7B_v1.1_init/point_bert_v1.2.pt"))
    parser.add_argument("--gpus", type=str, default="2,3,4,5")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--dataloader_workers", type=int, default=4)
    parser.add_argument(
        "--subset_file",
        type=str,
        default="",
        help="Path to file containing list of object IDs to process (one per line)",
    )

    # Internal worker flags
    parser.add_argument("--worker_mode", action="store_true")
    parser.add_argument("--rank", type=int, default=-1)
    parser.add_argument("--chunk_file", type=str, default="")

    args = parser.parse_args()

    if args.worker_mode:
        with open(args.chunk_file, "r") as f:
            files_to_process = json.load(f)
        process_worker(args.rank, args, files_to_process)
        return

    os.makedirs(args.output_dir, exist_ok=True)

    if args.subset_file:
        with open(args.subset_file, "r") as f:
            subset_ids = [line.strip() for line in f if line.strip()]
        all_files = [os.path.join(args.data_dir, f"{sid}_8192.npy") for sid in subset_ids]
    else:
        all_files = sorted([str(p) for p in Path(args.data_dir).glob("*_8192.npy")])

    # Skip already-done objects
    already_done = {
        p.stem.replace(".pack", "")
        for p in Path(args.output_dir).glob("*.pack.npz")
    }
    if already_done:
        before = len(all_files)
        all_files = [f for f in all_files if Path(f).stem.replace("_8192", "") not in already_done]
        print(f"Skipping {before - len(all_files)} already-done objects. Remaining: {len(all_files)}")

    print(f"Processing {len(all_files)} files → {args.output_dir}")

    safe_gpus = [int(g.strip()) for g in args.gpus.split(",")]
    num_workers = len(safe_gpus)
    splits = np.array_split(all_files, num_workers)

    import subprocess

    processes = []
    for rank in range(num_workers):
        chunk = splits[rank].tolist()
        assigned_gpu = safe_gpus[rank % len(safe_gpus)]
        if len(chunk) == 0:
            continue

        chunk_file = os.path.join(args.output_dir, f"chunk_v3_w{rank}.json")
        with open(chunk_file, "w") as f:
            json.dump(chunk, f)

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(assigned_gpu)

        cmd = [
            "bash",
            "-c",
            f"export CUDA_VISIBLE_DEVICES={assigned_gpu} && {sys.executable} {__file__} "
            f"--worker_mode --rank {rank} --chunk_file {chunk_file} "
            f"--data_dir {args.data_dir} --output_dir {args.output_dir} --ckpt {args.ckpt} "
            f"--batch_size {args.batch_size} --dataloader_workers {args.dataloader_workers}",
        ]

        out_f = open(os.path.join(args.output_dir, f"worker_v3_{rank}.log"), "w")
        p = subprocess.Popen(cmd, env=env, stdout=out_f, stderr=out_f)
        processes.append((p, out_f))

    for p, out_f in processes:
        p.wait()
        out_f.close()

    print("All workers finished.")


if __name__ == "__main__":
    main()
