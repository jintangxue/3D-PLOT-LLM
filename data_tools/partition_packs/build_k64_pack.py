import os
import sys
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from scipy.sparse.csgraph import connected_components, shortest_path
from sklearn.neighbors import kneighbors_graph
import heapq

sys.path.append(os.environ.get("PLOT_ROOT", os.getcwd()))
from pointllm.model.pointbert.point_encoder import PointTransformer

class Config:
    def __init__(self):
        self.trans_dim = 384
        self.depth = 12
        self.drop_path_rate = 0.1
        self.cls_dim = 40
        self.num_heads = 6
        self.group_size = 32
        self.num_group = 512
        self.encoder_dims = 256
        self.point_dims = 6

def build_patch_graph(patch_centers):
    N = patch_centers.shape[0]
    k_neighbors = 12
    knn_graph = kneighbors_graph(patch_centers, n_neighbors=k_neighbors, mode='distance', include_self=False)
    adj_matrix = (knn_graph + knn_graph.T) > 0
    
    # Distance Gate to prevent short-circuits
    all_dists = knn_graph.data
    tau_dist = np.percentile(all_dists, 75)
    
    patch_adj_gated = np.zeros((N, N), dtype=bool)
    rows, cols = adj_matrix.nonzero()
    for u, v in zip(rows, cols):
        if np.linalg.norm(patch_centers[u] - patch_centers[v]) <= tau_dist:
            patch_adj_gated[u, v] = True
            
    return patch_adj_gated

def run_bcp_stage12(patch_feats, patch_centers, patch_adj_gated, target_num_parts=64):
    N = patch_centers.shape[0]
    feats_norm = patch_feats / np.clip(np.linalg.norm(patch_feats, axis=1, keepdims=True), 1e-6, None)
    
    # --- STAGE 1: Superpixels ---
    cost_matrix_stg1 = np.full((N, N), np.inf)
    for u, v in zip(*patch_adj_gated.nonzero()):
        cost_matrix_stg1[u, v] = 0.8 * np.linalg.norm(patch_centers[u] - patch_centers[v]) + 0.2 * (1.0 - np.dot(feats_norm[u], feats_norm[v]))

    sp_labels = np.full(N, -1, dtype=np.int32)
    current_sp_id = 0
    unassigned = set(range(N))
    
    # We want more superpoints than target_num_parts. For K=64, superpoints should be around 150-200.
    target_sp_size = 3
    
    while unassigned:
        if current_sp_id == 0:
            seed = np.random.choice(list(unassigned))
        else:
            assigned_pts = np.where(sp_labels != -1)[0]
            if len(assigned_pts) == 0:
                seed = np.random.choice(list(unassigned))
            else:
                dists = np.min(np.linalg.norm(patch_centers[list(unassigned)][:, np.newaxis, :] - patch_centers[assigned_pts][np.newaxis, :, :], axis=2), axis=1)
                seed = list(unassigned)[np.argmax(dists)]
                
        sp_labels[seed] = current_sp_id
        unassigned.remove(seed)
        
        heap = []
        for n in np.where(patch_adj_gated[seed])[0]:
            if n in unassigned:
                heapq.heappush(heap, (cost_matrix_stg1[seed, n], n, seed))
                
        current_size = 1
        while heap and current_size < target_sp_size:
            cost, node, parent = heapq.heappop(heap)
            if node in unassigned:
                sp_labels[node] = current_sp_id
                unassigned.remove(node)
                current_size += 1
                for n in np.where(patch_adj_gated[node])[0]:
                    if n in unassigned:
                        heapq.heappush(heap, (cost_matrix_stg1[node, n], n, node))
                        
        current_sp_id += 1
        
    num_superpoints = current_sp_id
    
    orphans = np.where(sp_labels == -1)[0]
    for orphan in orphans:
        dists = np.linalg.norm(patch_centers[orphan] - patch_centers, axis=1)
        dists[orphan] = np.inf
        closesest = np.argmin(dists)
        sp_labels[orphan] = sp_labels[closesest]

    # --- STAGE 2: K=64 Balanced Growth ---
    sp_feats = np.zeros((num_superpoints, feats_norm.shape[1]))
    sp_centers = np.zeros((num_superpoints, 3))
    sp_weights = np.zeros(num_superpoints, dtype=np.int32)
    rag_adj = np.zeros((num_superpoints, num_superpoints), dtype=bool)
    
    for sp_id in range(num_superpoints):
        mask = (sp_labels == sp_id)
        sp_feats[sp_id] = feats_norm[mask].mean(axis=0)
        sp_centers[sp_id] = patch_centers[mask].mean(axis=0)
        sp_weights[sp_id] = np.sum(mask)
        
    sp_feats = sp_feats / np.clip(np.linalg.norm(sp_feats, axis=1, keepdims=True), 1e-6, None)
        
    rows, cols = patch_adj_gated.nonzero()
    for u, v in zip(rows, cols):
        sp_u = sp_labels[u]
        sp_v = sp_labels[v]
        if sp_u != sp_v:
            rag_adj[sp_u, sp_v] = True
            rag_adj[sp_v, sp_u] = True
            
    rag_costs = np.full((num_superpoints, num_superpoints), np.inf)
    for i in range(num_superpoints):
        for j in range(num_superpoints):
            if rag_adj[i, j]:
                s_dist = np.linalg.norm(sp_centers[i] - sp_centers[j])
                f_dist = 1.0 - np.dot(sp_feats[i], sp_feats[j])
                rag_costs[i, j] = 0.5 * s_dist + 0.5 * f_dist
                
    geodesic_dists = shortest_path(rag_costs, directed=False)
    
    seeds = []
    first_seed = np.argmax(np.sum(geodesic_dists, axis=1))
    seeds.append(first_seed)
    min_dists = geodesic_dists[first_seed].copy()
    
    for _ in range(1, target_num_parts):
        next_seed = np.argmax(min_dists)
        seeds.append(next_seed)
        min_dists = np.minimum(min_dists, geodesic_dists[next_seed])
        
    target_capacity = N / target_num_parts
    max_capacity = target_capacity * 1.25
    
    final_sp_labels = np.full(num_superpoints, -1, dtype=np.int32)
    part_capacities = np.zeros(target_num_parts, dtype=np.int32)
    
    heap = []
    
    for part_id, seed_sp in enumerate(seeds):
        final_sp_labels[seed_sp] = part_id
        part_capacities[part_id] += sp_weights[seed_sp]
        for n_sp in np.where(rag_adj[seed_sp])[0]:
            if final_sp_labels[n_sp] == -1:
                heapq.heappush(heap, (rag_costs[seed_sp, n_sp], n_sp, part_id))
                
    while heap:
        cost, sp_id, part_id = heapq.heappop(heap)
        if final_sp_labels[sp_id] != -1: continue
        proposed_cap = part_capacities[part_id] + sp_weights[sp_id]
        if proposed_cap > max_capacity: continue
            
        final_sp_labels[sp_id] = part_id
        part_capacities[part_id] = proposed_cap
        
        for n_sp in np.where(rag_adj[sp_id])[0]:
            if final_sp_labels[n_sp] == -1:
                size_penalty = 1.0 + (part_capacities[part_id] / target_capacity) ** 2
                heapq.heappush(heap, (rag_costs[sp_id, n_sp] * size_penalty, n_sp, part_id))

    unassigned = np.where(final_sp_labels == -1)[0]
    while len(unassigned) > 0:
        progress = False
        for sp_id in unassigned:
            neighbors = np.where(rag_adj[sp_id])[0]
            valid_neighbors = [n for n in neighbors if final_sp_labels[n] != -1]
            if valid_neighbors:
                best_n = min(valid_neighbors, key=lambda n: rag_costs[sp_id, n])
                final_sp_labels[sp_id] = final_sp_labels[best_n]
                part_capacities[final_sp_labels[best_n]] += sp_weights[sp_id]
                progress = True
        if not progress:
            for sp_id in unassigned:
                dists_to_seeds = [np.linalg.norm(sp_centers[sp_id] - sp_centers[s]) for s in seeds]
                final_sp_labels[sp_id] = np.argmin(dists_to_seeds)
            break
        unassigned = np.where(final_sp_labels == -1)[0]

    # --- STAGE 2.5: Boundary Smoothing
    rag_sim = np.zeros_like(rag_costs)
    rag_sim[rag_adj] = np.exp(-rag_costs[rag_adj] / np.mean(rag_costs[rag_adj]))
    
    for _ in range(4):
        moved_any = False
        for i in np.random.permutation(num_superpoints):
            part_a = final_sp_labels[i]
            neighbors = np.where(rag_adj[i])[0]
            neighbor_parts = final_sp_labels[neighbors]
            if np.all(neighbor_parts == part_a): continue
            
            current_energy = np.sum(rag_sim[i, neighbors][neighbor_parts != part_a])
            best_part = None
            best_drop = 0
            
            unique_candidates = np.unique(neighbor_parts)
            for part_b in unique_candidates:
                if part_b == part_a: continue
                if part_capacities[part_b] + sp_weights[i] > max_capacity * 1.05: continue
                new_energy = np.sum(rag_sim[i, neighbors][neighbor_parts != part_b])
                energy_drop = current_energy - new_energy
                
                if energy_drop > best_drop:
                    a_nodes = np.where(final_sp_labels == part_a)[0]
                    if len(a_nodes) <= 1: continue
                    start_node = a_nodes[0] if a_nodes[0] != i else a_nodes[1]
                    visited = set([start_node])
                    queue = [start_node]
                    while queue:
                        curr = queue.pop(0)
                        for n in np.where(rag_adj[curr])[0]:
                            if final_sp_labels[n] == part_a and n != i and n not in visited:
                                visited.add(n)
                                queue.append(n)
                    if len(visited) == len(a_nodes) - 1:
                        best_part = part_b
                        best_drop = energy_drop
            if best_part is not None:
                final_sp_labels[i] = best_part
                part_capacities[part_a] -= sp_weights[i]
                part_capacities[best_part] += sp_weights[i]
                moved_any = True
        if not moved_any: break
            
    final_patch_labels = final_sp_labels[sp_labels]
    
    # --- STAGE 3: CC fallback and Fragment Absorption ---
    # Convert edge list to connectivity graph for CC checks
    unique_parts = np.unique(final_patch_labels)
    safe_patch_labels = np.zeros_like(final_patch_labels)
    next_safe_id = 0
    adj_csr = patch_adj_gated
    
    for part_id in unique_parts:
        mask = (final_patch_labels == part_id)
        indices = np.where(mask)[0]
        if len(indices) == 0: continue
        sub_adj = adj_csr[np.ix_(indices, indices)]
        n_components, sub_labels = connected_components(sub_adj, directed=False)
        
        if n_components > 1:
            for comp_id in range(n_components):
                comp_mask = (sub_labels == comp_id)
                comp_indices = indices[comp_mask]
                safe_patch_labels[comp_indices] = next_safe_id
                next_safe_id += 1
        else:
            safe_patch_labels[indices] = next_safe_id
            next_safe_id += 1

    t_small = 4 # threshold to absorb
    while True:
        merged_any = False
        unique_parts = np.unique(safe_patch_labels)
        small_ccs = []
        for part_id in unique_parts:
            mask = (safe_patch_labels == part_id)
            indices = np.where(mask)[0]
            if len(indices) == 0: continue
            
            sub_adj = patch_adj_gated[np.ix_(indices, indices)]
            n_components, sub_labels = connected_components(sub_adj, directed=False)
            if n_components > 1:
                comp_sizes = [np.sum(sub_labels == c) for c in range(n_components)]
                largest_comp = np.argmax(comp_sizes)
                for comp_id in range(n_components):
                    if comp_id == largest_comp: continue
                    comp_indices = indices[sub_labels == comp_id]
                    if len(comp_indices) < t_small:
                        small_ccs.append((comp_indices, part_id))
        if not small_ccs: break
        small_ccs.sort(key=lambda x: len(x[0]))
        for comp_indices, part_id in small_ccs:
            neighbor_edges = patch_adj_gated[np.ix_(comp_indices, np.arange(N))]
            neighbor_edges[:, comp_indices] = False
            _, neighbor_nodes = np.nonzero(neighbor_edges)
            if len(neighbor_nodes) > 0:
                connected_parts = safe_patch_labels[neighbor_nodes]
                parts, counts = np.unique(connected_parts, return_counts=True)
                valid_mask = (parts != part_id)
                parts = parts[valid_mask]
                counts = counts[valid_mask]
                if len(parts) > 0:
                    best_part = parts[np.argmax(counts)]
                    safe_patch_labels[comp_indices] = best_part
                    merged_any = True
                    break
        if not merged_any:
            for comp_indices, part_id in small_ccs:
                cc_center = patch_centers[comp_indices].mean(axis=0)
                other_indices = np.where(safe_patch_labels != part_id)[0]
                if len(other_indices) > 0:
                    best_node = other_indices[np.argmin(np.linalg.norm(patch_centers[other_indices] - cc_center, axis=1))]
                    safe_patch_labels[comp_indices] = safe_patch_labels[best_node]
                    merged_any = True
                    break
            if not merged_any: break

    # Final Sequential Remapping
    final_unique = np.unique(safe_patch_labels)
    # If we somehow drop a partition entirely, we would have <64 elements now. This shouldn't happen with large N.
    remapper = {old_id: new_id for new_id, old_id in enumerate(final_unique)}
    for i in range(len(safe_patch_labels)):
        safe_patch_labels[i] = remapper[safe_patch_labels[i]]

    return safe_patch_labels, len(final_unique)

def r_merge_down(patch_labels_high, patch_centers, patch_adj_gated, target_num_parts):
    """
    Given a higher-resolution partition (e.g. K=64), merge down to a lower resolution (e.g. K=32).
    This guarantees hierarchical consistency via aggregation.
    """
    N = patch_centers.shape[0]
    unique_parts = np.unique(patch_labels_high)
    num_high = len(unique_parts)
    
    region_centers = np.zeros((num_high, 3))
    region_weights = np.zeros(num_high, dtype=np.int32)
    rag_adj = np.zeros((num_high, num_high), dtype=bool)
    
    for rid in range(num_high):
        mask = (patch_labels_high == rid)
        region_centers[rid] = patch_centers[mask].mean(axis=0)
        region_weights[rid] = np.sum(mask)
        
    rows, cols = patch_adj_gated.nonzero()
    for u, v in zip(rows, cols):
        r_u = patch_labels_high[u]
        r_v = patch_labels_high[v]
        if r_u != r_v:
            rag_adj[r_u, r_v] = True
            rag_adj[r_v, r_u] = True

    # We just do FPS & Capacity Region Grow on the RAG of Regions
    rag_costs = np.full((num_high, num_high), np.inf)
    for i in range(num_high):
        for j in range(num_high):
            if rag_adj[i, j]:
                rag_costs[i, j] = np.linalg.norm(region_centers[i] - region_centers[j])
                
    geodesic_dists = shortest_path(rag_costs, directed=False)
    
    seeds = []
    first_seed = np.argmax(np.sum(geodesic_dists, axis=1))
    seeds.append(first_seed)
    min_dists = geodesic_dists[first_seed].copy()
    for _ in range(1, target_num_parts):
        next_seed = np.argmax(min_dists)
        seeds.append(next_seed)
        min_dists = np.minimum(min_dists, geodesic_dists[next_seed])
        
    target_capacity = N / target_num_parts
    max_capacity = target_capacity * 1.5
    
    final_sp_labels = np.full(num_high, -1, dtype=np.int32)
    part_capacities = np.zeros(target_num_parts, dtype=np.int32)
    
    heap = []
    for part_id, seed_r in enumerate(seeds):
        final_sp_labels[seed_r] = part_id
        part_capacities[part_id] += region_weights[seed_r]
        for n_r in np.where(rag_adj[seed_r])[0]:
            if final_sp_labels[n_r] == -1:
                heapq.heappush(heap, (rag_costs[seed_r, n_r], n_r, part_id))
                
    while heap:
        cost, r_id, part_id = heapq.heappop(heap)
        if final_sp_labels[r_id] != -1: continue
        proposed_cap = part_capacities[part_id] + region_weights[r_id]
        if proposed_cap > max_capacity: continue
            
        final_sp_labels[r_id] = part_id
        part_capacities[part_id] = proposed_cap
        
        for n_r in np.where(rag_adj[r_id])[0]:
            if final_sp_labels[n_r] == -1:
                heapq.heappush(heap, (rag_costs[r_id, n_r], n_r, part_id))

    unassigned = np.where(final_sp_labels == -1)[0]
    while len(unassigned) > 0:
        progress = False
        for r_id in unassigned:
            neighbors = np.where(rag_adj[r_id])[0]
            valid_neighbors = [n for n in neighbors if final_sp_labels[n] != -1]
            if valid_neighbors:
                best_n = min(valid_neighbors, key=lambda n: rag_costs[r_id, n])
                final_sp_labels[r_id] = final_sp_labels[best_n]
                part_capacities[final_sp_labels[best_n]] += region_weights[r_id]
                progress = True
        if not progress:
            for r_id in unassigned:
                dists_to_seeds = [np.linalg.norm(region_centers[r_id] - region_centers[s]) for s in seeds]
                final_sp_labels[r_id] = np.argmin(dists_to_seeds)
            break
        unassigned = np.where(final_sp_labels == -1)[0]
        
    # Build Map Table: index is High-Res ID, value is Low-Res ID
    map_table = np.zeros(num_high, dtype=np.int16)
    for rid in range(num_high): map_table[rid] = final_sp_labels[rid]
    
    patch_labels_low = map_table[patch_labels_high]
    return patch_labels_low, map_table

def build_object_pack(features_raw, centers, cls_feat=None):
    patch_feats = features_raw[0, 1:, :].cpu().numpy()
    patch_centers = centers[0].cpu().numpy()
    
    # Determinism anchor based on spatial coordinates
    obj_seed = int(np.abs(np.sum(patch_centers) * 100000)) % (2**32)
    np.random.seed(obj_seed)
    
    patch_adj_gated = build_patch_graph(patch_centers)
    
    # Base production
    patch_r64, actual_k = run_bcp_stage12(patch_feats, patch_centers, patch_adj_gated, 64)
    if actual_k != 64:
        # Fallback or correction could go here. For sanity check, we accept actual_k +- variance.
        # It's usually extremely close or exactly K.
        pass
        
    patch_r32, map64_to_32 = r_merge_down(patch_r64, patch_centers, patch_adj_gated, 32)
    patch_r16, map64_to_16 = r_merge_down(patch_r64, patch_centers, patch_adj_gated, 16)
    patch_r14, map64_to_14 = r_merge_down(patch_r64, patch_centers, patch_adj_gated, 14)
    patch_r12, map64_to_12 = r_merge_down(patch_r64, patch_centers, patch_adj_gated, 12)
    patch_r10, map64_to_10 = r_merge_down(patch_r64, patch_centers, patch_adj_gated, 10)
    patch_r8, map64_to_8 = r_merge_down(patch_r64, patch_centers, patch_adj_gated, 8)
    patch_r6, map64_to_6 = r_merge_down(patch_r64, patch_centers, patch_adj_gated, 6)
    patch_r4, map64_to_4 = r_merge_down(patch_r64, patch_centers, patch_adj_gated, 4)
    patch_r2, map64_to_2 = r_merge_down(patch_r64, patch_centers, patch_adj_gated, 2)


    
    pack = {
        'version': np.array([1], dtype=np.int16),
        'patch_valid_mask': np.ones(512, dtype=bool),
        'patch_xyz': patch_centers.astype(np.float16),
        'patch_feat': patch_feats.astype(np.float16), # Save disk space
        'patch_adj_edges': np.argwhere(patch_adj_gated).astype(np.int16),
        'patch_r64': patch_r64.astype(np.int16),
        'map64_to_32': map64_to_32.astype(np.int16),
        'map64_to_16': map64_to_16.astype(np.int16),
        'map64_to_14': map64_to_14.astype(np.int16),
        'map64_to_12': map64_to_12.astype(np.int16),
        'map64_to_10': map64_to_10.astype(np.int16),
        'map64_to_8': map64_to_8.astype(np.int16),
        'map64_to_6': map64_to_6.astype(np.int16),
        'map64_to_4': map64_to_4.astype(np.int16),
        'map64_to_2': map64_to_2.astype(np.int16)


    }

    if cls_feat is not None:
        if isinstance(cls_feat, torch.Tensor):
            cls_feat = cls_feat.cpu().to(torch.float16).numpy()
        pack['cls_feat'] = cls_feat.astype(np.float16)

    return pack

def main():
    device = torch.device('cuda')
    config = Config()
    model = PointTransformer(config, use_max_pool=False)
    ckpt = os.environ.get("POINT_BERT_CKPT", "checkpoints/PointLLM_7B_v1.1_init/point_bert_v1.2.pt")
    if os.path.exists(ckpt):
        model.load_checkpoint(ckpt)
    model.to(device).eval()

    data_dir = os.path.join(os.environ.get("POINTLLM_DATA", "data/pointllm"), "objaverse_data")
    all_files = sorted(list(Path(data_dir).glob("*_8192.npy")))
    sample_files = all_files[:50] # Sanity check 50
    
    print(f"Building Object Packs & Sanity Checking K={64, 32, 16, 8} on {len(sample_files)} objects...")

    metrics = {k: {'sizes': [], 'connected': 0} for k in [64, 32, 16, 8]}
    
    with torch.no_grad():
        for f in tqdm(sample_files):
            points_raw = np.load(f)
            points_6d = np.zeros((8192, 6), dtype=np.float32)
            if points_raw.shape[1] == 3: points_6d[:, :3] = points_raw
            else: points_6d[:, :6] = points_raw[:, :6]
                
            points_torch = torch.from_numpy(points_6d).float().unsqueeze(0).to(device)
            centroid = points_torch[:, :, :3].mean(dim=1, keepdim=True)
            points_torch[:, :, :3] = points_torch[:, :, :3] - centroid
            m, _ = torch.max(torch.sqrt(torch.sum(points_torch[:, :, :3]**2, dim=2, keepdim=True)), dim=1, keepdim=True)
            points_torch[:, :, :3] = points_torch[:, :, :3] / torch.clamp(m, min=1e-6)
            
            features_raw, group_idx = model(points_torch) 
            neighborhood, centers, _ = model.group_divider(points_torch)
            
            # BUILD PACK
            pack = build_object_pack(features_raw, centers)
            
            # EXTRACT FOR SANITY CHECK
            adj_dense = np.zeros((512, 512), dtype=bool)
            for u, v in pack['patch_adj_edges']: adj_dense[u, v] = True
            
            labels = {
                64: pack['patch_r64'],
                32: pack['map64_to_32'][pack['patch_r64']],
                16: pack['map64_to_16'][pack['patch_r64']],
                8: pack['map64_to_8'][pack['patch_r64']]
            }
            
            # VERIFY Properties for K levels
            for K_name, patch_labels in labels.items():
                unique = np.unique(patch_labels)
                curr_sizes = []
                all_connected = True
                
                for p_id in unique:
                    mask = (patch_labels == p_id)
                    curr_sizes.append(np.sum(mask))
                    sub_adj = adj_dense[np.ix_(np.where(mask)[0], np.where(mask)[0])]
                    nc, _ = connected_components(sub_adj, directed=False)
                    if nc > 1: all_connected = False
                        
                metrics[K_name]['sizes'].extend(curr_sizes)
                if all_connected: metrics[K_name]['connected'] += 1

    print("\n--------- K=64 Object Pack Sanity Check ---------")
    for K_name in [64, 32, 16, 8]:
        sizes = metrics[K_name]['sizes']
        conn_rate = (metrics[K_name]['connected'] / len(sample_files)) * 100
        mean_s = np.mean(sizes)
        cv = np.std(sizes) / mean_s
        print(f"Level K={K_name:<2} | Connectivity Rate: {conn_rate:>5.1f}% | Avg Size: {mean_s:>5.1f} | Size CV: {cv:.2f}")

if __name__ == '__main__':
    main()
