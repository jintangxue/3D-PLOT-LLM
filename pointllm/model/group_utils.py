"""
Utility functions for BCP group-based patch token reordering.
"""
import torch
import numpy as np
from collections import deque


def bfs_order_from_adj(region_adj, region_stats=None, num_groups=16):
    """
    Compute a BFS traversal order over the region adjacency graph.

    Starting node: the region whose centroid is closest to the global centroid
    (deterministic, no randomness). Ties broken by lowest index.

    Args:
        region_adj: (K, K) numpy array or torch tensor, binary adjacency
        region_stats: (K, 7) numpy array or torch tensor, optional.
                      If provided, columns 0:3 are centroids used to pick BFS root.
                      If None, starts from region 0.
        num_groups: int

    Returns:
        order: list of int, length K, a permutation of range(K)
    """
    if isinstance(region_adj, torch.Tensor):
        region_adj = region_adj.cpu().numpy()
    if region_stats is not None and isinstance(region_stats, torch.Tensor):
        region_stats = region_stats.cpu().numpy()

    # Pick BFS root: region closest to global centroid
    if region_stats is not None:
        centroids = region_stats[:num_groups, 0:3]
        global_centroid = centroids.mean(axis=0)
        dists = np.linalg.norm(centroids - global_centroid, axis=1)
        root = int(np.argmin(dists))
    else:
        root = 0

    visited = set()
    order = []
    queue = deque([root])
    visited.add(root)
    while queue:
        node = queue.popleft()
        order.append(node)
        # Get neighbors sorted by index for determinism
        neighbors = np.where(region_adj[node] > 0)[0]
        for nb in sorted(neighbors):
            if nb not in visited:
                visited.add(nb)
                queue.append(nb)
    # Handle disconnected components (shouldn't happen but be safe)
    for g in range(num_groups):
        if g not in visited:
            order.append(g)
    return order


def remap_group_ids_by_order(group_ids, order):
    """
    Remap group_ids so that BFS-order[i] becomes new group i.

    Args:
        group_ids: (512,) numpy int32 array, original group assignments
        order: list of int, BFS traversal order (old_group_id sequence)

    Returns:
        new_group_ids: (512,) numpy int32 array, remapped
    """
    old_to_new = np.zeros(len(order), dtype=np.int32)
    for new_id, old_id in enumerate(order):
        old_to_new[old_id] = new_id
    return old_to_new[group_ids]


def reorder_by_group(patch_tokens, group_ids, num_groups=16):
    """
    Reorder 512 patch tokens into [G0, G1, ..., G(K-1)] without any per-group cap or padding.
    Each group Gi will have variable number of tokens.
    """
    total_assigned = 0
    chunks = []
    for i in range(num_groups):
        idx = (group_ids == i).nonzero(as_tuple=True)[0]
        chunks.append(patch_tokens[idx])
        total_assigned += len(idx)
    
    assert total_assigned == 512, f"Total assigned tokens {total_assigned} != 512"
    return torch.cat(chunks, dim=0)


def assemble_part_interleaved(cls_feat, patch_feat, group_ids,
                               num_groups=16, marker=None,
                               object_ids=None,
                               bcp_part_no_mean=False,
                               summary_mlp=None,
                               graph_adapter=None,
                               region_adj=None,
                               region_stats=None):
    """
    Assemble interleaved part token sequence with variable-length groups.
    Core Invariant: preserve exactly 512 tokens. T = 1 + 16 (headers) + 512 = 529.
    
    Validation: sorted(perm_indices) == range(512).
    """
    B, N, D = patch_feat.shape
    has_marker = marker is not None
    out = []

    for b in range(B):
        obj_id = object_ids[b] if object_ids is not None else f"batch_idx_{b}"
        seq = [cls_feat[b].unsqueeze(0)]            # (1, D)
        
        perm_indices = []
        total_assigned = 0

        # Phase 1: gather means and clusters
        headers_list = []
        clusters_list = []
        
        for g in range(num_groups):
            idx = (group_ids[b] == g).nonzero(as_tuple=True)[0]
            n = len(idx)
            
            # Strict Assertion: No empty groups allowed in this research phase
            assert n > 0, f"Empty group {g} detected for {obj_id}. Logical invariant violated."
            
            group_tokens = patch_feat[b][idx]       # (n, D)
            total_assigned += n
            perm_indices.extend(idx.tolist())
            
            header = group_tokens.mean(dim=0, keepdim=True)    # (1, D)
            headers_list.append(header)
            clusters_list.append(group_tokens)

        # Permutation Invariant Check
        assert total_assigned == 512, f"Total assigned tokens {total_assigned} != 512 for {obj_id}"
        assert sorted(perm_indices) == list(range(512)), f"Permutation index coverage failed for {obj_id}"
        
        # Phase 2: apply GraphAdapter if available
        headers_t = torch.cat(headers_list, dim=0).unsqueeze(0) # (1, 16, D)
        if graph_adapter is not None and region_adj is not None and region_stats is not None:
            headers_t = graph_adapter(headers_t, region_adj[b:b+1], region_stats[b:b+1])

        # Phase 3: assembly
        for g in range(num_groups):
            header = headers_t[0, g:g+1]
            if summary_mlp is not None:
                header = summary_mlp(header)                   # (1, D)
            
            group_tokens = clusters_list[g]

            if has_marker:
                if marker.dim() == 3:
                    # Per-sample marker update path: marker shape (B, K, D).
                    m = marker[b, g:g+1]
                else:
                    m = marker[0:1] if marker.shape[0] == 1 else marker[g:g+1]
                if bcp_part_no_mean:
                    seq.extend([m, group_tokens])
                else:
                    seq.extend([m, header, group_tokens])
            else:
                if bcp_part_no_mean:
                    seq.extend([group_tokens])
                else:
                    seq.extend([header, group_tokens])
        
        out.append(torch.cat(seq, dim=0))

    return torch.stack(out, dim=0)


def assemble_global_region_prefix(cls_feat, patch_feat, group_ids,
                                  num_groups=16, object_ids=None,
                                  summary_mlp=None,
                                  graph_adapter=None,
                                  region_adj=None,
                                  region_stats=None):
    """
    Assemble sequence with global region-prefix tokens:
    [CLS, R0, R1, ..., R15, reordered_patch_tokens]

    - Rg is mean(group g tokens) by default.
    - If summary_mlp is provided, Rg = summary_mlp(mean(group g tokens)).
    """
    B, _, _ = patch_feat.shape
    out = []

    for b in range(B):
        obj_id = object_ids[b] if object_ids is not None else f"batch_idx_{b}"
        seq = [cls_feat[b].unsqueeze(0)]  # (1, D)

        total_assigned = 0
        grouped_chunks = []
        headers_list = []

        for g in range(num_groups):
            idx = (group_ids[b] == g).nonzero(as_tuple=True)[0]
            n = len(idx)
            assert n > 0, f"Empty group {g} detected for {obj_id}. Logical invariant violated."

            group_tokens = patch_feat[b][idx]  # (n, D)
            total_assigned += n
            grouped_chunks.append(group_tokens)

            header = group_tokens.mean(dim=0, keepdim=True)  # (1, D)
            headers_list.append(header)

        assert total_assigned == 512, f"Total assigned tokens {total_assigned} != 512 for {obj_id}"

        headers_t = torch.cat(headers_list, dim=0).unsqueeze(0) # (1, 16, D)
        if graph_adapter is not None and region_adj is not None and region_stats is not None:
            headers_t = graph_adapter(headers_t, region_adj[b:b+1], region_stats[b:b+1])

        for g in range(num_groups):
            header = headers_t[0, g:g+1]
            if summary_mlp is not None:
                header = summary_mlp(header)
            seq.append(header)

        reordered_patch = torch.cat(grouped_chunks, dim=0)  # (512, D)
        seq.append(reordered_patch)
        out.append(torch.cat(seq, dim=0))  # (529, D)

    return torch.stack(out, dim=0)


def assemble_part_vocab_tokens(cls_proj, patch_proj, group_ids,
                                part_embeddings,
                                num_groups=16,
                                object_ids=None,
                                graph_means_proj=None):
    """
    Assemble interleaved sequence in post-projection (LLM) space.
    <part_n> tokens come from the LLM's embed_tokens (real vocab tokens), NOT the projector.

    Two layouts depending on graph_means_proj:

    Without graph (bcp_part_vocab_token only):
      [CLS, <part_0>, G0_patches, <part_1>, G1_patches, ..., <part_K-1>, GK-1_patches]
      Total: 1 + K + 512  tokens

    With graph (bcp_part_vocab_token + bcp_graph_proj_d):
      [CLS, <part_0>, graph_mean_0, G0_patches, ..., <part_K-1>, graph_mean_K-1, GK-1_patches]
      Total: 1 + 2K + 512  tokens
      Mirrors graph_e (assemble_part_interleaved with per_part marker + graph) but uses
      real vocab tokens instead of nn.Parameter markers.

    Args:
        cls_proj:        (B, 1, D_llm)   CLS already projected via point_proj
        patch_proj:      (B, 512, D_llm) patches already projected via point_proj
        group_ids:       (B, 512)        group assignments
        part_embeddings: (K, D_llm)      embed_tokens(part_token_ids)
        num_groups:      int             K
        object_ids:      list[str]       for diagnostic messages
        graph_means_proj:(B, K, D_llm) | None
                         graph-enhanced region means, already projected via point_proj.
                         When provided, inserted as an explicit token after each <part_g>.

    Returns:
        Without graph: (B, 1+K+512, D_llm)
        With graph:    (B, 1+2K+512, D_llm)
    """
    B = patch_proj.shape[0]
    _per_sample_pe = (part_embeddings.dim() == 3)  # (B, K, D) vs (K, D)
    out = []

    for b in range(B):
        obj_id = object_ids[b] if object_ids is not None else f"batch_idx_{b}"
        seq = [cls_proj[b]]   # (1, D_llm)

        total_assigned = 0
        perm_indices = []

        for g in range(num_groups):
            idx = (group_ids[b] == g).nonzero(as_tuple=True)[0]
            n = len(idx)
            assert n > 0, f"Empty group {g} detected for {obj_id}. Invariant violated."

            part_emb = part_embeddings[b, g:g+1] if _per_sample_pe else part_embeddings[g:g+1]
            group_patches = patch_proj[b][idx]        # (n, D_llm)
            total_assigned += n
            perm_indices.extend(idx.tolist())

            if graph_means_proj is not None:
                # Layout: [<part_g>, graph_mean_g, G_g_patches]
                graph_mean = graph_means_proj[b, g:g+1]  # (1, D_llm)
                seq.extend([part_emb, graph_mean, group_patches])
            else:
                # Layout: [<part_g>, G_g_patches]
                seq.extend([part_emb, group_patches])

        assert total_assigned == 512, \
            f"Total assigned {total_assigned} != 512 for {obj_id}"
        assert sorted(perm_indices) == list(range(512)), \
            f"Permutation coverage failed for {obj_id}"

        out.append(torch.cat(seq, dim=0))

    return torch.stack(out, dim=0)


def interleave_insert_vocab_tokens(seq_proj, group_ids, part_embeddings,
                                   num_groups=16,
                                   has_marker=False,
                                   has_header=True,
                                   object_ids=None):
    """
    Insert real vocab <part_g> tokens into an already assembled interleaved sequence.

    Input sequence is projected and follows interleave layout:
      [CLS, (marker?), (header?), G0_patches, (marker?), (header?), G1_patches, ...]

    Output inserts one vocab token before each group block:
      [CLS, <part_0>, block0, <part_1>, block1, ..., <part_{K-1}>, blockK-1]

    This enables hybrid "point-side interleave markers/headers + vocab tokens"
    while keeping patch ordering and group membership unchanged.
    """
    B, T, D = seq_proj.shape
    _per_sample_pe = (part_embeddings.dim() == 3)  # (B, K, D) vs (K, D)
    out = []
    extra_per_group = (1 if has_marker else 0) + (1 if has_header else 0)

    for b in range(B):
        obj_id = object_ids[b] if object_ids is not None else f"batch_idx_{b}"
        seq = [seq_proj[b, 0:1]]  # CLS
        cursor = 1
        total_assigned = 0

        for g in range(num_groups):
            idx = (group_ids[b] == g).nonzero(as_tuple=True)[0]
            n = len(idx)
            assert n > 0, f"Empty group {g} detected for {obj_id}. Invariant violated."
            block_len = extra_per_group + n
            block = seq_proj[b, cursor:cursor + block_len]
            assert block.shape[0] == block_len, f"Interleave block length mismatch for {obj_id}, group {g}"
            _pe = part_embeddings[b, g:g+1] if _per_sample_pe else part_embeddings[g:g+1]
            seq.extend([_pe, block])
            cursor += block_len
            total_assigned += n

        assert total_assigned == 512, f"Total assigned {total_assigned} != 512 for {obj_id}"
        assert cursor == T, f"Cursor did not consume full sequence for {obj_id}: {cursor} vs {T}"
        out.append(torch.cat(seq, dim=0))

    return torch.stack(out, dim=0)


