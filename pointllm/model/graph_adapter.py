import torch
import torch.nn as nn

class RegionGraphAdapter(nn.Module):
    """
    Implements a simple relation-aware graph enhancement layer.
    """
    def __init__(self, in_features=384, stats_dim=7):
        super().__init__()
        
        # Linear projections for different branches
        self.W_n = nn.Linear(in_features, in_features, bias=False)
        self.W_s = nn.Linear(in_features, in_features, bias=False)
        self.W_g = nn.Linear(stats_dim, in_features, bias=False)
        
        # MLP updater
        self.mlp = nn.Sequential(
            nn.Linear(in_features, in_features),
            nn.GELU(),
            nn.Linear(in_features, in_features)
        )
        
    def forward(self, R0, adj, stats):
        """
        R0: (B, 16, in_features) - Original region means
        adj: (B, 16, 16) - Binary adjacency matrix for region k-NN graph
        stats: (B, 16, stats_dim) - Region statistics (centroid, size, span)
        """
        # Self branch
        s_i = self.W_s(R0)  # (B, 16, 384)
        
        # Neighbor branch
        n_feat = self.W_n(R0)  # (B, 16, 384)
        degree = adj.sum(dim=-1, keepdim=True)  # (B, 16, 1)
        
        # Protect division by zero for isolated isolated regions
        degree_safe = degree.clone()
        degree_safe[degree == 0] = 1.0
        
        # Mean aggregation
        m_i = torch.bmm(adj, n_feat) / degree_safe  # (B, 16, 384)
        # Note: if degree == 0, the row in adj is all zeros, so bmm gives 0 for that row. 
        # m_i is safely 0 for isolated nodes.
        
        # Stats branch
        g_i = self.W_g(stats)  # (B, 16, 384)
        
        # Fusion
        u_i = s_i + m_i + g_i
        
        # Update
        R1 = R0 + self.mlp(u_i)
        
        return R1


class VocabGraphPropagator(nn.Module):
    """
    Graph message-passing on vocab token embeddings in LLM space (4096-dim).

    Operates AFTER embed_tokens(<part_k>) lookup, BEFORE assembly into the
    point token sequence. This avoids the point_proj bottleneck that plagues
    backbone-space graph approaches.

    Uses a bottleneck MLP (D_llm -> D_bottleneck -> D_llm) to keep parameter
    count manageable in the high-dimensional LLM space.

    Update rule:
        agg_g = mean({W_n(E_j) : j in neighbors(g)})
        delta_g = MLP(W_s(E_g) + agg_g)
        E'_g = E_g + scale * delta_g

    Final layer of MLP and scale are zero-initialized for stable training start.
    """

    def __init__(self, in_features=4096, bottleneck=256, stats_dim=7):
        super().__init__()
        self.W_n = nn.Linear(in_features, bottleneck, bias=False)
        self.W_s = nn.Linear(in_features, bottleneck, bias=False)
        self.W_g = nn.Linear(stats_dim, bottleneck, bias=False)
        self.mlp = nn.Sequential(
            nn.Linear(bottleneck, bottleneck),
            nn.GELU(),
            nn.Linear(bottleneck, in_features),
        )
        # Zero-init output layer for identity-start
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, E, adj, stats=None):
        """
        E:     (B, K, D_llm)  — vocab token embeddings
        adj:   (B, K, K)      — binary adjacency
        stats: (B, K, 7) | None — region statistics (optional)
        Returns: (B, K, D_llm)
        """
        degree = adj.sum(dim=-1, keepdim=True)
        degree_safe = degree.clone()
        degree_safe[degree == 0] = 1.0

        n_feat = self.W_n(E)                         # (B, K, bottleneck)
        agg = torch.bmm(adj, n_feat) / degree_safe   # (B, K, bottleneck)
        u = self.W_s(E) + agg                         # (B, K, bottleneck)
        if stats is not None:
            u = u + self.W_g(stats)
        delta = self.mlp(u)                            # (B, K, D_llm)
        return E + delta


class VocabStatsMLP(nn.Module):
    """
    LLM-vocab-space stats-to-residual MLP (analog of marker_stats_mlp at D_llm).
    Used ONLY by the decoupled vocab path (bcp_vocab_context_mode in stats_only/stats_graph).
    Zero-init output so identity-start.
    """

    def __init__(self, stats_dim=7, in_features=4096, hidden=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(stats_dim, hidden, bias=False),
            nn.GELU(),
            nn.Linear(hidden, in_features, bias=False),
        )
        nn.init.zeros_(self.mlp[-1].weight)

    def forward(self, stats):
        # stats: (B, K, stats_dim) -> (B, K, in_features)
        return self.mlp(stats)


class VocabGraphPropagatorDecoupled(nn.Module):
    """
    Graph-only propagator on vocab token embeddings (adj + self only; no stats).
    Matches MarkerGraphPropagator's two-stage philosophy: stats flows through a
    separate VocabStatsMLP; graph runs on the stats-updated embeddings.

    Used ONLY by the decoupled vocab path (bcp_vocab_context_mode in graph_only/stats_graph).
    """

    def __init__(self, in_features=4096, bottleneck=256):
        super().__init__()
        self.W_n = nn.Linear(in_features, bottleneck, bias=False)
        self.W_s = nn.Linear(in_features, bottleneck, bias=False)
        self.mlp = nn.Sequential(
            nn.Linear(bottleneck, bottleneck),
            nn.GELU(),
            nn.Linear(bottleneck, in_features),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, E, adj):
        # E:   (B, K, D_llm)
        # adj: (B, K, K)
        degree = adj.sum(dim=-1, keepdim=True)
        degree_safe = degree.clone()
        degree_safe[degree == 0] = 1.0
        n_feat = self.W_n(E)
        agg = torch.bmm(adj, n_feat) / degree_safe
        u = self.W_s(E) + agg
        delta = self.mlp(u)
        return E + delta


class MarkerGraphPropagator(nn.Module):
    """
    Marker-path graph update: **adjacency + self** only (no stats branch).
    Used so `stats_graph` mode does not run a second heavy stats encoder inside the graph step
    (stats enter only via the separate marker StatsMLP on M).

    Returns M_out = M + delta(M, adj) with delta from a small MLP; caller should apply
    outer residual: M' = M_ref + scale * (M_out - M_ref) so only the **delta** is scaled.
    """

    def __init__(self, in_features=384):
        super().__init__()
        self.W_n = nn.Linear(in_features, in_features, bias=False)
        self.W_s = nn.Linear(in_features, in_features, bias=False)
        self.mlp = nn.Sequential(
            nn.Linear(in_features, in_features),
            nn.GELU(),
            nn.Linear(in_features, in_features),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, M, adj):
        # M: (B, K, D), adj: (B, K, K)
        degree = adj.sum(dim=-1, keepdim=True)
        degree_safe = degree.clone()
        degree_safe[degree == 0] = 1.0
        n_feat = self.W_n(M)
        agg = torch.bmm(adj, n_feat) / degree_safe
        u = self.W_s(M) + agg
        delta = self.mlp(u)
        return M + delta
