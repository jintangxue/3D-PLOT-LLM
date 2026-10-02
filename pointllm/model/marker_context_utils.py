"""Resolve marker-context mode (mutually exclusive ablations)."""

VALID_MARKER_CONTEXT_MODES = frozenset({"none", "stats_only", "graph_only", "stats_graph"})


def resolve_marker_context_mode(mode, stats_legacy=False, graph_legacy=False) -> str:
    """
    If `mode` is a non-None string, it wins (including explicit 'none' to disable).
    If `mode` is None, fall back to the boolean pair (bcp_marker_context_stats/graph).
    """
    if mode is not None:
        m = str(mode).strip().lower()
        if m in ("none", "", "null"):
            return "none"
        if m not in VALID_MARKER_CONTEXT_MODES:
            raise ValueError(
                f"bcp_marker_context_mode must be one of {sorted(VALID_MARKER_CONTEXT_MODES)}, got {mode!r}"
            )
        return m

    if stats_legacy and graph_legacy:
        return "stats_graph"
    if stats_legacy:
        return "stats_only"
    if graph_legacy:
        return "graph_only"
    return "none"
