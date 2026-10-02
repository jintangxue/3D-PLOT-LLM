#    Copyright 2023 Runsen Xu

from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from .utils import *
from pointllm.utils import *
from .group_utils import (
    assemble_part_interleaved,
    assemble_global_region_prefix,
    reorder_by_group,
    assemble_part_vocab_tokens,
    interleave_insert_vocab_tokens,
)

from contextlib import nullcontext
from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM

from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from .graph_adapter import RegionGraphAdapter, MarkerGraphPropagator
from .marker_context_utils import resolve_marker_context_mode

import os

# * add logger
import logging
logger = logging.getLogger(__name__)


class PointLLMConfig(LlamaConfig):
    model_type = "pointllm"

class PointLLMLlamaModel(LlamaModel):
    config_class = PointLLMConfig 

    def __init__(self, config: LlamaConfig):
        super(PointLLMLlamaModel, self).__init__(config)

        self.point_backbone_type = config.point_backbone
        logger.info(f"Using {self.point_backbone_type}.")

        if self.point_backbone_type == "PointBERT":
            from pointllm.model import PointTransformer
            # address of config file, in the same dir of this file
            point_bert_config_name = getattr(config, "point_backbone_config_name", "PointTransformer_8192point_2layer") # * default for v1.2, v1.1 uses PointTransformer_base_8192point.yaml
            point_bert_config_addr = os.path.join(os.path.dirname(__file__), "pointbert", f"{point_bert_config_name}.yaml")
            print(f"Loading PointBERT config from {point_bert_config_addr}.")
            point_bert_config = cfg_from_yaml_file(point_bert_config_addr)
            if getattr(config, "use_color", False):
                point_bert_config.model.point_dims = 6
            use_max_pool = getattr(point_bert_config.model, "use_max_pool", False) # * default is false
            
            self.point_backbone = PointTransformer(point_bert_config.model, use_max_pool=use_max_pool)
            logger.info(f"Using {self.point_backbone.point_dims} dim of points.")

            self.point_backbone_config = {
                "point_cloud_dim": point_bert_config.model.point_dims,
                "backbone_output_dim": point_bert_config.model.trans_dim if not use_max_pool else point_bert_config.model.trans_dim * 2,
                "project_output_dim": self.config.hidden_size,
                "point_token_len": point_bert_config.model.num_group + 1 if not use_max_pool else 1, # * number of output features, with cls token
                "mm_use_point_start_end": self.config.mm_use_point_start_end,
                "projection_hidden_layer": point_bert_config.model.get('projection_hidden_layer', 0),
                "use_max_pool": use_max_pool
            }
            if point_bert_config.model.get('projection_hidden_layer', 0) > 0:
                self.point_backbone_config["projection_hidden_dim"] = point_bert_config.model.projection_hidden_dim # a list
            
            logger.info(f"Use max pool is {use_max_pool}. Number of point token is {self.point_backbone_config['point_token_len']}.")

        # Learnable part-marker embeddings (pre-projector space, 384-dim).
        # Created unconditionally so they appear in checkpoints.
        # Activated via point_backbone_config['bcp_part_marker'] = 'shared' | 'per_part'.
        # part_marker  : shared marker for Version B (1, backbone_output_dim)
        # part_markers : per-part markers for Version C (16, backbone_output_dim)
        _bb_dim = self.point_backbone_config["backbone_output_dim"]
        self.part_marker  = nn.Parameter(torch.zeros(1,  _bb_dim))  # Version B
        _MAX_GROUPS = 16  # always allocate 16 slots; forward uses [:K] based on bcp_num_groups
        self.part_markers = nn.Parameter(torch.zeros(_MAX_GROUPS, _bb_dim))  # Version C
        # Projected region summary token for "MLP(mean(region_patches))" experiments.
        # IMPORTANT: keep this lazy to avoid breaking resume for old checkpoints.
        self.region_summary_mlp = None
        if getattr(config, "bcp_part_projected_mean", False):
            self.enable_region_summary_mlp()
            
        self.region_graph_adapter = None
        if getattr(config, "bcp_graph_proj_d", False):
            self.enable_region_graph_adapter()
        self.vocab_graph_propagator = None
        if getattr(config, "bcp_vocab_graph_propagation", False):
            self.enable_vocab_graph_propagator()
        self.vocab_graph_scale = nn.Parameter(torch.tensor(0.05))
        self.marker_stats_mlp = None
        self.marker_graph_propagator = None
        # Keep update strengths as trainable scalars (initialized small) to preserve marker identity.
        self.marker_stats_scale = nn.Parameter(torch.tensor(0.05))
        self.marker_graph_scale = nn.Parameter(torch.tensor(0.05))
        _mctx = resolve_marker_context_mode(
            getattr(config, "bcp_marker_context_mode", None),
            getattr(config, "bcp_marker_context_stats", False),
            getattr(config, "bcp_marker_context_graph", False),
        )
        if _mctx in ("stats_only", "stats_graph"):
            self.enable_marker_stats_mlp()
        if _mctx in ("graph_only", "stats_graph"):
            self.enable_marker_graph_propagator()

        # --- Vocabulary-side refinement (LSR), exclusive with bcp_vocab_graph_propagation ---
        # Same stats/graph decomposition as MSR, applied to <part_k> vocab embeddings in LLM space.
        self.vocab_stats_mlp = None
        self.vocab_graph_propagator_v2 = None
        self.vocab_stats_scale = nn.Parameter(torch.tensor(0.05))
        # vocab_graph_scale (already declared above) is reused for the graph residual of the decoupled path.
        _vctx = resolve_marker_context_mode(
            getattr(config, "bcp_vocab_context_mode", None),
            False, False,
        )
        if _vctx in ("stats_only", "stats_graph"):
            self.enable_vocab_stats_mlp()
        if _vctx in ("graph_only", "stats_graph"):
            self.enable_vocab_graph_propagator_decoupled()
        # Safety: disallow mixing old fused path with new decoupled path.
        if _vctx != "none" and getattr(config, "bcp_vocab_graph_propagation", False):
            raise ValueError(
                "bcp_vocab_context_mode and bcp_vocab_graph_propagation are mutually exclusive. "
                "Set only one."
            )

        # * print relevant info with projection layers
        backbone_output_dim = self.point_backbone_config["backbone_output_dim"]
        logger.info(f"Point backbone output dim: {backbone_output_dim}.")
        logger.info(f"Use {self.point_backbone_config['projection_hidden_layer']} projection hiddent layers.")
        if self.point_backbone_config['projection_hidden_layer'] > 0:
            # Add projection layer with linear layers and GELU activation
            projection_layers = []
            last_dim = backbone_output_dim
            for i in range(point_bert_config.model.projection_hidden_layer):
                projection_layers.append(nn.Linear(last_dim, self.point_backbone_config["projection_hidden_dim"][i]))
                projection_layers.append(nn.GELU())
                last_dim = self.point_backbone_config["projection_hidden_dim"][i]

            projection_layers.append(nn.Linear(last_dim, self.point_backbone_config["project_output_dim"]))
            self.point_proj = nn.Sequential(*projection_layers)
            logger.info(f"Each layer with {point_bert_config.model.projection_hidden_dim} hidden units.")
        else:
            # Single layer
            self.point_proj = nn.Linear(backbone_output_dim, self.point_backbone_config['project_output_dim'])
        logger.info(f"Point projector output dim: {self.point_backbone_config['project_output_dim']}.")

        self.fix_pointnet = False
        self.fix_llm = False

    def enable_region_summary_mlp(self):
        """Create region summary MLP only when projected-mean mode is enabled."""
        if self.region_summary_mlp is not None:
            return
        _bb_dim = self.point_backbone_config["backbone_output_dim"]
        self.region_summary_mlp = nn.Sequential(
            nn.Linear(_bb_dim, _bb_dim),
            nn.GELU(),
            nn.Linear(_bb_dim, _bb_dim),
        )

    def enable_region_graph_adapter(self):
        """Create region graph adapter for Graph-Proj-D/E."""
        if self.region_graph_adapter is not None:
            return
        _bb_dim = self.point_backbone_config["backbone_output_dim"]
        self.region_graph_adapter = RegionGraphAdapter(in_features=_bb_dim, stats_dim=7)

    def enable_vocab_graph_propagator(self):
        """Create vocab-token graph propagator for LLM-space message passing."""
        if self.vocab_graph_propagator is not None:
            return
        from .graph_adapter import VocabGraphPropagator
        _llm_dim = self.point_backbone_config["project_output_dim"]
        self.vocab_graph_propagator = VocabGraphPropagator(
            in_features=_llm_dim, bottleneck=256, stats_dim=7
        )

    def enable_vocab_stats_mlp(self):
        """Decoupled vocab path: stats-to-residual MLP on <part_k> embeddings."""
        if self.vocab_stats_mlp is not None:
            return
        from .graph_adapter import VocabStatsMLP
        _llm_dim = self.point_backbone_config["project_output_dim"]
        self.vocab_stats_mlp = VocabStatsMLP(
            stats_dim=7, in_features=_llm_dim, hidden=256
        )

    def enable_vocab_graph_propagator_decoupled(self):
        """Decoupled vocab path: graph-only propagator (stats handled separately)."""
        if self.vocab_graph_propagator_v2 is not None:
            return
        from .graph_adapter import VocabGraphPropagatorDecoupled
        _llm_dim = self.point_backbone_config["project_output_dim"]
        self.vocab_graph_propagator_v2 = VocabGraphPropagatorDecoupled(
            in_features=_llm_dim, bottleneck=256
        )

    def enable_marker_stats_mlp(self):
        """Create stats-to-marker residual mapper (bias-free so stats=0 => zero update)."""
        if self.marker_stats_mlp is not None:
            return
        _bb_dim = self.point_backbone_config["backbone_output_dim"]
        self.marker_stats_mlp = nn.Sequential(
            nn.Linear(7, _bb_dim, bias=False),
            nn.GELU(),
            nn.Linear(_bb_dim, _bb_dim, bias=False),
        )

    def enable_marker_graph_propagator(self):
        """Adjacency-primary marker graph (no stats inside — avoids double stats with StatsMLP)."""
        if self.marker_graph_propagator is not None:
            return
        _bb_dim = self.point_backbone_config["backbone_output_dim"]
        self.marker_graph_propagator = MarkerGraphPropagator(in_features=_bb_dim)

    def _apply_marker_context_update(self, marker_base, region_adj, region_stats, point_backbone_config):
        """
        Mutually exclusive modes (see bcp_marker_context_mode):
          stats_only:  M' = M + s_stats * StatsMLP(stats)
          graph_only:  M' = M + s_graph * (G(M, adj) - M)   # delta form; G uses adj+self only
          stats_graph: M_t = M + s_stats * StatsMLP(stats);
                       M' = M_t + s_graph * (G(M_t, adj) - M_t)

        StatsMLP uses bias=False so all-zero stats => zero contribution.
        G is MarkerGraphPropagator (no stats branch) so stats are not re-encoded inside the graph step.
        """
        if marker_base is None:
            return None

        mode = resolve_marker_context_mode(
            point_backbone_config.get("bcp_marker_context_mode"),
            point_backbone_config.get("bcp_marker_context_stats", False),
            point_backbone_config.get("bcp_marker_context_graph", False),
        )
        if mode == "none":
            return marker_base

        if mode in ("stats_only", "stats_graph"):
            if region_stats is None:
                return marker_base
            B, K, seven = region_stats.shape
            if seven != 7:
                raise ValueError(f"region_stats last dim must be 7, got {seven}")
        else:
            B = K = None

        if mode == "graph_only":
            if region_adj is None:
                return marker_base
            B, K = region_adj.shape[0], region_adj.shape[1]
        elif mode == "stats_graph" and region_adj is not None:
            B2, K2 = region_adj.shape[0], region_adj.shape[1]
            if B != B2 or K != K2:
                raise ValueError(
                    f"region_adj (B,K)=({B2},{K2}) vs region_stats (B,K)=({B},{K})"
                )

        if marker_base.shape[0] == 1:
            marker = marker_base.expand(K, -1).unsqueeze(0).expand(B, -1, -1).contiguous().clone()
        else:
            if marker_base.shape[0] != K:
                raise ValueError(f"per_part marker dim0 must be K={K} or 1, got {marker_base.shape[0]}")
            marker = marker_base.unsqueeze(0).expand(B, -1, -1).contiguous().clone()

        if mode in ("stats_only", "stats_graph"):
            if region_stats.shape[:2] != (B, K):
                raise ValueError(
                    f"region_stats batch/part mismatch: got {tuple(region_stats.shape[:2])}, expected ({B},{K})"
                )

        if mode == "graph_only" or (mode == "stats_graph" and region_adj is not None):
            if region_adj.shape[:2] != (B, K):
                raise ValueError(
                    f"region_adj batch/part mismatch: got {tuple(region_adj.shape[:2])}, expected ({B},{K})"
                )

        if mode == "stats_only":
            if self.marker_stats_mlp is None:
                self.enable_marker_stats_mlp()
            return marker + self.marker_stats_scale * self.marker_stats_mlp(region_stats)

        if mode == "graph_only":
            if self.marker_graph_propagator is None:
                self.enable_marker_graph_propagator()
            G = self.marker_graph_propagator(marker, region_adj.to(marker.dtype))
            delta_graph = G - marker
            return marker + self.marker_graph_scale * delta_graph

        # stats_graph
        if self.marker_stats_mlp is None:
            self.enable_marker_stats_mlp()
        m_tilde = marker + self.marker_stats_scale * self.marker_stats_mlp(region_stats)
        if region_adj is None:
            return m_tilde
        if self.marker_graph_propagator is None:
            self.enable_marker_graph_propagator()
        G = self.marker_graph_propagator(m_tilde, region_adj.to(m_tilde.dtype))
        delta_graph = G - m_tilde
        return m_tilde + self.marker_graph_scale * delta_graph

    def load_point_backbone_checkpoint(self, checkpoint_path=None):
        self.point_backbone.load_checkpoint(self.config.point_backbone_ckpt if checkpoint_path is None else checkpoint_path)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        point_clouds: Optional[torch.FloatTensor] = None,
        bcp_patch_feat: Optional[torch.FloatTensor] = None,  # (B, 512, 384) offline patch tokens
        bcp_cls_feat: Optional[torch.FloatTensor] = None,    # (B, 384)  offline CLS token (upgraded packs)
        bcp_group_ids: Optional[torch.LongTensor] = None,    # (B, 512) group assignment for reorder
        bcp_region_adj: Optional[torch.FloatTensor] = None,  # (B, 16, 16) adjacency matrix
        bcp_region_stats: Optional[torch.FloatTensor] = None, # (B, 16, 7) centroid, size, span
        object_ids: Optional[List[str]] = None,              # (B,) list of object IDs for debug
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:

        # Restore the original embeddings during Stage 1
        orig_embeds_params = getattr(self, 'orig_embeds_params', None)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        point_features = None
        point_backbone = getattr(self, 'point_backbone', None)
        point_backbone_config = getattr(self, 'point_backbone_config', None)

        # Determine whether we have fully offline features (cls + patch both precomputed).
        # When bcp_cls_feat is provided we can skip the PointBERT forward entirely.
        _has_full_offline = (bcp_patch_feat is not None and bcp_cls_feat is not None)
        _needs_backbone   = (point_backbone is not None
                             and (input_ids.shape[1] != 1 or self.training)
                             and point_clouds is not None
                             and not _has_full_offline)

        # Flag: does this batch use vocab-token interleave (already fully projected)?
        _point_features_projected = False

        if _has_full_offline and (input_ids.shape[1] != 1 or self.training):
            # ── Fully offline path ─────────────────────────────────────────────────
            # bcp_part_interleave: explicit [CLS, g_mean, g_patches×32] × 16 layout
            # This is an OFFLINE-ONLY feature (requires cls_feat + patch_feat + group_ids).
            _bcp_part_interleave = (
                point_backbone_config is not None and
                point_backbone_config.get('bcp_part_interleave', False)
            )
            _bcp_part_projected_mean = (
                point_backbone_config is not None and
                point_backbone_config.get('bcp_part_projected_mean', False)
            )
            _bcp_global_region_prefix = (
                point_backbone_config is not None and
                point_backbone_config.get('bcp_global_region_prefix', False)
            )
            _bcp_graph_proj_d = (
                point_backbone_config is not None and
                point_backbone_config.get('bcp_graph_proj_d', False)
            )
            _bcp_part_vocab_token = (
                point_backbone_config is not None and
                point_backbone_config.get('bcp_part_vocab_token', False)
            )

            import sys as _sys

            if _bcp_part_interleave and _bcp_part_vocab_token:
                # ── Hybrid path: interleave (marker/header) + real vocab <part_n> ───────
                # Build interleaved sequence in backbone space first, then project, then
                # insert <part_n> vocab embeddings in LLM space before each group block.
                _num_groups = point_backbone_config.get('bcp_num_groups', 16)
                marker_mode = point_backbone_config.get('bcp_part_marker', 'none')
                if marker_mode == 'shared':
                    _marker = self.part_marker
                elif marker_mode == 'per_part':
                    _marker = self.part_markers[:_num_groups]
                else:
                    _marker = None
                if _marker is not None:
                    _marker = self._apply_marker_context_update(
                        _marker,
                        bcp_region_adj.to(inputs_embeds) if bcp_region_adj is not None else None,
                        bcp_region_stats.to(inputs_embeds) if bcp_region_stats is not None else None,
                        point_backbone_config,
                    )
                _summary_mlp = None
                _no_mean = point_backbone_config.get('bcp_part_no_mean', False)
                _projected_mean = point_backbone_config.get('bcp_part_projected_mean', False)
                if _projected_mean:
                    if self.region_summary_mlp is None:
                        self.enable_region_summary_mlp()
                    _summary_mlp = self.region_summary_mlp

                interleave_bb = assemble_part_interleaved(
                    bcp_cls_feat.to(inputs_embeds),
                    bcp_patch_feat.to(inputs_embeds),
                    bcp_group_ids,
                    num_groups=_num_groups,
                    marker=_marker,
                    object_ids=object_ids,
                    bcp_part_no_mean=_no_mean,
                    summary_mlp=_summary_mlp,
                    graph_adapter=self.region_graph_adapter if _bcp_graph_proj_d else None,
                    region_adj=bcp_region_adj.to(inputs_embeds) if bcp_region_adj is not None else None,
                    region_stats=bcp_region_stats.to(inputs_embeds) if bcp_region_stats is not None else None,
                )  # (B, T_interleave, bb_dim)

                interleave_proj = self.point_proj(interleave_bb)  # (B, T_interleave, D_llm)

                part_ids = point_backbone_config.get('part_token_ids')
                assert part_ids is not None, "part_token_ids not found in point_backbone_config"
                part_id_tensor = torch.tensor(part_ids, dtype=torch.long, device=inputs_embeds.device)
                part_embeddings = self.embed_tokens(part_id_tensor)  # (K, D_llm)

                # Vocabulary-side refinement (used when bcp_vocab_context_mode is set)
                _vctx = getattr(self.config, "bcp_vocab_context_mode", None) or "none"
                if _vctx != "none" and bcp_region_adj is not None:
                    _pe = part_embeddings.unsqueeze(0).expand(interleave_proj.shape[0], -1, -1).to(inputs_embeds.dtype)
                    _adj = bcp_region_adj.to(_pe)
                    _stats = bcp_region_stats.to(_pe) if bcp_region_stats is not None else None
                    # Stage A: stats-only residual
                    if _vctx in ("stats_only", "stats_graph") and self.vocab_stats_mlp is not None and _stats is not None:
                        _pe = _pe + self.vocab_stats_scale * self.vocab_stats_mlp(_stats)
                    # Stage B: graph-only residual
                    if _vctx in ("graph_only", "stats_graph") and self.vocab_graph_propagator_v2 is not None:
                        _pe_g = self.vocab_graph_propagator_v2(_pe, _adj)
                        _pe = _pe + self.vocab_graph_scale * (_pe_g - _pe)
                    part_embeddings = _pe

                # Fused-path vocabulary graph propagation
                elif self.vocab_graph_propagator is not None and bcp_region_adj is not None:
                    _pe = part_embeddings.unsqueeze(0).expand(interleave_proj.shape[0], -1, -1)  # (B, K, D_llm)
                    _adj = bcp_region_adj.to(_pe)
                    _stats = bcp_region_stats.to(_pe) if bcp_region_stats is not None else None
                    _pe_updated = self.vocab_graph_propagator(_pe, _adj, _stats)
                    _delta = _pe_updated - _pe
                    part_embeddings = part_embeddings.unsqueeze(0).expand_as(_pe) + self.vocab_graph_scale * _delta
                    # part_embeddings is now (B, K, D_llm) — per-sample updated

                point_features = interleave_insert_vocab_tokens(
                    interleave_proj,
                    bcp_group_ids,
                    part_embeddings=part_embeddings,
                    num_groups=_num_groups,
                    has_marker=(marker_mode in ('shared', 'per_part')),
                    has_header=(not _no_mean),
                    object_ids=object_ids,
                )  # (B, 1 + K + interleave_without_cls, D_llm)
                _point_features_projected = True

            elif _bcp_part_vocab_token:
                # ── Vocab-token interleave path ────────────────────────────────────
                # Layout: [CLS, <part_0>, G0, ..., <part_K-1>, GK-1]  (1+K+512 tokens)
                # <part_n> tokens live in LLM embedding space; they are NOT routed through point_proj.
                # We project CLS and patches separately, then interleave with vocab embeddings.
                _num_groups = point_backbone_config.get('bcp_num_groups', 16)
                assert bcp_group_ids is not None, "bcp_part_vocab_token requires group_ids"

                # Project CLS and patches through point_proj
                cls_proj   = self.point_proj(bcp_cls_feat.to(inputs_embeds).unsqueeze(1))   # (B, 1, D_llm)
                patch_proj = self.point_proj(bcp_patch_feat.to(inputs_embeds))              # (B, 512, D_llm)

                # Fetch <part_n> embeddings from LLM vocab
                part_ids = point_backbone_config.get('part_token_ids')  # list of K token IDs
                assert part_ids is not None, "part_token_ids not found in point_backbone_config"
                part_id_tensor = torch.tensor(part_ids, dtype=torch.long, device=inputs_embeds.device)
                part_embeddings = self.embed_tokens(part_id_tensor)  # (K, D_llm)

                # ── Optional graph enhancement ──────────────────────────────────────
                # Compute graph_means_proj = point_proj(graph_adapter(region_means)).
                # These are inserted as EXPLICIT tokens after each <part_n>:
                #   [<part_g>, graph_mean_g, G_g_patches]
                # Layout becomes: [CLS, <part_0>, gm_0, G0, ..., <part_K-1>, gm_{K-1}, GK-1]
                # Total tokens: 1 + 2K + 512  (= 545 for K=16)
                _graph_means_proj = None
                if (_bcp_graph_proj_d
                        and self.region_graph_adapter is not None
                        and bcp_region_adj is not None
                        and bcp_region_stats is not None):
                    # Compute backbone-dim region means from raw patch features (before projection)
                    raw_patch = bcp_patch_feat.to(inputs_embeds)  # (B, 512, backbone_dim)
                    means_list = []
                    for b in range(raw_patch.shape[0]):
                        means_list.append(torch.stack([
                            raw_patch[b][(bcp_group_ids[b] == g).nonzero(as_tuple=True)[0]].mean(0)
                            for g in range(_num_groups)
                        ]))  # (K, backbone_dim)
                    means_bb = torch.stack(means_list, dim=0)  # (B, K, backbone_dim)

                    # GraphAdapter: (B, K, bb_dim) → (B, K, bb_dim)
                    enhanced_means = self.region_graph_adapter(
                        means_bb,
                        bcp_region_adj.to(inputs_embeds),
                        bcp_region_stats.to(inputs_embeds),
                    )

                    # Project to LLM dim — reuse point_proj
                    B_sz = enhanced_means.shape[0]
                    _graph_means_proj = self.point_proj(
                        enhanced_means.view(B_sz * _num_groups, -1)
                    ).view(B_sz, _num_groups, -1)  # (B, K, D_llm)

                # Vocabulary-side refinement
                _vctx = getattr(self.config, "bcp_vocab_context_mode", None) or "none"
                if _vctx != "none" and bcp_region_adj is not None:
                    B_sz = patch_proj.shape[0]
                    _pe = part_embeddings.unsqueeze(0).expand(B_sz, -1, -1).to(patch_proj.dtype)
                    _adj = bcp_region_adj.to(_pe)
                    _stats = bcp_region_stats.to(_pe) if bcp_region_stats is not None else None
                    if _vctx in ("stats_only", "stats_graph") and self.vocab_stats_mlp is not None and _stats is not None:
                        _pe = _pe + self.vocab_stats_scale * self.vocab_stats_mlp(_stats)
                    if _vctx in ("graph_only", "stats_graph") and self.vocab_graph_propagator_v2 is not None:
                        _pe_g = self.vocab_graph_propagator_v2(_pe, _adj)
                        _pe = _pe + self.vocab_graph_scale * (_pe_g - _pe)
                    part_embeddings = _pe

                # Fused-path vocabulary graph propagation
                elif self.vocab_graph_propagator is not None and bcp_region_adj is not None:
                    B_sz = patch_proj.shape[0]
                    _pe = part_embeddings.unsqueeze(0).expand(B_sz, -1, -1)  # (B, K, D_llm)
                    _adj = bcp_region_adj.to(_pe)
                    _stats = bcp_region_stats.to(_pe) if bcp_region_stats is not None else None
                    _pe_updated = self.vocab_graph_propagator(_pe, _adj, _stats)
                    _delta = _pe_updated - _pe
                    part_embeddings = _pe + self.vocab_graph_scale * _delta  # (B, K, D_llm)

                point_features = assemble_part_vocab_tokens(
                    cls_proj,
                    patch_proj,
                    bcp_group_ids,
                    part_embeddings=part_embeddings,
                    num_groups=_num_groups,
                    object_ids=object_ids,
                    graph_means_proj=_graph_means_proj,
                )  # (B, 1+K+512, D_llm) or (B, 1+2K+512, D_llm)
                _point_features_projected = True  # skip outer point_proj call


            elif _bcp_global_region_prefix:
                point_features = assemble_global_region_prefix(
                    bcp_cls_feat.to(inputs_embeds),
                    bcp_patch_feat.to(inputs_embeds),
                    bcp_group_ids,
                    num_groups=point_backbone_config.get('bcp_num_groups', 16),
                    object_ids=object_ids,
                    summary_mlp=None,  # this mode is for insertion-position ablation
                    graph_adapter=self.region_graph_adapter if _bcp_graph_proj_d else None,
                    region_adj=bcp_region_adj.to(inputs_embeds) if bcp_region_adj is not None else None,
                    region_stats=bcp_region_stats.to(inputs_embeds) if bcp_region_stats is not None else None,
                )  # (B, 529, 384)
            elif _bcp_part_interleave:
                # Interleaved layout: (B, T, D)
                # group_ids MUST be present — loaded when bcp_part_interleave=True in dataset
                _num_groups = point_backbone_config.get('bcp_num_groups', 16)
                marker_mode = point_backbone_config.get('bcp_part_marker', 'none')
                if marker_mode == 'shared':
                    _marker = self.part_marker   # (1, D)
                elif marker_mode == 'per_part':
                    _marker = self.part_markers[:_num_groups]  # slice to actual K
                else:
                    _marker = None
                if _marker is not None:
                    _marker = self._apply_marker_context_update(
                        _marker,
                        bcp_region_adj.to(inputs_embeds) if bcp_region_adj is not None else None,
                        bcp_region_stats.to(inputs_embeds) if bcp_region_stats is not None else None,
                        point_backbone_config,
                    )
                _summary_mlp = None
                if _bcp_part_projected_mean:
                    if self.region_summary_mlp is None:
                        # Defensive path for eval/inference when config enables projected mean.
                        self.enable_region_summary_mlp()
                    _summary_mlp = self.region_summary_mlp

                point_features = assemble_part_interleaved(
                    bcp_cls_feat.to(inputs_embeds),
                    bcp_patch_feat.to(inputs_embeds),
                    bcp_group_ids,
                    num_groups=point_backbone_config.get('bcp_num_groups', 16),
                    marker=_marker,
                    object_ids=object_ids,
                    bcp_part_no_mean=point_backbone_config.get('bcp_part_no_mean', False),
                    summary_mlp=_summary_mlp,
                    graph_adapter=self.region_graph_adapter if _bcp_graph_proj_d else None,
                    region_adj=bcp_region_adj.to(inputs_embeds) if bcp_region_adj is not None else None,
                    region_stats=bcp_region_stats.to(inputs_embeds) if bcp_region_stats is not None else None,
                )  # (B, 529 or 545, 384)
            else:
                # Flat layout: (B, 513, D) — [CLS, p0..p511]
                offline = bcp_patch_feat.to(inputs_embeds)  # (B, 512, 384)
                if bcp_group_ids is not None:
                    offline = torch.stack([
                        reorder_by_group(offline[b], bcp_group_ids[b], num_groups=point_backbone_config.get('bcp_num_groups', 16))
                        for b in range(offline.shape[0])
                    ])  # (B, 512, 384)
                cls = bcp_cls_feat.unsqueeze(1).to(inputs_embeds)  # (B, 1, 384)
                point_features = torch.cat([cls, offline], dim=1)   # (B, 513, 384)

        elif _needs_backbone:
            # ── Online / partial-offline path (no precomputed CLS) ──────────────────
            with torch.no_grad() if self.fix_pointnet else nullcontext():
                if self.fix_pointnet:
                    self.point_backbone.eval()
                if type(point_clouds) is list:
                    # * variable numbers of points — run full encoder per sample
                    point_features = []
                    for i, point_cloud in enumerate(point_clouds):
                        raw = self.point_backbone(point_cloud.unsqueeze(0))  # (1, 513, 384)
                        if bcp_patch_feat is not None:
                            # Replace patch tokens [1:] with offline feat; keep cls [0]
                            offline = bcp_patch_feat[i:i+1].to(raw)  # (1, 512, 384)
                            if bcp_group_ids is not None:
                                offline = reorder_by_group(offline[0], bcp_group_ids[i]).unsqueeze(0)
                            raw = torch.cat([raw[:, :1, :], offline], dim=1)  # (1, 513, 384)
                        point_features.append(raw[0])
                else:
                    raw = self.point_backbone(point_clouds)  # (B, 513, 384)
                    if bcp_patch_feat is not None:
                        # Replace patch tokens [1:] with offline feat; keep cls [0]
                        offline = bcp_patch_feat.to(raw)     # (B, 512, 384)
                        if bcp_group_ids is not None:
                            reordered = torch.stack([
                                reorder_by_group(offline[b], bcp_group_ids[b], num_groups=point_backbone_config.get('bcp_num_groups', 16))
                                for b in range(offline.shape[0])
                            ])  # (B, 512, 384)
                            offline = reordered
                        raw = torch.cat([raw[:, :1, :], offline], dim=1)  # (B, 513, 384)
                    point_features = raw


        if (_has_full_offline and (input_ids.shape[1] != 1 or self.training)) or _needs_backbone:
            if not _point_features_projected:
                # Vocab-token path already projects inside assembly; skip here.
                if type(point_features) is list:
                    point_features = [self.point_proj(point_feature) for point_feature in point_features]
                else:
                    point_features = self.point_proj(point_features)

            dummy_point_features = torch.zeros(point_backbone_config['point_token_len'], point_backbone_config['backbone_output_dim'], device=inputs_embeds.device, dtype=inputs_embeds.dtype)
            dummy_point_features = self.point_proj(dummy_point_features)

            new_input_embeds = []
            cur_point_idx = 0
            for cur_input_ids, cur_input_embeds in zip(input_ids, inputs_embeds): # * input_ids: B, L; input_embeds: B, L, C
                if (cur_input_ids == point_backbone_config['point_patch_token']).sum() == 0:
                    # multimodal LLM, but the current sample is not multimodal
                    cur_input_embeds = cur_input_embeds + (0. * dummy_point_features).sum() # * do nothing
                    new_input_embeds.append(cur_input_embeds)
                    cur_point_idx += 1
                    continue
                cur_point_features = point_features[cur_point_idx].to(device=cur_input_embeds.device)
                num_patches = cur_point_features.shape[0] # * number of point tokens
                if point_backbone_config['mm_use_point_start_end']:
                    if (cur_input_ids == point_backbone_config["point_start_token"]).sum() != (cur_input_ids == point_backbone_config["point_end_token"]).sum():
                        raise ValueError("The number of point start tokens and point end tokens should be the same.")
                    point_start_tokens = torch.where(cur_input_ids == point_backbone_config["point_start_token"])[0]
                    for point_start_token_pos in point_start_tokens:
                        point_end_pos = int(point_start_token_pos) + int(num_patches) + 1
                        if point_end_pos >= cur_input_ids.shape[0]:
                            raise ValueError("The point end token should follow the point start token.")
                        if cur_input_ids[point_end_pos] != point_backbone_config["point_end_token"]:
                            raise ValueError("The point end token should follow the point start token.")
                        if orig_embeds_params is not None: # * will not update the original embeddings except for POINT_START_TOKEN and POINT_END_TOKEN
                            cur_new_input_embeds = torch.cat((cur_input_embeds[:point_start_token_pos].detach(), cur_input_embeds[point_start_token_pos:point_start_token_pos+1], cur_point_features, cur_input_embeds[point_start_token_pos + num_patches + 1:point_start_token_pos + num_patches + 2], cur_input_embeds[point_start_token_pos + num_patches + 2:].detach()), dim=0)
                        else:
                            cur_new_input_embeds = torch.cat((cur_input_embeds[:point_start_token_pos+1], cur_point_features, cur_input_embeds[point_start_token_pos + num_patches + 1:]), dim=0)
                        cur_point_idx += 1
                    new_input_embeds.append(cur_new_input_embeds)
                else:
                    if (cur_input_ids == point_backbone_config["point_patch_token"]).sum() != num_patches:
                        raise ValueError("The number of point patch tokens should be the same as the number of point patches.")
                    masked_indices = torch.where(cur_input_ids == point_backbone_config["point_patch_token"])[0]
                    mask_index_start = masked_indices[0]
                    if (masked_indices != torch.arange(mask_index_start, mask_index_start+num_patches, device=masked_indices.device, dtype=masked_indices.dtype)).any():
                        raise ValueError("The point patch tokens should be consecutive.")
                    if orig_embeds_params is not None:
                        cur_new_input_embeds = torch.cat((cur_input_embeds[:mask_index_start].detach(), cur_point_features, cur_input_embeds[mask_index_start+num_patches:].detach()), dim=0)
                    else:
                        cur_new_input_embeds = torch.cat((cur_input_embeds[:mask_index_start], cur_point_features, cur_input_embeds[mask_index_start+num_patches:]), dim=0)
                    new_input_embeds.append(cur_new_input_embeds)
                    cur_point_idx += 1
            inputs_embeds = torch.stack(new_input_embeds, dim=0)

        return super(PointLLMLlamaModel, self).forward(
            input_ids=None, attention_mask=attention_mask, past_key_values=past_key_values,
            inputs_embeds=inputs_embeds, use_cache=use_cache,
            output_attentions=output_attentions, output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )


class PointLLMLlamaForCausalLM(LlamaForCausalLM):
    config_class = PointLLMConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = PointLLMLlamaModel(config)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        point_clouds: Optional[torch.FloatTensor] = None,
        bcp_patch_feat: Optional[torch.FloatTensor] = None,
        bcp_cls_feat: Optional[torch.FloatTensor] = None,
        bcp_group_ids: Optional[torch.LongTensor] = None,
        bcp_region_adj: Optional[torch.FloatTensor] = None,
        bcp_region_stats: Optional[torch.FloatTensor] = None,
        object_ids: Optional[List[str]] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            point_clouds=point_clouds,
            bcp_patch_feat=bcp_patch_feat,
            bcp_cls_feat=bcp_cls_feat,
            bcp_group_ids=bcp_group_ids,
            bcp_region_adj=bcp_region_adj,
            bcp_region_stats=bcp_region_stats,
            object_ids=object_ids,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous() # * B, L, V(32003)
            shift_labels = labels[..., 1:].contiguous() # * B, L
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model/pipeline parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "point_clouds": kwargs.get("point_clouds", None),
                "bcp_patch_feat": kwargs.get("bcp_patch_feat", None),
                "bcp_cls_feat": kwargs.get("bcp_cls_feat", None),
                "bcp_group_ids": kwargs.get("bcp_group_ids", None),
                "bcp_region_adj": kwargs.get("bcp_region_adj", None),
                "bcp_region_stats": kwargs.get("bcp_region_stats", None),
                "object_ids": kwargs.get("object_ids", None),
            }
        )
        return model_inputs

    def initialize_tokenizer_point_backbone_config_wo_embedding(self, tokenizer):
        # * called when stage2 or inference or inference without pre-training, assume tokenizer has point tokens
        config = self.config
        point_backbone_config = self.get_model().point_backbone_config
        mm_use_point_start_end = point_backbone_config['mm_use_point_start_end'] = config.mm_use_point_start_end
        point_backbone_config['bcp_part_interleave'] = getattr(config, 'bcp_part_interleave', point_backbone_config.get('bcp_part_interleave', False))
        point_backbone_config['bcp_part_marker'] = getattr(config, 'bcp_part_marker', point_backbone_config.get('bcp_part_marker', 'none'))
        point_backbone_config['bcp_part_no_mean'] = getattr(config, 'bcp_part_no_mean', point_backbone_config.get('bcp_part_no_mean', False))
        point_backbone_config['bcp_graph_proj_d'] = getattr(config, 'bcp_graph_proj_d', point_backbone_config.get('bcp_graph_proj_d', False))
        _mctx = resolve_marker_context_mode(
            getattr(config, 'bcp_marker_context_mode', None),
            getattr(config, 'bcp_marker_context_stats', False),
            getattr(config, 'bcp_marker_context_graph', False),
        )
        point_backbone_config['bcp_marker_context_mode'] = _mctx
        point_backbone_config['bcp_marker_context_stats'] = _mctx in ('stats_only', 'stats_graph')
        point_backbone_config['bcp_marker_context_graph'] = _mctx in ('graph_only', 'stats_graph')
        point_backbone_config['bcp_num_groups'] = getattr(config, 'bcp_num_groups', point_backbone_config.get('bcp_num_groups', 16))

        default_point_patch_token = config.DEFAULT_POINT_PATCH_TOKEN

        tokenizer.add_tokens([default_point_patch_token], special_tokens=True)

        # * assert tokenizer has the default_point_patch_token
        point_backbone_config['default_point_patch_token'] = default_point_patch_token
        point_backbone_config['point_patch_token'] = tokenizer.convert_tokens_to_ids([default_point_patch_token])[0]

        if mm_use_point_start_end:
            default_point_start_token = config.DEFAULT_POINT_START_TOKEN
            default_point_end_token = config.DEFAULT_POINT_END_TOKEN
            tokenizer.add_tokens([default_point_start_token, default_point_end_token], special_tokens=True)

            point_backbone_config['default_point_start_token'] = default_point_start_token
            point_backbone_config['default_point_end_token'] = default_point_end_token

            point_backbone_config["point_start_token"] = tokenizer.convert_tokens_to_ids([default_point_start_token])[0]
            point_backbone_config["point_end_token"] = tokenizer.convert_tokens_to_ids([default_point_end_token])[0]

        # Register <part_n> vocab tokens if this model uses vocab-token interleave.
        if getattr(config, 'bcp_part_vocab_token', False):
            _K = getattr(config, 'bcp_num_groups', 16)
            part_tokens = [f"<part_{g}>" for g in range(_K)]
            num_part_tokens = tokenizer.add_tokens(part_tokens, special_tokens=True)
            if num_part_tokens > 0:
                # Stage2-only vocab path: checkpoint may not contain <part_n> yet.
                # Resize safely only when tokenizer really grew.
                self.resize_token_embeddings(len(tokenizer))
                inp_emb = self.get_input_embeddings().weight.data
                out_emb = self.get_output_embeddings().weight.data
                avg_in = inp_emb[:-num_part_tokens].mean(dim=0, keepdim=True)
                avg_out = out_emb[:-num_part_tokens].mean(dim=0, keepdim=True)
                inp_emb[-num_part_tokens:] = avg_in
                out_emb[-num_part_tokens:] = avg_out
            part_token_ids = tokenizer.convert_tokens_to_ids(part_tokens)
            point_backbone_config['bcp_part_vocab_token'] = True
            point_backbone_config['part_token_ids'] = part_token_ids
    
    def initialize_tokenizer_point_backbone_config(self, tokenizer, device, fix_llm=True):

        config = self.config
        point_backbone_config = self.get_model().point_backbone_config
        mm_use_point_start_end = point_backbone_config['mm_use_point_start_end'] = config.mm_use_point_start_end
        point_backbone_config['bcp_part_interleave'] = getattr(config, 'bcp_part_interleave', point_backbone_config.get('bcp_part_interleave', False))
        point_backbone_config['bcp_part_marker'] = getattr(config, 'bcp_part_marker', point_backbone_config.get('bcp_part_marker', 'none'))
        point_backbone_config['bcp_part_no_mean'] = getattr(config, 'bcp_part_no_mean', point_backbone_config.get('bcp_part_no_mean', False))
        point_backbone_config['bcp_graph_proj_d'] = getattr(config, 'bcp_graph_proj_d', point_backbone_config.get('bcp_graph_proj_d', False))
        _mctx = resolve_marker_context_mode(
            getattr(config, 'bcp_marker_context_mode', None),
            getattr(config, 'bcp_marker_context_stats', False),
            getattr(config, 'bcp_marker_context_graph', False),
        )
        point_backbone_config['bcp_marker_context_mode'] = _mctx
        point_backbone_config['bcp_marker_context_stats'] = _mctx in ('stats_only', 'stats_graph')
        point_backbone_config['bcp_marker_context_graph'] = _mctx in ('graph_only', 'stats_graph')
        point_backbone_config['bcp_num_groups'] = getattr(config, 'bcp_num_groups', point_backbone_config.get('bcp_num_groups', 16))

        default_point_patch_token = config.DEFAULT_POINT_PATCH_TOKEN
        point_backbone_config['default_point_patch_token'] = default_point_patch_token
        tokenizer.add_tokens([default_point_patch_token], special_tokens=True) # * no need to update embed since it will be replaced
        self.resize_token_embeddings(len(tokenizer)) # ! resize_token_embeddings will make the tokens trainable again
        point_backbone_config['point_patch_token'] = tokenizer.convert_tokens_to_ids([default_point_patch_token])[0]

        # ── <part_n> vocab tokens ──────────────────────────────────────────────────
        # Register K part tokens so their embeddings live in LLM space (not point_proj space).
        # Done BEFORE the start/end token block so orig_embeds_params captures the correct baseline.
        if getattr(config, 'bcp_part_vocab_token', False):
            _K = getattr(config, 'bcp_num_groups', 16)
            part_tokens = [f"<part_{g}>" for g in range(_K)]
            num_part_tokens = tokenizer.add_tokens(part_tokens, special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))
            part_token_ids = tokenizer.convert_tokens_to_ids(part_tokens)
            point_backbone_config['bcp_part_vocab_token'] = True
            point_backbone_config['part_token_ids'] = part_token_ids
            # Initialise new token embeddings to vocab average for stable training start
            if num_part_tokens > 0:
                inp_emb = self.get_input_embeddings().weight.data
                out_emb = self.get_output_embeddings().weight.data
                avg_in  = inp_emb[:-num_part_tokens].mean(dim=0, keepdim=True)
                avg_out = out_emb[:-num_part_tokens].mean(dim=0, keepdim=True)
                inp_emb[-num_part_tokens:] = avg_in
                out_emb[-num_part_tokens:] = avg_out
            print(f"Registered {_K} part vocab tokens: {part_tokens[:3]}... IDs={part_token_ids[:3]}...")

        if mm_use_point_start_end:
            default_point_start_token = config.DEFAULT_POINT_START_TOKEN
            default_point_end_token = config.DEFAULT_POINT_END_TOKEN
            point_backbone_config['default_point_start_token'] = default_point_start_token
            point_backbone_config['default_point_end_token'] = default_point_end_token

            num_new_tokens = tokenizer.add_tokens([default_point_start_token, default_point_end_token], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))
            point_backbone_config["point_start_token"] = tokenizer.convert_tokens_to_ids([default_point_start_token])[0]
            point_backbone_config["point_end_token"] = tokenizer.convert_tokens_to_ids([default_point_end_token])[0]

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

                # need to update the input embeding, but no need to update the output embedding
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                if fix_llm:
                    self.get_model().orig_embeds_params = [self.get_input_embeddings().weight.data.clone().to(device=device)] # * only tuning the new embeddings
                    for p in self.get_output_embeddings().parameters(): # * the llm head
                        p.requires_grad = False
                    print(f"Setting output embeddings fixed and {num_new_tokens} new tokens' input embeddings trainable.")
                else:
                    self.get_model().orig_embeds_params = None
                    for p in self.get_output_embeddings().parameters():
                        p.requires_grad = True
                    print("Setting output embeddings and all input embeddings trainable.")

AutoConfig.register("pointllm", PointLLMConfig)
AutoModelForCausalLM.register(PointLLMConfig, PointLLMLlamaForCausalLM)
