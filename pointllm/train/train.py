#  Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

from dataclasses import dataclass, field
import pathlib
from typing import Optional, List


import transformers
from pointllm.train.pointllm_trainer import PointLLMTrainer

from pointllm import conversation as conversation_lib
from pointllm.model import *
from pointllm.data import make_object_point_data_module
from pointllm.model.marker_context_utils import resolve_marker_context_mode

# * logger
from pointllm.utils import build_logger

IGNORE_INDEX = -100

DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "</s>"
DEFAULT_UNK_TOKEN = "<unk>"


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="")
    version: Optional[str] = field(default="v1")

@dataclass
class DataArguments:
    data_path: str = field(default="ScanNet", metadata={"help": "Path to the training data."})
    anno_path: str = field(default=None, metadata={"help": "Path to the utterance data. If None, will use referit3d by defautl."})
    use_color: bool = field(default=False, metadata={"help": "Whether to use color."})
    data_debug_num: int = field(default=0, metadata={"help": "Number of data to use in debug mode. If larger than 0, use debug mode, else use the whole data"})
    split_train_val: bool = field(default=False, metadata={"help": "Whether to split train and val."})
    split_ratio: float = field(default=0.9, metadata={"help": "Ratio of train and val."})
    pointnum: int = field(default=8192, metadata={"help": "Number of points."})
    conversation_types: List[str] = field(default_factory=lambda: ["simple_description"], metadata={"help": "Conversation types to use."})
    is_multimodal: bool = True
    bcp_pack_dir: Optional[str] = field(default=None, metadata={"help": "Path to the partition packs dir. If set, offline patch_feat and patch_r16 are loaded. Samples without a pack are filtered out."})
    bcp_reorder: bool = field(default=False, metadata={"help": "If True and bcp_pack_dir is set, reorder patch tokens by BCP K=16 group structure before projector."})
    bcp_kmeans_reorder: bool = field(default=False, metadata={"help": "If True, reorder patch tokens by offline K-Means spatial grouping."})
    bcp_part_interleave: bool = field(default=False, metadata={"help": """
        If True, assemble 529-token part-interleaved sequence:
        [CLS, g0_mean, g0_patches(32), g1_mean, ..., g15_mean, g15_patches(32)].
        Requires partition packs with cls_feat, patch_feat and group_ids.
        Offline-only: does not work with the live PointBERT path.
        Automatically loads bcp_group_ids from dataset (independent of bcp_reorder).
    """})
    bcp_part_marker: str = field(default="none", metadata={"help": """
        Learnable marker token inserted before each part's header in the interleaved layout.
        Only active when bcp_part_interleave=True.
        'none'     : Version A — no marker, 529 tokens.
        'shared'   : Version B — one shared marker for all 16 parts, 545 tokens.
        'per_part' : Version C — 16 independent markers (one per group), 545 tokens.
    """})
    bcp_part_no_mean: bool = field(default=False, metadata={"help": """
        If True, do not include the g_mean token when bcp_part_interleave is True.
    """})
    bcp_part_projected_mean: bool = field(default=False, metadata={"help": """
        If True with bcp_part_interleave, use projected region summary token:
        header = MLP(mean(group_tokens)) instead of raw mean(group_tokens).
    """})
    bcp_global_region_prefix: bool = field(default=False, metadata={"help": """
        If True, use global region-prefix layout:
        [CLS, R0..R15, grouped_patch_tokens]
        where grouped_patch_tokens preserve local continuity within each regrouped segment.
    """})
    bcp_graph_proj_d: bool = field(default=False, metadata={"help": """
        If True, use Graph-Proj-D/E adapter instead of independent region MLP.
        It calculates graph adjacency based on offline pack adjacency and statistics.
    """})
    bcp_part_vocab_token: bool = field(default=False, metadata={"help": """
        If True, use real LLM vocabulary tokens <part_0>..<part_K-1> as structural
        separators instead of learnable nn.Parameter markers.
        Layout: [CLS, <part_0>, G0_patches, ..., <part_K-1>, GK-1_patches] (1+K+512 tokens).
        Requires bcp_pack_dir with cls_feat + group_ids.
    """})
    bcp_marker_context_mode: Optional[str] = field(default=None, metadata={"help": """
        Mutually exclusive marker context: none | stats_only | graph_only | stats_graph.
        Leave unset (None) to use legacy bcp_marker_context_stats / bcp_marker_context_graph bools.
    """})
    bcp_marker_context_stats: bool = field(default=False, metadata={"help": """
        LEGACY: used only when bcp_marker_context_mode is none — maps to stats_only.
    """})
    bcp_marker_context_graph: bool = field(default=False, metadata={"help": """
        LEGACY: used only when bcp_marker_context_mode is none — maps to graph_only / stats_graph.
    """})
    bcp_num_groups: int = field(default=16, metadata={"help": "Number of groups for BCP reordering/interleaving (e.g. 16 or 8)."})
    bcp_graph_order_bfs: bool = field(default=False, metadata={"help": """
        If True, remap group_ids using BFS traversal of the region adjacency graph.
        Spatially adjacent regions become sequential in the token sequence.
        Zero parameters, zero extra tokens. Compatible with all assembly strategies.
        Requires bcp_pack_dir with patch_adj_edges (region_adj is computed on the fly).
    """})
    bcp_vocab_graph_propagation: bool = field(default=False, metadata={"help": """
        LEGACY (fused path): If True, apply graph message-passing on vocab token embeddings in LLM space
        (4096-dim) using the region adjacency graph. Avoids point_proj bottleneck.
        Requires bcp_part_vocab_token=True and bcp_pack_dir with adjacency data.
        Mutually exclusive with bcp_vocab_context_mode.
    """})
    bcp_vocab_context_mode: Optional[str] = field(default=None, metadata={"help": """
        NEW decoupled path (mirrors bcp_marker_context_mode at the LLM vocab layer):
        none | stats_only | graph_only | stats_graph. When non-none, replaces the
        fused bcp_vocab_graph_propagation with two-residual (stats MLP + graph)
        symmetric to MSR. Mutually exclusive with bcp_vocab_graph_propagation.
    """})
    bcp_zero_adj: bool = field(default=False, metadata={"help": """
        Ablation: zero out region_adj during training. Used to test whether the
        adjacency-conditioned branch of VocabGraphPropagator is actually necessary.
        If model trained with bcp_zero_adj=True matches vocab_graph in performance,
        adjacency topology is not critical (self + stats branches dominate).
    """})

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    # * can refer to https://huggingface.co/docs/transformers/v4.28.1/en/main_classes/trainer#transformers.TrainingArgument
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=2048,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    model_debug: bool = field(default=False, metadata={"help": "Whether to use small model."}) # * whether to load checkpoints at the mo
    fix_llm: bool = field(default=True, metadata={"help": "Whether to fix the LLM."})
    fix_pointnet: bool = field(default=True, metadata={"help": "Whether to fix the PointNet."})

    remove_unused_columns: bool = field(default=False)
    force_fsdp: bool = field(default=False)

    # * for two stage training
    tune_mm_mlp_adapter: bool = field(default=True) # * set True when pre-training, and false when fine-tuning
    stage_2: bool = field(default=False) # * set True when fine-tuning
    pretrained_mm_mlp_adapter: Optional[str] = field(default=None) # * path to the pre-trained projector & output_embed & input_embed
    detatch_point_token: bool = field(default=False) # * deprecated
    # * point backbone ckpt path
    point_backbone_ckpt: str = field(default=None)

def safe_save_model_for_hf_trainer(trainer: transformers.Trainer,
                                   output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


class SaveFinalStepCallback(transformers.TrainerCallback):
    """
    Forces a clean standard checkpoint save (e.g., checkpoint-13176) at the exact final step.
    This bypasses the fragile end-of-training `save_model` root directory write
    that often fails or corrupts weights in ZeRO-3.
    """
    def on_step_end(self, args, state, control, **kwargs):
        # Trigger save if we exactly hit max_steps (but avoid saving twice if save_steps already triggered)
        if state.global_step >= state.max_steps and state.global_step % args.save_steps != 0:
            control.should_save = True
        return control


def train():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    _marker_ctx_mode = resolve_marker_context_mode(
        getattr(data_args, "bcp_marker_context_mode", None),
        getattr(data_args, "bcp_marker_context_stats", False),
        getattr(data_args, "bcp_marker_context_graph", False),
    )
    data_args.bcp_marker_context_mode = _marker_ctx_mode

    _vocab_ctx_mode = resolve_marker_context_mode(
        getattr(data_args, "bcp_vocab_context_mode", None), False, False,
    )
    data_args.bcp_vocab_context_mode = _vocab_ctx_mode
    if _vocab_ctx_mode != "none" and getattr(data_args, "bcp_vocab_graph_propagation", False):
        raise ValueError(
            "bcp_vocab_context_mode and bcp_vocab_graph_propagation are mutually exclusive; set only one."
        )

    training_args.log_level = "info" # * default is passive(warning)
    # * build logger
    logger = build_logger(__name__, training_args.output_dir + '/train.log')

    if training_args.model_debug:
        # * do not load checkpoint, load from config
        config = transformers.AutoConfig.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
            )
        model = PointLLMLlamaForCausalLM._from_config(config)
    else:
        model, loading_info = PointLLMLlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            output_loading_info=True
        )
        logger.info(f"Model loaded from {model_args.model_name_or_path}")
        logger.info(f"Loading info: {loading_info}")
        if loading_info['missing_keys']:
            logger.warning(f"Missing keys: {loading_info['missing_keys']}")
        if loading_info['unexpected_keys']:
            logger.warning(f"Unexpected keys: {loading_info['unexpected_keys']}")

    model.config.use_cache = False
    # Persist projected-mean mode into config for future stage2/inference loading.
    setattr(model.config, 'bcp_part_projected_mean', getattr(data_args, 'bcp_part_projected_mean', False))
    setattr(model.config, 'bcp_graph_proj_d', getattr(data_args, 'bcp_graph_proj_d', False))
    setattr(model.config, 'bcp_part_vocab_token', getattr(data_args, 'bcp_part_vocab_token', False))
    setattr(model.config, 'bcp_part_interleave', getattr(data_args, 'bcp_part_interleave', False))
    setattr(model.config, 'bcp_part_marker', getattr(data_args, 'bcp_part_marker', 'none'))
    setattr(model.config, 'bcp_part_no_mean', getattr(data_args, 'bcp_part_no_mean', False))
    setattr(model.config, 'bcp_marker_context_mode', _marker_ctx_mode)
    setattr(model.config, 'bcp_marker_context_stats', _marker_ctx_mode in ('stats_only', 'stats_graph'))
    setattr(model.config, 'bcp_marker_context_graph', _marker_ctx_mode in ('graph_only', 'stats_graph'))
    setattr(model.config, 'bcp_vocab_context_mode', _vocab_ctx_mode)

    # Ensure learnable adapter components are constructed even if fix_llm=False
    if getattr(data_args, 'bcp_part_projected_mean', False):
        model.get_model().enable_region_summary_mlp()
    if getattr(data_args, 'bcp_graph_proj_d', False):
        model.get_model().enable_region_graph_adapter()
    if _marker_ctx_mode in ('stats_only', 'stats_graph'):
        model.get_model().enable_marker_stats_mlp()
    if _marker_ctx_mode in ('graph_only', 'stats_graph'):
        model.get_model().enable_marker_graph_propagator()
    if getattr(data_args, 'bcp_vocab_graph_propagation', False):
        model.get_model().enable_vocab_graph_propagator()
    if _vocab_ctx_mode in ('stats_only', 'stats_graph'):
        model.get_model().enable_vocab_stats_mlp()
    if _vocab_ctx_mode in ('graph_only', 'stats_graph'):
        model.get_model().enable_vocab_graph_propagator_decoupled()

    if training_args.fix_llm:
        # * This will fix all the parameters
        logger.info("LLM is fixed. Fix_llm flag is set to True")
        # * fix llama, lm_head, pointnet, projection layer here
        model.requires_grad_(False)
        model.get_model().fix_llm = True
        model.get_model().point_proj.requires_grad_(True)
        model.get_model().point_backbone.requires_grad_(True) # * set as True for fsdp, use fix_pointnet flag to control
        # Unfreeze learnable part markers if they will be used
        _marker_mode = getattr(data_args, 'bcp_part_marker', 'none')
        if _marker_mode == 'shared':
            model.get_model().part_marker.requires_grad_(True)
        elif _marker_mode == 'per_part':
            model.get_model().part_markers.requires_grad_(True)
        if getattr(data_args, 'bcp_part_projected_mean', False):
            model.get_model().region_summary_mlp.requires_grad_(True)
        if getattr(data_args, 'bcp_graph_proj_d', False):
            model.get_model().region_graph_adapter.requires_grad_(True)
        if _marker_ctx_mode in ('stats_only', 'stats_graph'):
            model.get_model().marker_stats_mlp.requires_grad_(True)
            model.get_model().marker_stats_scale.requires_grad_(True)
        if _marker_ctx_mode in ('graph_only', 'stats_graph') and model.get_model().marker_graph_propagator is not None:
            model.get_model().marker_graph_propagator.requires_grad_(True)
            model.get_model().marker_graph_scale.requires_grad_(True)
        if getattr(data_args, 'bcp_vocab_graph_propagation', False) and model.get_model().vocab_graph_propagator is not None:
            model.get_model().vocab_graph_propagator.requires_grad_(True)
            model.get_model().vocab_graph_scale.requires_grad_(True)
        if _vocab_ctx_mode in ('stats_only', 'stats_graph') and model.get_model().vocab_stats_mlp is not None:
            model.get_model().vocab_stats_mlp.requires_grad_(True)
            model.get_model().vocab_stats_scale.requires_grad_(True)
        if _vocab_ctx_mode in ('graph_only', 'stats_graph') and model.get_model().vocab_graph_propagator_v2 is not None:
            model.get_model().vocab_graph_propagator_v2.requires_grad_(True)
            model.get_model().vocab_graph_scale.requires_grad_(True)
    else:
        model.get_model().fix_llm = False
        # When fix_llm=False, model.requires_grad_(True) applies globally (including adapter if it was unconditionally enabled).
        logger.warning("LLM is trainable. Fix_llm flag is set to False")

    # Sync bcp flags into model config
    model.config.bcp_num_groups = getattr(data_args, 'bcp_num_groups', 16)
    model.config.bcp_graph_order_bfs = getattr(data_args, 'bcp_graph_order_bfs', False)
    model.config.bcp_vocab_graph_propagation = getattr(data_args, 'bcp_vocab_graph_propagation', False)
    model.config.bcp_vocab_context_mode = _vocab_ctx_mode

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    if model_args.version == "v0" or "v0" in model_args.model_name_or_path:
        raise ValueError("v0 is deprecated.")
    else:
        tokenizer.pad_token = tokenizer.unk_token
        conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1_1"]

    if not training_args.fix_pointnet:
        # * not fix pointnet
        logger.info("Point backbone is trainable. Fix_pointnet flag is set to False, pointnet grad will be recorded.")
        model.get_model().fix_pointnet = False
    else:
        logger.info("Point backbone is fixed. Fix_pointnet flag is set to True, pointnet grad will not be recorded.")
        model.get_model().fix_pointnet = True # * use with torch.inference_mode to control, not requires_grad for fsdp for second stage
        if not training_args.stage_2:
            logger.info("Set requires_grad of point backbone to False")
            model.get_model().point_backbone.requires_grad_(False) # * fix pointnet for first stage, need for fsdp in stage2
    
    if training_args.tune_mm_mlp_adapter:
        # * not fix the projection layer
        # * may need to set the embed_tokens to require_grad = True if added new tokens
        # * this is done in initialize_tokenizer_point_backbone_config
        logger.info("Point projection layer is trainable.")
    else:
        model.get_model().point_proj.requires_grad_(False)
        logger.info("Point prejcetion layer is fixed.")

    if not training_args.stage_2:
        # * we assume in stage2, llm, point_backbone, and projection layer can be loaded from the model checkpoint
        print(f"Default point_backbone_ckpt is {training_args.point_backbone_ckpt}.")
        model.get_model().load_point_backbone_checkpoint(training_args.point_backbone_ckpt)
        model.initialize_tokenizer_point_backbone_config(tokenizer=tokenizer, device=training_args.device, fix_llm=training_args.fix_llm)
    else:
        # * stage2
        model.initialize_tokenizer_point_backbone_config_wo_embedding(tokenizer=tokenizer) 

    point_backbone_config = model.get_model().point_backbone_config

    data_args.point_token_len = point_backbone_config['point_token_len']
    data_args.mm_use_point_start_end = point_backbone_config['mm_use_point_start_end']
    data_args.point_backbone_config = point_backbone_config

    # BCP structured token layouts: override token count and store flags in model config.
    # Must happen AFTER point_backbone_config is set so preprocess_multimodal_point_cloud
    # picks up the correct placeholder count.
    _part_interleave = getattr(data_args, 'bcp_part_interleave', False)
    _global_prefix = getattr(data_args, 'bcp_global_region_prefix', False)
    _part_vocab = getattr(data_args, 'bcp_part_vocab_token', False)
    if _part_interleave and _global_prefix:
        raise ValueError("bcp_part_interleave and bcp_global_region_prefix cannot both be True.")
    # NOTE: interleave + vocab is now supported (hybrid path in model.forward).
    if _global_prefix and _part_vocab:
        raise ValueError(
            "bcp_global_region_prefix and bcp_part_vocab_token cannot both be True "
            "(same train vs forward priority mismatch)."
        )

    if _part_interleave and _part_vocab:
        _marker_mode = getattr(data_args, 'bcp_part_marker', 'none')
        _no_mean = getattr(data_args, 'bcp_part_no_mean', False)
        _projected_mean = getattr(data_args, 'bcp_part_projected_mean', False)
        _num_groups = getattr(data_args, 'bcp_num_groups', 16)
        if _no_mean and _projected_mean:
            raise ValueError("bcp_part_projected_mean requires mean tokens; cannot be used with bcp_part_no_mean=True.")
        has_marker = _marker_mode in ('shared', 'per_part')
        # Base interleave: 1 + K*(marker? + header?) + 512
        base_interleave_len = 1 + _num_groups * ((1 if has_marker else 0) + (0 if _no_mean else 1)) + 512
        # Hybrid adds one vocab token <part_g> per group.
        part_token_len = base_interleave_len + _num_groups
        point_backbone_config['point_token_len'] = part_token_len
        point_backbone_config['bcp_part_interleave'] = True
        point_backbone_config['bcp_part_vocab_token'] = True
        point_backbone_config['bcp_part_marker'] = _marker_mode
        point_backbone_config['bcp_part_no_mean'] = _no_mean
        point_backbone_config['bcp_part_projected_mean'] = _projected_mean
        point_backbone_config['bcp_graph_proj_d'] = getattr(data_args, 'bcp_graph_proj_d', False)
        point_backbone_config['bcp_marker_context_mode'] = _marker_ctx_mode
        point_backbone_config['bcp_marker_context_stats'] = _marker_ctx_mode in ('stats_only', 'stats_graph')
        point_backbone_config['bcp_marker_context_graph'] = _marker_ctx_mode in ('graph_only', 'stats_graph')
        point_backbone_config['bcp_num_groups'] = _num_groups
        data_args.point_token_len = part_token_len
    elif _part_interleave:
        _marker_mode = getattr(data_args, 'bcp_part_marker', 'none')
        _no_mean = getattr(data_args, 'bcp_part_no_mean', False)
        _projected_mean = getattr(data_args, 'bcp_part_projected_mean', False)
        _num_groups = getattr(data_args, 'bcp_num_groups', 16)
        if _no_mean and _projected_mean:
            raise ValueError("bcp_part_projected_mean requires mean tokens; cannot be used with bcp_part_no_mean=True.")
        has_marker = _marker_mode in ('shared', 'per_part')
        # Calculate tokens depending on marker and no_mean (total patches is always 512)
        part_token_len = 1 + _num_groups * ((1 if has_marker else 0) + (0 if _no_mean else 1)) + 512
        point_backbone_config['point_token_len'] = part_token_len
        point_backbone_config['bcp_part_interleave'] = True
        point_backbone_config['bcp_part_marker'] = _marker_mode
        point_backbone_config['bcp_part_no_mean'] = _no_mean
        point_backbone_config['bcp_part_projected_mean'] = _projected_mean
        point_backbone_config['bcp_graph_proj_d'] = getattr(data_args, 'bcp_graph_proj_d', False)
        point_backbone_config['bcp_marker_context_mode'] = _marker_ctx_mode
        point_backbone_config['bcp_marker_context_stats'] = _marker_ctx_mode in ('stats_only', 'stats_graph')
        point_backbone_config['bcp_marker_context_graph'] = _marker_ctx_mode in ('graph_only', 'stats_graph')
        point_backbone_config['bcp_num_groups'] = _num_groups
        data_args.point_token_len = part_token_len
    elif _global_prefix:
        _num_groups = getattr(data_args, 'bcp_num_groups', 16)
        point_backbone_config['point_token_len'] = 1 + _num_groups + 512
        point_backbone_config['bcp_global_region_prefix'] = True
        point_backbone_config['bcp_graph_proj_d'] = getattr(data_args, 'bcp_graph_proj_d', False)
        point_backbone_config['bcp_num_groups'] = _num_groups
        data_args.point_token_len = point_backbone_config['point_token_len']
    elif _part_vocab:
        _num_groups = getattr(data_args, 'bcp_num_groups', 16)
        _with_graph = getattr(data_args, 'bcp_graph_proj_d', False)
        if _with_graph:
            # Layout: [CLS, <part_g>, graph_mean_g, G_g_patches, ...]
            # Total:  1 + 2K + 512
            part_token_len = 1 + 2 * _num_groups + 512
        else:
            # Layout: [CLS, <part_g>, G_g_patches, ...]
            # Total:  1 + K + 512
            part_token_len = 1 + _num_groups + 512
        point_backbone_config['point_token_len'] = part_token_len
        point_backbone_config['bcp_part_vocab_token'] = True
        point_backbone_config['bcp_graph_proj_d'] = _with_graph
        point_backbone_config['bcp_num_groups'] = _num_groups
        data_args.point_token_len = part_token_len


    params_no_grad = [n for n, p in model.named_parameters() if not p.requires_grad]
    if len(params_no_grad) > 0:
        if training_args.fsdp is not None and len(training_args.fsdp) > 0:
            if len(params_no_grad) < 10:
                print('[WARNING] Attempting to use FSDP while {} parameters do not require gradients: {}'. format(len(params_no_grad), params_no_grad))
            else:
                print('[WARNING] Attempting to use FSDP while {} parameters do not require gradients: {}...(omitted)'. format(len(params_no_grad), ', '.join(params_no_grad[:10])))
            print("[WARNING] Attempting to use FSDP with partially frozen paramters, this is experimental.")
            print("[WARNING] As of 4/30/23, this feature requires PyTorch-nightly build.  See here for details: https://github.com/haotian-liu/LLaVA#experimental-use-fsdp-to-save-memory-in-pretraining")

            from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP
            def patch_FSDP_use_orig_params(func):
                def wrap_func(*args, **kwargs):
                    use_orig_params = kwargs.pop('use_orig_params', True)
                    return func(*args, **kwargs, use_orig_params=use_orig_params)
                return wrap_func

            FSDP.__init__ = patch_FSDP_use_orig_params(FSDP.__init__)

    data_module = make_object_point_data_module(tokenizer=tokenizer,
                                                    data_args=data_args)

    trainer = PointLLMTrainer(model=model,
                    tokenizer=tokenizer,
                    args=training_args,
                    callbacks=[SaveFinalStepCallback()],
                    **data_module)

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()
    safe_save_model_for_hf_trainer(trainer=trainer,
                                   output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
