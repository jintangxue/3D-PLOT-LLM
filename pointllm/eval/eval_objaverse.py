import argparse
import torch
from torch.utils.data import DataLoader
import os
from pointllm.conversation import conv_templates, SeparatorStyle
from pointllm.utils import disable_torch_init
from pointllm.model import *
from pointllm.model.utils import KeywordsStoppingCriteria
from pointllm.data import ObjectPointCloudDataset
from pointllm.model.marker_context_utils import resolve_marker_context_mode
from tqdm import tqdm
from transformers import AutoTokenizer
# evaluator is imported lazily below (requires openai, only needed with --start_eval)

import os
import json

PROMPT_LISTS = [
    "What is this?",
    "This is an object of ",
    "Caption this 3D model in detail."
]

def init_model(args):
    # Model
    disable_torch_init()
    model_name = os.path.expanduser(args.model_name)

    # * print the model_name (get the basename)
    print(f'[INFO] Model name: {os.path.basename(model_name)}')

    tokenizer_name = getattr(args, 'tokenizer_name', None)
    if tokenizer_name is None:
        tokenizer_name = model_name

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=False)
    model = PointLLMLlamaForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=False, use_cache=True, torch_dtype=torch.bfloat16).cuda()
    model.initialize_tokenizer_point_backbone_config_wo_embedding(tokenizer)

    conv_mode = "vicuna_v1_1"

    conv = conv_templates[conv_mode].copy()

    return model, tokenizer, conv

def load_dataset(data_path, anno_path, pointnum, conversation_types, use_color, bcp_pack_dir=None, point_backbone_config=None):
    print("Loading validation datasets.")
    import argparse as _ap
    _data_args = _ap.Namespace(
        data_debug_num=0,
        split_train_val=False,
        bcp_pack_dir=bcp_pack_dir,
        bcp_reorder=False,
        bcp_part_interleave=point_backbone_config.get('bcp_part_interleave', False) if point_backbone_config else False,
        bcp_global_region_prefix=point_backbone_config.get('bcp_global_region_prefix', False) if point_backbone_config else False,
        bcp_part_vocab_token=point_backbone_config.get('bcp_part_vocab_token', False) if point_backbone_config else False,
        bcp_graph_proj_d=point_backbone_config.get('bcp_graph_proj_d', False) if point_backbone_config else False,
        bcp_marker_context_mode=point_backbone_config.get('bcp_marker_context_mode') if point_backbone_config else None,
        bcp_marker_context_stats=point_backbone_config.get('bcp_marker_context_stats', False) if point_backbone_config else False,
        bcp_marker_context_graph=point_backbone_config.get('bcp_marker_context_graph', False) if point_backbone_config else False,
        bcp_num_groups=point_backbone_config.get('bcp_num_groups', 16) if point_backbone_config else 16,
        point_backbone_config=point_backbone_config,
    )
    dataset = ObjectPointCloudDataset(
        data_path=data_path,
        anno_path=anno_path,
        pointnum=pointnum,
        conversation_types=conversation_types,
        use_color=use_color,
        tokenizer=None, # * load point cloud only
        bcp_pack_dir=bcp_pack_dir,
        data_args=_data_args,
    )
    print("Done!")
    return dataset

def get_dataloader(dataset, batch_size, shuffle=False, num_workers=4):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataloader

def generate_outputs(model, tokenizer, input_ids, point_clouds, stopping_criteria,
                     bcp_patch_feat=None, bcp_cls_feat=None, bcp_group_ids=None,
                     bcp_region_adj=None, bcp_region_stats=None,
                     do_sample=True, temperature=1.0, top_k=50, max_length=2048, top_p=0.95):
    model.eval()
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            point_clouds=point_clouds,
            bcp_patch_feat=bcp_patch_feat,
            bcp_cls_feat=bcp_cls_feat,
            bcp_group_ids=bcp_group_ids,
            bcp_region_adj=bcp_region_adj,
            bcp_region_stats=bcp_region_stats,
            do_sample=do_sample,
            temperature=temperature,
            top_k=top_k,
            max_length=max_length,
            top_p=top_p,
            stopping_criteria=[stopping_criteria]) # * B, L'

    input_token_len = input_ids.shape[1]
    n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
    if n_diff_input_output > 0:
        print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
    outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)
    outputs = [output.strip() for output in outputs]

    return outputs

def start_generation(model, tokenizer, conv, dataloader, annos, prompt_index, output_dir, output_file, bcp_pack_dir=None):
    # Reset the conversation state before appending. The same conv object is reused
    # across multiple runs (when --num_runs > 1), and conv.append_message
    # accumulates messages each call, which grows the prompt and breaks point
    # feature indexing. Resetting .messages = [] ensures each run starts with
    # a clean conv (only system prompt, template stays intact).
    conv.messages = []

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    qs = PROMPT_LISTS[prompt_index]

    results = {"prompt": qs}

    point_backbone_config = model.get_model().point_backbone_config
    point_token_len = point_backbone_config['point_token_len']

    # Adjust point_token_len to the interleaved region layout
    # Prevents IndexError when the true sequence of visual tokens differs from the saved config.
    bcp_num_groups = point_backbone_config.get('bcp_num_groups', 16)
    _vocab = point_backbone_config.get('bcp_part_vocab_token', False)
    _interleave = point_backbone_config.get('bcp_part_interleave', False)
    _graph = point_backbone_config.get('bcp_graph_proj_d', False) or point_backbone_config.get('bcp_part_graph_proj', False)
    _marker = point_backbone_config.get('bcp_part_marker', 'none') != 'none'
    _no_mean = point_backbone_config.get('bcp_part_no_mean', False)

    if _vocab and _interleave:
        # Hybrid path: CLS + K * (1[vocab] + (1[marker]?) + (1[mean/graph]?)) + 512
        tokens_per_grp = 1 + (1 if _marker else 0) + (0 if _no_mean else 1)
        point_token_len = 1 + (tokens_per_grp * bcp_num_groups) + 512
    elif _vocab:
        # Pure vocab path: CLS + (K or 2K[if graph]) + 512
        point_token_len = 1 + (2 * bcp_num_groups if _graph else bcp_num_groups) + 512
    elif _interleave:
        # Pure interleave path: CLS + K * ((1[marker]?) + (1[mean/graph]?)) + 512
        tokens_per_grp = (1 if _marker else 0) + (0 if _no_mean else 1)
        point_token_len = 1 + (tokens_per_grp * bcp_num_groups) + 512
    # ──────────────────────────────────────────────────────────────
    default_point_patch_token = point_backbone_config['default_point_patch_token']
    default_point_start_token = point_backbone_config['default_point_start_token']
    default_point_end_token = point_backbone_config['default_point_end_token']
    mm_use_point_start_end = point_backbone_config['mm_use_point_start_end']

    if mm_use_point_start_end:
        qs = default_point_start_token + default_point_patch_token * point_token_len + default_point_end_token + '\n' + qs
    else:
        qs = default_point_patch_token * point_token_len + '\n' + qs
    
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)

    prompt = conv.get_prompt()
    inputs = tokenizer([prompt])

    input_ids_ = torch.as_tensor(inputs.input_ids).cuda() # * tensor of 1, L

    stopping_criteria = KeywordsStoppingCriteria([stop_str], tokenizer, input_ids_)

    responses = []

    for batch in tqdm(dataloader):
        point_clouds = batch["point_clouds"].cuda().to(model.dtype) # * tensor of B, N, C(3)
        object_ids = batch["object_ids"] # * list of string

        # -- Offline BCP features (if pack dir was provided) --
        bcp_patch_feat = None
        bcp_cls_feat = None
        bcp_group_ids = None
        bcp_region_adj = None
        bcp_region_stats = None
        if bcp_pack_dir is not None:
            if 'bcp_patch_feat' in batch:
                bcp_patch_feat = batch['bcp_patch_feat'].cuda().to(model.dtype)  # (B, 512, 384)
            if 'bcp_cls_feat' in batch:
                bcp_cls_feat = batch['bcp_cls_feat'].cuda().to(model.dtype)      # (B, 384)
            if 'bcp_group_ids' in batch:
                bcp_group_ids = batch['bcp_group_ids'].cuda()
            if 'bcp_region_adj' in batch:
                bcp_region_adj = batch['bcp_region_adj'].cuda()
            if 'bcp_region_stats' in batch:
                bcp_region_stats = batch['bcp_region_stats'].cuda()
            
            if bcp_patch_feat is None or bcp_cls_feat is None:
                print('[Warning] bcp_pack_dir set but batch missing bcp features — falling back to online PointBERT')
            
            _mctx = resolve_marker_context_mode(
                point_backbone_config.get('bcp_marker_context_mode'),
                point_backbone_config.get('bcp_marker_context_stats', False),
                point_backbone_config.get('bcp_marker_context_graph', False),
            )
            _needs_region_context = (
                point_backbone_config.get('bcp_graph_proj_d', False)
                or _mctx != 'none'
            )
            if _needs_region_context:
                if bcp_region_adj is None or bcp_region_stats is None:
                    print('[Warning] Model uses region context but bcp_region_adj/stats are missing in batch! Context update will be skipped.')

        batchsize = len(object_ids)

        input_ids = input_ids_.repeat(batchsize, 1) # * tensor of B, L

        outputs = generate_outputs(model, tokenizer, input_ids, point_clouds, stopping_criteria,
                                   bcp_patch_feat=bcp_patch_feat, bcp_cls_feat=bcp_cls_feat, bcp_group_ids=bcp_group_ids,
                                   bcp_region_adj=bcp_region_adj, bcp_region_stats=bcp_region_stats) # List of str, length is B

        # saving results
        for obj_id, output in zip(object_ids, outputs):
            responses.append({
                "object_id": obj_id,
                "ground_truth": annos[obj_id],
                "model_output": output
            })
    
    results["results"] = responses

    os.makedirs(output_dir, exist_ok=True)
    # save the results to a JSON file
    with open(os.path.join(output_dir, output_file), 'w') as fp:
        json.dump(results, fp, indent=2)

    # * print info
    print(f"Saved results to {os.path.join(output_dir, output_file)}")

    return results

def main(args):
    # * output dir
    args.output_dir = os.path.join(args.model_name, "evaluation")

    anno_file = os.path.splitext(os.path.basename(args.anno_path))[0]
    base_prefix = f"{anno_file}_Objaverse_{args.task_type}_prompt{args.prompt_index}"

    # Determine list of run IDs to execute. num_runs=0 or 1 with run_start=1 keeps
    # single-file behavior (no _run{i} suffix). Otherwise each run writes to
    # "{base_prefix}_run{i}.json" and we skip runs whose output already exists.
    num_runs = max(1, int(getattr(args, "num_runs", 1)))
    run_start = int(getattr(args, "run_start", 1))
    legacy_single = (num_runs == 1 and run_start == 1 and not getattr(args, "force_run_suffix", False))

    if legacy_single:
        run_specs = [(None, f"{base_prefix}.json")]
    else:
        run_specs = [(i, f"{base_prefix}_run{i}.json") for i in range(run_start, run_start + num_runs)]

    # Skip runs whose output already exists — useful for resuming.
    pending = [(rid, fname) for rid, fname in run_specs
               if not os.path.exists(os.path.join(args.output_dir, fname))]
    if not pending:
        print(f"[INFO] All {len(run_specs)} runs already exist, nothing to do.")
        last_fname = run_specs[-1][1]
        args.output_file = last_fname
        args.output_file_path = os.path.join(args.output_dir, last_fname)
        with open(args.output_file_path, "r") as fp:
            results = json.load(fp)
    else:
        # * load shared resources once
        with open(args.anno_path, "r") as fp:
            annos_raw = json.load(fp)
        bcp_pack_dir = getattr(args, "bcp_pack_dir", None)
        model, tokenizer, conv = init_model(args)

        dataset = load_dataset(args.data_path, args.anno_path, args.pointnum, ("simple_description",), args.use_color,
                               bcp_pack_dir=bcp_pack_dir, point_backbone_config=model.get_model().point_backbone_config)
        dataloader = get_dataloader(dataset, args.batch_size, args.shuffle, args.num_workers)

        annos = {anno["object_id"]: anno["conversations"][1]["value"] for anno in annos_raw}

        if bcp_pack_dir:
            print(f"[INFO] Using offline BCP features from: {bcp_pack_dir}")

        results = None
        for rid, fname in pending:
            tag = f"run {rid}" if rid is not None else "single run"
            print(f"[INFO] Start generating results for {fname} ({tag}).")
            results = start_generation(
                model, tokenizer, conv, dataloader, annos, args.prompt_index,
                args.output_dir, fname, bcp_pack_dir=bcp_pack_dir,
            )

        # Also set args.output_file / args.output_file_path for downstream GPT eval
        last_rid, last_fname = pending[-1]
        args.output_file = last_fname
        args.output_file_path = os.path.join(args.output_dir, last_fname)

        del model
        del tokenizer
        torch.cuda.empty_cache()

    if args.start_eval:
        from pointllm.eval.evaluator import start_evaluation
        evaluated_output_file = args.output_file.replace(".json", f"_evaluated_{args.gpt_type}.json")
        eval_type_mapping = {
            "captioning": "object-captioning",
            "classification": "open-free-form-classification"
        }
        start_evaluation(results, output_dir=args.output_dir, output_file=evaluated_output_file, eval_type=eval_type_mapping[args.task_type], model_type=args.gpt_type, parallel=True, num_workers=20)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, \
        default="RunsenXu/PointLLM_7B_v1.2") 
    parser.add_argument("--tokenizer_name", type=str, default=None,
                        help="Path to tokenizer, if different from model_name.")

    # * dataset type
    parser.add_argument("--data_path", type=str, default="data/objaverse_data", required=False)
    parser.add_argument("--anno_path", type=str, default="data/anno_data/PointLLM_brief_description_val_200_GT.json", required=False)
    parser.add_argument("--pointnum", type=int, default=8192)
    parser.add_argument("--bcp_pack_dir", type=str, default=None,
                        help="Path to the partition pack directory. If set, uses offline features (matching training).")
    parser.add_argument("--use_color",  action="store_true", default=True)

    # * data loader, batch_size, shuffle, num_workers
    parser.add_argument("--batch_size", type=int, default=6)
    parser.add_argument("--shuffle", type=bool, default=False)
    parser.add_argument("--num_workers", type=int, default=10)

    # * multi-run support (load model once, do N sampling passes)
    parser.add_argument("--num_runs", type=int, default=1,
                        help="Number of inference passes with the same loaded model. Each pass writes to ..._run{i}.json.")
    parser.add_argument("--run_start", type=int, default=1,
                        help="Starting run id (inclusive). Runs from run_start to run_start+num_runs-1.")
    parser.add_argument("--force_run_suffix", action="store_true", default=False,
                        help="Use ..._run{i}.json suffix even when num_runs=1 (for single-run extensions of a multi-run series).")

    # * evaluation setting
    parser.add_argument("--prompt_index", type=int, default=0)
    parser.add_argument("--start_eval", action="store_true", default=False)
    parser.add_argument("--gpt_type", type=str, default="gpt-4-0613", choices=["gpt-3.5-turbo-0613", "gpt-3.5-turbo-1106", "gpt-4-0613", "gpt-4-1106-preview"], help="Type of the model used to evaluate.")
    parser.add_argument("--task_type", type=str, default="captioning", choices=["captioning", "classification"], help="Type of the task to evaluate.")

    args = parser.parse_args()

    # * check prompt index
    # * * classification: 0, 1 and captioning: 2. Raise Warning otherwise.
    if args.task_type == "classification":
        if args.prompt_index != 0 and args.prompt_index != 1:
            print("[Warning] For classification task, prompt_index should be 0 or 1.")
    elif args.task_type == "captioning":
        if args.prompt_index != 2:
            print("[Warning] For captioning task, prompt_index should be 2.")
    else:
        raise NotImplementedError

    main(args)