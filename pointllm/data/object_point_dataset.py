import os
import json
import torch
import numpy as np

import copy
import transformers
from torch.utils.data import Dataset

from .utils import *
from pointllm.model.marker_context_utils import resolve_marker_context_mode


def make_object_point_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args) -> Dict:
    """Make dataset and collator for Joint3Ddataset with text and point cloud data."""
    """Initialize datasets."""

    data_collator = DataCollatorForPointTextDataset(tokenizer=tokenizer)
    bcp_pack_dir = getattr(data_args, 'bcp_pack_dir', None)
    if data_args.split_train_val:
        print("Loading training datasets.")
        train_dataset = ObjectPointCloudDataset(
            split='train',
            data_path=data_args.data_path,
            anno_path=data_args.anno_path,
            pointnum=data_args.pointnum,
            conversation_types=data_args.conversation_types,
            tokenizer=tokenizer,
            use_color=data_args.use_color,
            bcp_pack_dir=bcp_pack_dir,
            data_args=data_args
        )
        print("Done!")
        if data_args.data_debug_num > 0:
            print('Debug mode, using training set as val set.')
            val_dataset = train_dataset
        else:
            # * make a val dataset
            print("Loading validation datasets.")
            val_dataset = ObjectPointCloudDataset(
                split='val', # * load train split
                data_path=data_args.data_path,
                anno_path=data_args.anno_path,
                pointnum=data_args.pointnum,
                conversation_types=data_args.conversation_types,
                tokenizer=tokenizer,
                use_color=data_args.use_color,
                bcp_pack_dir=bcp_pack_dir,
                data_args=data_args
            )
        return dict(train_dataset=train_dataset, eval_dataset=val_dataset, data_collator=data_collator)
    else:
        # * use all data as training data
        train_dataset = ObjectPointCloudDataset(
            split='train',
            data_path=data_args.data_path,
            anno_path=data_args.anno_path,
            pointnum=data_args.pointnum,
            conversation_types=data_args.conversation_types,
            use_color=data_args.use_color,
            tokenizer=tokenizer,
            bcp_pack_dir=bcp_pack_dir,
            data_args=data_args
        )
        return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)

class ObjectPointCloudDataset(Dataset):
    """Dataset utilities for objaverse."""
    def __init__(self,
                 data_path=None,
                 anno_path=None,
                 tokenizer=None,
                 pointnum=8192,
                 split='train',
                 conversation_types=None, # * default is simple_des, used for stage1 pre-train
                 use_color=True,
                 bcp_pack_dir=None,       # * path to bcp_packs_lite dir; if set, samples without pack are filtered
                 data_args=None):

        """
        split: only considered when data_args.split_train_val is True.
        conversation_types: tuple, used to filter the data, default is ('simple_description'), other types is:
            "detailed_description", "single_round", "multi_round".
        tokenizer: load point clouds only if None
        """
        super(ObjectPointCloudDataset, self).__init__()

        """Initialize dataset with object point clouds and text"""
        self.data_path = data_path
        self.anno_path = anno_path
        self.tokenizer = tokenizer
        self.split = split 
        if conversation_types is None:
            self.conversation_types = ("simple_description",)
        else:
            self.conversation_types = conversation_types

        self.data_args = data_args
        self.normalize_pc = True
        self.use_color = use_color
        self.bcp_pack_dir = bcp_pack_dir
        self.bcp_reorder = getattr(data_args, 'bcp_reorder', False) if data_args is not None else False
        self.bcp_kmeans_reorder = getattr(data_args, 'bcp_kmeans_reorder', False) if data_args is not None else False
        self.bcp_part_interleave = getattr(data_args, 'bcp_part_interleave', False) if data_args is not None else False
        self.bcp_global_region_prefix = getattr(data_args, 'bcp_global_region_prefix', False) if data_args is not None else False
        self.bcp_num_groups = getattr(data_args, 'bcp_num_groups', 16) if data_args is not None else 16
        self.bcp_part_vocab_token = getattr(data_args, 'bcp_part_vocab_token', False) if data_args is not None else False
        self.bcp_graph_order_bfs = getattr(data_args, 'bcp_graph_order_bfs', False) if data_args is not None else False
        # Ablation: zero out the adjacency matrix during training to test whether
        # VocabGraphPropagator's 'agg' branch actually contributes beyond self/stats branches.
        self.bcp_zero_adj = getattr(data_args, 'bcp_zero_adj', False) if data_args is not None else False
        self.bcp_marker_context_mode = resolve_marker_context_mode(
            getattr(data_args, 'bcp_marker_context_mode', None) if data_args is not None else None,
            getattr(data_args, 'bcp_marker_context_stats', False) if data_args is not None else False,
            getattr(data_args, 'bcp_marker_context_graph', False) if data_args is not None else False,
        )
        
        assert not (self.bcp_reorder and self.bcp_kmeans_reorder), "Cannot use both bcp_reorder and bcp_kmeans_reorder simultaneously."
        
        # group_ids are needed for reorder, kmeans_reorder, part-interleave, global-prefix, part_vocab_token, and graph (graph uses them too)
        self._need_group_ids = (
            self.bcp_reorder
            or self.bcp_kmeans_reorder
            or self.bcp_part_interleave
            or self.bcp_global_region_prefix
            or self.bcp_part_vocab_token
        )
        self.bcp_graph_proj = getattr(data_args, 'bcp_graph_proj_d', False) if data_args is not None else False
        self.bcp_vocab_graph_propagation = getattr(data_args, 'bcp_vocab_graph_propagation', False) if data_args is not None else False
        self.bcp_need_region_context = (
            self.bcp_graph_proj
            or (self.bcp_marker_context_mode != 'none')
            or self.bcp_graph_order_bfs
            or self.bcp_vocab_graph_propagation
        )


        self.pointnum = pointnum
        self.point_backbone_config = data_args.point_backbone_config if data_args is not None else None
        self.point_indicator = '<point>'

        # Load the data list from JSON
        print(f"Loading anno file from {anno_path}.")
        with open(anno_path, "r") as json_file:
            self.list_data_dict = json.load(json_file)
        
        # * print the conversations_type
        print(f"Using conversation_type: {self.conversation_types}") 
        # * print before filtering
        print(f"Before filtering, the dataset size is: {len(self.list_data_dict)}.")

        # * iterate the list and filter
        # * these two ids have corrupted colored point files, so filter them when use_color is True
        filter_ids = ['6760e543e1d645d5aaacd3803bcae524', 'b91c0711149d460a8004f9c06d3b7f38'] if self.use_color else []

        # Iterate the list, filter those "conversation_type" not in self.conversation_types
        self.list_data_dict = [
            data for data in self.list_data_dict 
            if data.get('conversation_type', 'simple_description') in self.conversation_types 
            and data.get('object_id') not in filter_ids
        ]

        # * print after filtering
        print(f"After filtering, the dataset size is: {len(self.list_data_dict)}.")
        # * print the size of different conversation_type
        for conversation_type in self.conversation_types:
            print(f"Number of {conversation_type}: {len([data for data in self.list_data_dict if data.get('conversation_type', 'simple_description') == conversation_type])}")

        # * BCP pack filter: remove samples that don't have a pack file
        if self.bcp_pack_dir is not None:
            before = len(self.list_data_dict)
            self.list_data_dict = [
                data for data in self.list_data_dict
                if os.path.exists(os.path.join(self.bcp_pack_dir, f"{data['object_id']}.pack.npz"))
            ]
            after = len(self.list_data_dict)
            print(f"Partition packs: {after} of {before} samples have a pack file ({before - after} dropped).")

        if self.data_args is not None and self.data_args.data_debug_num > 0:
            self.list_data_dict = self.list_data_dict[:self.data_args.data_debug_num]
            # * print all the scan_id in debug mode, not using for loop
            print('Debug mode, using: ' + ' '.join([data['object_id'] for data in self.list_data_dict]))
        elif self.data_args is not None and self.data_args.split_train_val:
            # * split train and val with 9:1 ratios
            if self.split == 'train':
                self.list_data_dict = self.list_data_dict[:int(self.data_args.split_ratio * len(self.list_data_dict))]
                print(f"Train set size: {len(self.list_data_dict)}")
            else:
                self.list_data_dict = self.list_data_dict[int(self.data_args.split_ratio * len(self.list_data_dict)):]
                print(f"Val set size: {len(self.list_data_dict)}")

    def _load_point_cloud(self, object_id, type='objaverse'):
        if type == 'objaverse':
            return self._load_objaverse_point_cloud(object_id) 

    def _load_objaverse_point_cloud(self, object_id):
        filename = f"{object_id}_{self.pointnum}.npy"
        point_cloud = np.load(os.path.join(self.data_path, filename))

        if not self.use_color:
            point_cloud = point_cloud[:, :3]

        return point_cloud

    def pc_norm(self, pc):
        """ pc: NxC, return NxC """
        xyz = pc[:, :3]
        other_feature = pc[:, 3:]

        centroid = np.mean(xyz, axis=0)
        xyz = xyz - centroid
        m = np.max(np.sqrt(np.sum(xyz ** 2, axis=1)))
        xyz = xyz / m

        pc = np.concatenate((xyz, other_feature), axis=1)
        return pc
    
    def _process_bcp_pack(self, object_id, pack_path, data_dict):
        try:
            with np.load(pack_path) as pack:
                # patch_feat: (512, 384) float16 -> float32
                data_dict['bcp_patch_feat'] = torch.from_numpy(pack['patch_feat'].astype(np.float32))
                if 'cls_feat' in pack:
                    data_dict['bcp_cls_feat'] = torch.from_numpy(pack['cls_feat'].astype(np.float32))
                
                group_ids = None
                if self._need_group_ids or self.bcp_need_region_context:
                    _K = self.bcp_num_groups
                    if getattr(self, 'bcp_kmeans_reorder', False):
                        kmeans_label_path = os.path.join(self.bcp_pack_dir + f"_kmeans{_K}", f"{object_id}_kmeans.npy")
                        if os.path.exists(kmeans_label_path):
                            group_ids = np.load(kmeans_label_path).astype(np.int32)
                        else:
                            group_ids = np.zeros(512, dtype=np.int32)
                    else:
                        key_r = f'patch_r{_K}'
                        key_map = f'map64_to_{_K}'
                        if key_r in pack:
                            group_ids = pack[key_r].astype(np.int32)
                        elif 'patch_r64' in pack and key_map in pack:
                            group_ids = pack[key_map][pack['patch_r64']].astype(np.int32)
                        else:
                            group_ids = np.zeros(512, dtype=np.int32)
                    data_dict['bcp_group_ids'] = torch.from_numpy(group_ids)
                
                if self.bcp_need_region_context and group_ids is not None:
                    # Compute region_adj and region_stats on the fly
                    patch_xyz = pack['patch_xyz']
                    patch_adj_edges = pack['patch_adj_edges']
                    
                    region_adj = np.zeros((_K, _K), dtype=np.float32)
                    if not self.bcp_zero_adj:
                        for u, v in patch_adj_edges:
                            g_u, g_v = group_ids[u], group_ids[v]
                            if g_u != g_v:
                                region_adj[g_u, g_v] = 1.0
                                region_adj[g_v, g_u] = 1.0
                    # else: zero-adj ablation — leave region_adj as all zeros
                    data_dict['bcp_region_adj'] = torch.from_numpy(region_adj)

                    # Centroid (3), size (1), span (3)
                    region_stats = np.zeros((_K, 7), dtype=np.float32)
                    for i in range(_K):
                        mask = (group_ids == i)
                        g_pts = patch_xyz[mask]
                        region_stats[i, 3] = len(g_pts) / 512.0 # size proportion
                        if len(g_pts) > 0:
                            region_stats[i, 0:3] = g_pts.mean(axis=0) # centroid
                            region_stats[i, 4:7] = g_pts.max(axis=0) - g_pts.min(axis=0) # span
                    data_dict['bcp_region_stats'] = torch.from_numpy(region_stats)

                    # BFS-based group remapping: spatially adjacent regions become sequential
                    if self.bcp_graph_order_bfs:
                        from pointllm.model.group_utils import bfs_order_from_adj, remap_group_ids_by_order
                        bfs_order = bfs_order_from_adj(region_adj, region_stats, num_groups=_K)
                        group_ids = remap_group_ids_by_order(group_ids, bfs_order)
                        data_dict['bcp_group_ids'] = torch.from_numpy(group_ids)
                        # Permute region_adj and region_stats to match new group numbering
                        perm = np.array(bfs_order)
                        region_adj = region_adj[np.ix_(perm, perm)]
                        region_stats = region_stats[perm]
                        data_dict['bcp_region_adj'] = torch.from_numpy(region_adj)
                        data_dict['bcp_region_stats'] = torch.from_numpy(region_stats)

        except Exception as e:
            print(f"WARNING: corrupted pack for {object_id}: {e}")
            data_dict['bcp_patch_feat'] = torch.zeros((512, 384), dtype=torch.float32)
            # graph/context paths load group_ids in the try-path via (_need_group_ids or bcp_need_region_context); mirror that here
            if self._need_group_ids or self.bcp_need_region_context:
                data_dict['bcp_group_ids'] = torch.zeros((512,), dtype=torch.int32)
            if self.bcp_need_region_context:
                _K = self.bcp_num_groups
                data_dict['bcp_region_adj'] = torch.zeros((_K, _K), dtype=torch.float32)
                data_dict['bcp_region_stats'] = torch.zeros((_K, 7), dtype=torch.float32)

    
    def __getitem__(self, index):
        sources = self.list_data_dict[index]
        if isinstance(index, int):
            sources = [sources]
        assert len(sources) == 1, "sources should be a list"
        if self.point_indicator in sources[0]['conversations'][0]['value']:

            object_id = self.list_data_dict[index]['object_id']

            # Point cloud representation
            point_cloud = self._load_point_cloud(object_id) # * N, C
            if self.normalize_pc:
                point_cloud = self.pc_norm(point_cloud) # * need to norm since point encoder is norm

            if self.tokenizer is None and self.bcp_pack_dir is None:
                # Eval mode, no offline features needed — return early
                data_dict = dict(
                    point_clouds=torch.from_numpy(point_cloud.astype(np.float32)),
                    object_ids=object_id
                )
                return data_dict
            elif self.tokenizer is None:
                # Eval mode WITH offline features — load pack and return, skip tokenizer pipeline
                data_dict = dict(
                    point_clouds=torch.from_numpy(point_cloud.astype(np.float32)),
                    object_ids=object_id
                )
                pack_path = os.path.join(self.bcp_pack_dir, f"{object_id}.pack.npz")
                if os.path.exists(pack_path):
                    self._process_bcp_pack(object_id, pack_path, data_dict)
                return data_dict

            sources = preprocess_multimodal_point_cloud(
                copy.deepcopy([e["conversations"] for e in sources]), self.point_backbone_config, point_indicator=self.point_indicator)
        else:
            sources = copy.deepcopy([e["conversations"] for e in sources])

        data_dict = preprocess_v1(
            sources,
            self.tokenizer)

        if isinstance(index, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0],
                             labels=data_dict["labels"][0])

        # point exist in the data
        if self.point_indicator in self.list_data_dict[index]['conversations'][0]['value']:
            data_dict['point_clouds'] = torch.from_numpy(point_cloud.astype(np.float32))

            if self.bcp_pack_dir is not None:
                pack_path = os.path.join(self.bcp_pack_dir, f"{object_id}.pack.npz")
                if os.path.exists(pack_path):
                    self._process_bcp_pack(object_id, pack_path, data_dict)
            
            # Always include object_id for debugging/diagnostics
            data_dict['object_id'] = object_id

        return data_dict

    def __len__(self):
        """Return number of utterances."""
        return len(self.list_data_dict)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_path", default="data/objaverse_data", type=str,
                        help="Path to the data directory.")
    parser.add_argument("--anno_path", default=None, type=str, required=True,
                        help="Path to the annotation file.")
    parser.add_argument("--split", default='train', type=str, 
                        help="Whether to use the train or validation dataset.")
    parser.add_argument("--pointnum", default=8192, type=int,
                        help="Number of points in the point cloud.")
    parser.add_argument("--data_debug_num", default=0, type=int,
                        help="Number of data to debug with.")
    parser.add_argument("--split_train_val", default=False, type=bool,
                        help="Whether to split the dataset into training and validation.")
    parser.add_argument("--split_ratio", default=0.9, type=float,
                        help="The ratio of training to validation data.")
    parser.add_argument("--tokenizer_path", default=None, type=str, required=True,
                        help="Path to the tokenizer config file.")
    
    args = parser.parse_args()

    # Initialize tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.tokenizer_path)

    args.point_backbone_config = None

    # Initialize dataset
    dataset = ObjectPointCloudDataset(
        data_path=args.data_path,
        anno_path=args.anno_path,
        pointnum=args.pointnum,
        split=args.split,
        tokenizer=tokenizer,
        data_args=args
    )

    # Example usage
    print(f'Dataset length: {len(dataset)}')

