'''
Context: mixed strategy strictly uses a one-out-of-three selection for itself.
Incorporate 2D context aligned with target + text-driven support. 
'''
import pytorch_lightning as pl
import torch
from torch import nn 
from torch.nn import functional as F
import numpy as np
import random
import os
from torchvision.utils import save_image
from torch.optim.lr_scheduler import ReduceLROnPlateau
import cv2

# Import Model
from .models.PrimusS_ICL import primus_icl_s

# Import text-driven module
from .tool.text_guide import TextEmbeddingAdapter

# Import Prompt Generators
from .tool.prompt_generators import (
    InteractiveScribbleGenerator, 
    interactive_lasso_3d, 
    randomized_slice_rects_3d, 
    randomized_slice_points_3d,
    randomized_dense_slice_3d,
    # 2D Imports
    randomized_rects_2d,
    randomized_points_2d,
    randomized_dense_2d,
    interactive_lasso_2d,
    generate_scribbles_2d
)

# Import Utils
from .util.shapecheck import ShapeChecker
from utils.pairwise_measures import BinaryPairwiseMeasures as PM
from utils.imagefilter import GINGroupConv_3D
from utils.losses import sigmoid_modified, CustomActivation1, GradientLoss, SmoothL3_L1Loss
from monai.losses import DiceLoss, FocalLoss, DiceCELoss

class LightningModel(pl.LightningModule):
    """
    PrimusICL Lightning Model - Hybrid 2D/3D Training Mode.
    Modified for 7-Strategy Context Configuration with Physical Context Removal for Text-Only Mode.
    """

    def __init__(self, hparams):
        super().__init__()
        print("--"*20)
        print("This is Medverse_prompts (7-Strategy Mode)!")
        print("--"*20)
        self.save_hyperparameters(hparams)
        self.automatic_optimization = False
        
        cv2.setNumThreads(0)
        cv2.ocl.setUseOpenCL(False) 

        # Model Init
        self.net = primus_icl_s(
            input_channels=1, 
            output_channels=1, 
            patch_size=8, 
            input_shape=(128, 128, 128), 
            grad_checkpointing = self.hparams.grad_checkpointing > 0
        )

        # Initialize text adapter
        # Dynamically retrieve the embed_dim from the visual model
        # Attempt to retrieve the dimension from self.net.eva.embed_dim
        if hasattr(self.net, "eva") and hasattr(self.net.eva, "embed_dim"):
            target_dim = self.net.eva.embed_dim
        elif hasattr(self.net, "embed_dim"):
            target_dim = self.net.embed_dim
            
        self.text_adapter = TextEmbeddingAdapter(
            cache_path="biomedbert_embeddings_cache_cls.pt", 
            target_embed_dim=target_dim 
        )
        
        # ==========================================================
        # Move Text Cache to CPU RAM
        # Prevents deadlocks in DDP mode caused by multiple GPUs simultaneously reading the cache on GPU 0
        # ==========================================================
        print(f">> [System] Rank {self.global_rank}: Moving Text Cache to CPU RAM to avoid DDP Deadlock...")
        if hasattr(self.text_adapter, 'embedding_cache') and isinstance(self.text_adapter.embedding_cache, dict):
            # 1. Convert to CPU Tensor
            # 2. Set to non-gradient mode (detach)
            self.text_adapter.embedding_cache = {
                k: v.cpu().detach() for k, v in self.text_adapter.embedding_cache.items()
            }
        print(f">> [System] Text Cache moved to CPU successfully.")
        # ==========================================================
        for param in self.text_adapter.parameters():
            param.requires_grad = False
        
        for name, param in self.text_adapter.named_parameters():
            if "linear" in name or "projection" in name or "adapter" in name:
                param.requires_grad = True
                print(f"  -> Unfrozen: {name}")
        
        # Whitelist of datasets enabled for text prompts
        self.text_enabled_datasets = [
            'Dataset827_hab_t1_freesurfer_raw',
            'Dataset727_hab_t1_freesurfer',
            'Dataset829_nimh_t1_freesurfer_raw',
            'Dataset729_nimh_t1_freesurfer'
        ]
        # ==========================================================
        # Mapping table from Label ID to Text Cache ID
        # ==========================================================
        # Key: Freesurfer Label ID
        # Value: The corresponding Index in the biomedbert_embeddings_cache.pt
        # Inferred based on sorted(list(set(cache_keys))) from preprocess_embeddings.ipynb:
        # 0: (3, 42) Cortex
        # 1: (4, 43) Lateral Ventricle
        # 2: (10, 49) Thalamus
        # 3: (12, 51) Putamen
        # 4: (17, 53) Hippocampus
        # 5: (18, 54) Amygdala
        # Map to ID List (must be consistent with the Key in the cache file)
        self.LABEL_TO_TEXT_ID = {
            # Cortex (Key should be 3, 42)
            3: [3, 42], 42: [3, 42],
            # Ventricle (Key should be 4, 43)
            4: [4, 43], 43: [4, 43],
            # Thalamus
            10: [10, 49], 49: [10, 49],
            # Putamen
            12: [12, 51], 51: [12, 51],
            # Hippocampus
            17: [17, 53], 53: [17, 53],
            # Amygdala
            18: [18, 54], 54: [18, 54],
        }
        
        self.DEFAULT_TEXT_ID = [0] # Default to a list format
        # ==========================================================
        # Strategy probability initialization (7 types)
        # 0: A   (Random Vis)
        # 1: C   (All Target Vis)
        # 2: AC  (Mix Vis) -> Originally Mode B
        # 3: Text(Text Only, Zero Vis)
        # 4: AT  (Random Vis + Dense Prompt + Text)
        # 5: CT  (All Target Vis + Text)
        # 6: ACT (Mix Vis + Mode B Prompt + Text) -> Originally Mode B + Text
        # ==========================================================
        raw_probs = [
            self.hparams.get('prob_a', 0.2),    # 0
            self.hparams.get('prob_c', 0.1),    # 1
            self.hparams.get('prob_ac', 0.1),   # 2
            self.hparams.get('prob_text', 0.2), # 3
            self.hparams.get('prob_at', 0.1),   # 4
            self.hparams.get('prob_ct', 0.1),   # 5
            self.hparams.get('prob_act', 0.2)   # 6
        ]
        
        # Normalize all strategy probabilities (for datasets supporting text)
        total_p = sum(raw_probs)
        if total_p > 0:
            self.probs_all = [p / total_p for p in raw_probs]
        else:
            self.probs_all = [1/7] * 7
            
        # Normalize vision-only strategy probabilities (for datasets not supporting text: A, C, AC only)
        vis_raw = [raw_probs[0], raw_probs[1], raw_probs[2]]
        total_vis = sum(vis_raw)
        if total_vis > 0:
            self.probs_vis_only = [p / total_vis for p in vis_raw]
        else:
            self.probs_vis_only = [0.4, 0.3, 0.3] # Fallback default

        print(f"Strategy Probs (All): {self.probs_all}")
        print(f"Strategy Probs (Vis Only): {self.probs_vis_only}")

        self.scribble_engine = InteractiveScribbleGenerator(
            line_params={'thickness': 2, 'deformation_strength_factor': 0.15},
            centerline_params={'thickness': 3, 'deformation_strength_factor': 0.1},
            contour_params={'thickness': 3, 'deformation_strength_factor': 0.05}
        )
        
        # Prompt Types
        self.prompt_types_3d = ['point', 'box', 'lasso', 'scribble', 'dense_slice', 'dense']
        self.prompt_types_2d = ['point', 'box', 'lasso', 'scribble', 'dense']
        
        self.metrics = ['DSC', 'L1', 'MSE', 'PSNR', 'SSIM']
        
        # === Hyperparameters from args ===
        self.prob_2d = self.hparams.get('prob_2d', 0.5)
        
        raw_num_slices = self.hparams.get('num_slices', [4])
        if isinstance(raw_num_slices, (list, tuple)):
            if len(raw_num_slices) == 1:
                self.num_slices = int(raw_num_slices[0])
            else:
                self.num_slices = list(raw_num_slices)
        else:
            self.num_slices = int(raw_num_slices)

        self.consistency_prob = self.hparams.get('consistency_prob', 0.5)
        self.mode_b_prompt_type = self.hparams.get('mode_b_target_prompt', 'scribble')
        
        # Losses
        self.gradient_loss = GradientLoss(mode = 'l1')
        self.dice_loss = DiceLoss(to_onehot_y=False, sigmoid=False, softmax=False, other_act=None)
        self.focal_loss = FocalLoss(to_onehot_y=False, gamma=2.0, include_background=True)
        self.dice_ce_loss_function = DiceCELoss(to_onehot_y=False, sigmoid=False, softmax=False)
        self.smoothl3l1Loss = SmoothL3_L1Loss(beta=1.0)
        
        # ==========================================================
        # Dynamic Context Size sampling configuration
        # ==========================================================
        self.context_size_pool = self.hparams.get('context_size_pool', [])
        probs_str = self.hparams.get('context_size_probs', ['random'])
        
        if len(probs_str) == 1 and str(probs_str[0]).lower() == 'random':
            self.context_size_probs = None  # Represents a uniform distribution
        else:
            self.context_size_probs = [float(p) for p in probs_str]
            total_prob = sum(self.context_size_probs)
            if total_prob > 0:
                self.context_size_probs = [p / total_prob for p in self.context_size_probs]
            else:
                self.context_size_probs = None
                
        if self.context_size_pool and self.context_size_probs is not None:
            assert len(self.context_size_pool) == len(self.context_size_probs), "The lengths of context_size_pool and context_size_probs must be equal!"
            
    # Update forward to accept visual_mask (for compatibility)
    def forward(self, target_in, context_in, context_out, language_tokens=None, visual_mask=None):
        target_in = target_in.to(self.device)
        context_in = context_in.to(self.device)
        context_out = context_out.to(self.device)
        
        # In Scheme C (physical removal), the model determines this automatically via L=0, so the visual mask is not needed.
        # Omit visual_mask parameter passing
        y_pred = self.net(target_in, context_in, context_out, language_tokens=language_tokens)
        return y_pred

    def _get_target_ids(self, batch, batch_size):
        dataset_name = batch.get('dataset', ['Unknown'])[0]
        # Initialize with default values
        target_ids_batch = [[self.DEFAULT_TEXT_ID]] * batch_size 
        
        if dataset_name in self.text_enabled_datasets:
            if 'target_ids' in batch:
                raw_ids = batch['target_ids'][0]
                val = int(raw_ids.flatten()[0].item()) if torch.is_tensor(raw_ids) else raw_ids
                target_ids_batch = [[val]] * batch_size
            
            elif 'samples' in batch:
                try:
                    samples_data = batch['samples']
                    ids_for_this_batch = []
                    
                    for b_idx in range(batch_size):
                        m_idx = b_idx if b_idx < len(samples_data) else 0
                        sample_item = samples_data[m_idx]
                        if isinstance(sample_item, list): slice_meta = sample_item[0]
                        else: slice_meta = sample_item
                        
                        task_cfg = slice_meta.get('task_config', {})
                        if isinstance(task_cfg, list): task_cfg = task_cfg[0]
                        fg_classes = task_cfg.get('foreground_classes', None)
                        
                        rep_label = None
                        if isinstance(fg_classes, list) and len(fg_classes) > 0:
                            v = fg_classes[0]
                            rep_label = int(v.item()) if torch.is_tensor(v) else int(v)
                        elif torch.is_tensor(fg_classes):
                            rep_label = int(fg_classes.flatten()[0].item())
                        elif isinstance(fg_classes, (int, float)):
                            rep_label = int(fg_classes)
                        
                        # Retrieve ID list
                        # Note: self.DEFAULT_TEXT_ID is currently [0]
                        text_id_list = self.LABEL_TO_TEXT_ID.get(rep_label, self.DEFAULT_TEXT_ID)
                        
                        # Ensure correct structural format: [[3, 42], [17, 53], ...]
                        if isinstance(text_id_list, list):
                            ids_for_this_batch.append(text_id_list)
                        else:
                            ids_for_this_batch.append([text_id_list])

                    target_ids_batch = ids_for_this_batch

                except Exception as e:
                    pass
                    
        return target_ids_batch

    # ==============================================================================
    # Core strategy selection logic
    # ==============================================================================
    def _sample_strategy(self, dataset_name):
        # --- DEBUG START ---
        if not hasattr(self, '_debug_printed'):
            print(f"\n[DEBUG CHECK]")
            print(f"Incoming Dataset Name: '{dataset_name}' (Type: {type(dataset_name)})")
            print(f"Whitelist: {self.text_enabled_datasets}")
            is_in = dataset_name in self.text_enabled_datasets
            print(f"Is In Whitelist? -> {is_in}")
            self._debug_printed = True
        # --- DEBUG END ---

        if dataset_name in self.text_enabled_datasets:
            return np.random.choice(7, p=self.probs_all)
        else:
            # If execution reaches here, setting prob_text=1 has no effect
            return np.random.choice(3, p=self.probs_vis_only)
        
    def _sample_dynamic_context_size(self, current_cases):
        """ Dynamically determine the number of Context Cases to retain for the current Batch based on the configured probability pool. """
        if not self.context_size_pool:
            return current_cases
            
        # Filter out values exceeding the maximum capacity provided by the current DataLoader
        valid_pool = [c for c in self.context_size_pool if c <= current_cases]
        if not valid_pool:
            return current_cases
            
        # Uniform random sampling
        if self.context_size_probs is None:
            return int(np.random.choice(valid_pool))
        else:
            # Extract corresponding probabilities based on valid_pool and re-normalize
            valid_probs = []
            for c in valid_pool:
                idx = self.context_size_pool.index(c)
                valid_probs.append(self.context_size_probs[idx])
            
            total = sum(valid_probs)
            if total <= 0:
                return int(np.random.choice(valid_pool))
            
            valid_probs = [p / total for p in valid_probs]
            return int(np.random.choice(valid_pool, p=valid_probs))

    def _apply_strategy_context(self, strategy_idx, target_in, target_out, context_in, context_lbl, context_flip, L):
        """
        Modify Context content and generate Prompt strategy list based on strategy ID.
        """
        use_text = False
        specific_prompts = None # None means random
        
        # --- 1. Context content modification (Visual Source) ---
        # Strategy 1(C), 5(CT): All Target
        if strategy_idx in [1, 5]:
            for l in range(L): 
                context_in[:, l] = target_in
                context_lbl[:, l] = target_out
        
        # Strategy 2(AC), 6(ACT): Mixed (1 Target + Rest) [Mode B Style]
        elif strategy_idx in [2, 6]:
            if L > 0:  # Prevents index out-of-bounds error if truncated to 0 by dynamic sampling
                context_in[:, 0] = target_in
                context_lbl[:, 0] = target_out
        
        else:
            pass

        # --- 2. Text toggle & Prompt types ---
        if strategy_idx == 0: # A
            use_text = False
            specific_prompts = None 
        elif strategy_idx == 1: # C
            use_text = False
            specific_prompts = None 
        elif strategy_idx == 2: # AC
            use_text = False
            specific_prompts = [self.mode_b_prompt_type] + ['dense'] * (L - 1)
        elif strategy_idx == 3: # Text Only
            use_text = True
            # Scheme C: Context will be physically removed in training_step, so prompt doesn't matter here
            specific_prompts = None
        elif strategy_idx == 4: # AT
            use_text = True
            specific_prompts = ['dense'] * L 
        elif strategy_idx == 5: # CT
            use_text = True
            specific_prompts = None
        elif strategy_idx == 6: # ACT
            use_text = True
            specific_prompts = [self.mode_b_prompt_type] + ['dense'] * (L - 1)

        return context_in, context_lbl, use_text, specific_prompts


    def training_step(self, batch, batch_idx):
        self.automatic_optimization = False
        # DDP Safe Randomness
        rng_seed = int(self.current_epoch * 100000 + batch_idx)
        local_rng = np.random.RandomState(rng_seed)
        
        if local_rng.rand() < self.prob_2d:
            return self._training_step_2d(batch, batch_idx)
        else:
            return self._training_step_3d(batch, batch_idx)

    # 2D Logic
    # =========================================================================
    # Replace the original _extract_slices_single_axis function
    # (Strictly preserve the original area-weighted random selection logic)
    # =========================================================================
    def _extract_multi_context_slices(self, imgs_3d, labs_3d, axis, num_samples, context_multiplier, target_g):
        G = imgs_3d.shape[0]
        local_target_dim = axis + 1
        dims_to_sum = [0, 1, 2, 3]
        dims_to_sum.remove(local_target_dim)

        # --- 1. Base metadata and foreground filtering (Preserving the original area-weighted logic) ---
        group_metadata = []
        for g in range(G):
            lab = labs_3d[g]
            slice_areas = lab.sum(dim=tuple(dims_to_sum)).float()
            
            if slice_areas.max() > 0:
                # Exactly restore the filtering threshold from the original training script
                threshold = 0.1 * slice_areas.max()
                valid_mask = slice_areas > max(threshold, 1.0)
                valid_indices = torch.nonzero(valid_mask).view(-1)
                
                if valid_indices.numel() > 0:
                    # Weights correspond directly to the area size (larger area = higher selection probability)
                    valid_weights = slice_areas[valid_indices]
                    group_metadata.append({
                        'valid': True, 
                        'valid_indices': valid_indices, 
                        'weights': valid_weights,
                        'max_dim': lab.shape[local_target_dim]
                    })
                else:
                    group_metadata.append({'valid': False, 'max_dim': lab.shape[local_target_dim]})
            else:
                group_metadata.append({'valid': False, 'max_dim': lab.shape[local_target_dim]})

        out_t_in_list = []
        out_t_out_list = []
        out_c_in_list = []
        out_c_out_list = []

        # --- 2. Extract Target and Context ---
        for n in range(num_samples):
            # 2.1 Sample Target (Draw 1 slice based on area-weighted probability)
            target_meta = group_metadata[target_g]
            if target_meta['valid']:
                # Multinomial uses weights for weighted probability sampling
                t_idx_local = torch.multinomial(target_meta['weights'], 1, replacement=True).item()
                t_abs_idx = target_meta['valid_indices'][t_idx_local].item()
            else:
                t_abs_idx = torch.randint(0, target_meta['max_dim'], (1,)).item()
                
            t_img = torch.index_select(imgs_3d[target_g], local_target_dim, torch.tensor([t_abs_idx], device=self.device)).squeeze(local_target_dim)
            t_lab = torch.index_select(labs_3d[target_g], local_target_dim, torch.tensor([t_abs_idx], device=self.device)).squeeze(local_target_dim)
            
            # 2.2 Sample Context (Draw context_multiplier slices per support image based on area-weighted probability)
            c_imgs_for_this_sample = []
            c_labs_for_this_sample = []
            
            for g in range(G):
                if g == target_g: continue # Skip the target itself
                follower_meta = group_metadata[g]
                
                # Iteratively expand the number of contexts
                for _ in range(context_multiplier):
                    if follower_meta['valid']:
                        # Contexts are strictly sampled using area-weighted probabilities
                        c_idx_local = torch.multinomial(follower_meta['weights'], 1, replacement=True).item()
                        c_abs_idx = follower_meta['valid_indices'][c_idx_local].item()
                    else:
                        c_abs_idx = torch.randint(0, follower_meta['max_dim'], (1,)).item()
                        
                    c_img = torch.index_select(imgs_3d[g], local_target_dim, torch.tensor([c_abs_idx], device=self.device)).squeeze(local_target_dim)
                    c_lab = torch.index_select(labs_3d[g], local_target_dim, torch.tensor([c_abs_idx], device=self.device)).squeeze(local_target_dim)
                    
                    c_imgs_for_this_sample.append(c_img)
                    c_labs_for_this_sample.append(c_lab)
            
            out_t_in_list.append(t_img)
            out_t_out_list.append(t_lab)
            out_c_in_list.append(torch.stack(c_imgs_for_this_sample, dim=0)) 
            out_c_out_list.append(torch.stack(c_labs_for_this_sample, dim=0))
            
        return (torch.stack(out_t_in_list, dim=0), torch.stack(out_t_out_list, dim=0),
                torch.stack(out_c_in_list, dim=0), torch.stack(out_c_out_list, dim=0))

    # =========================================================================
    # Replace the original _prepare_batch_multi_axis function
    # =========================================================================
    def _prepare_batch_multi_axis(self, imgs_3d, labs_3d, flipped_status_group, batch_idx):
        # -------------------------------------------------------------
        # Configure how many 2D slices each 3D Context image provides
        # Assume DataLoader provides G=5 (1 Target + 4 Contexts)
        # With context_multiplier = 4, the total context count L = 4 * 4 = 16
        # -------------------------------------------------------------
        context_multiplier = 1
        
        ns_config = self.num_slices
        if isinstance(ns_config, int): ns_list = [ns_config, ns_config, ns_config] 
        elif isinstance(ns_config, (list, tuple)) and len(ns_config) == 3: ns_list = ns_config
        elif isinstance(ns_config, (list, tuple)) and len(ns_config) == 1: 
             val = int(ns_config[0]); ns_list = [val, val, val]
        else: ns_list = [4, 4, 4]
            
        target_in_list = []; target_out_list = []
        context_in_list = []; context_label_list = []; context_flip_list = []
        
        G = imgs_3d.shape[0]
        L_total = (G - 1) * context_multiplier
        
        for axis_i, num_s in enumerate(ns_list):
            if num_s <= 0: continue
            
            # Take turns serving as the Target
            for target_g in range(G):
                t_in, t_out, c_in, c_lbl = self._extract_multi_context_slices(
                    imgs_3d, labs_3d, axis_i, num_s, context_multiplier, target_g
                )
                
                # Align flip status
                curr_flipped = flipped_status_group.clone()
                context_indices = [i for i in range(G) if i != target_g]
                c_flip_base = curr_flipped[context_indices] # [G-1]
                
                # Expand the flip status to match L_total
                c_flip = c_flip_base.unsqueeze(1).repeat(1, context_multiplier).view(-1)
                c_flip = c_flip.unsqueeze(0).repeat(num_s, 1) # [num_s, L_total]
                
                target_in_list.append(t_in)
                target_out_list.append(t_out)
                context_in_list.append(c_in)
                context_label_list.append(c_lbl)
                context_flip_list.append(c_flip)
                
        if len(target_in_list) == 0: return None
        return (torch.cat(target_in_list, dim=0), torch.cat(target_out_list, dim=0), 
                torch.cat(context_in_list, dim=0), torch.cat(context_label_list, dim=0), 
                torch.cat(context_flip_list, dim=0))

            
    def _training_step_2d(self, batch, batch_idx):
        task = batch['task'][0]
        dataset_name = batch.get('dataset', ['Unknown'])[0]
        imgs_3d = batch['image'][0][:, None, :]; labs_3d = batch['label'][0][:, None, :]
        flipped_scalar = batch['flipped'][0].to(self.device) if 'flipped' in batch else torch.tensor(0, device=self.device)
        flipped_status = flipped_scalar.repeat(imgs_3d.shape[0]) 
        
        batch_data = self._prepare_batch_multi_axis(imgs_3d, labs_3d, flipped_status, batch_idx)
        if batch_data is None: return None
        target_in, target_out, context_in, context_lbl, context_flip = batch_data
        
        L = context_in.shape[1]
        
        # =================================================================
        # Dynamic Context Size truncation (2D mode)
        # =================================================================
        context_multiplier = 1  # Because it is set in _prepare_batch_multi_axis
        current_cases = L // context_multiplier
        sampled_cases = self._sample_dynamic_context_size(current_cases)
        
        if sampled_cases < current_cases:
            sampled_L = sampled_cases * context_multiplier
            # Physically discard surplus Context images via slicing
            context_in = context_in[:, :sampled_L]
            context_lbl = context_lbl[:, :sampled_L]
            context_flip = context_flip[:, :sampled_L]
            L = sampled_L
        # =================================================================
        
        # === 0. Retrieve Target IDs early to check for unknown categories ===
        is_unknown_label = False
        target_ids = None
        if dataset_name in self.text_enabled_datasets:
            target_ids = self._get_target_ids(batch, target_in.shape[0])
            
            # Flatten and check if all elements are the fallback 0
            def flatten_check(item):
                if isinstance(item, (list, tuple)):
                    return [x for sub in item for x in flatten_check(sub)]
                return [item]
                
            flat_ids = flatten_check(target_ids)
            # If all elements after flattening are 0 (i.e., DEFAULT_TEXT_ID), it indicates an unknown category or random
            if all(x == 0 for x in flat_ids):
                is_unknown_label = True
        
        # === 1. Sample strategy (includes fallback degradation logic) ===
        if is_unknown_label:
            # Force degradation to vision-only strategies (0: A, 1: C, 2: AC)
            strategy_idx = np.random.choice(3, p=self.probs_vis_only)
        else:
            strategy_idx = self._sample_strategy(dataset_name)
        
        # === 2. Apply strategy ===
        context_in, context_lbl, use_text, specific_prompts = self._apply_strategy_context(
            strategy_idx, target_in, target_out, context_in, context_lbl, context_flip, L
        )
        
        # === [Scheme C Modification] Physical Removal for Text Only ===
        visual_mask = None
        if strategy_idx == 3: # Text Only
            # Physical removal: Create an empty Tensor with dimensions [B, 0, C, H, W]
            B, L_old, C, H, W = context_in.shape
            context_in = torch.zeros((B, 0, C, H, W), device=self.device)
            context_lbl = torch.zeros((B, 0, C, H, W), device=self.device)
            context_prompt = torch.zeros((B, 0, C, H, W), device=self.device)
        else:
            # Generate Prompt normally
            context_prompt = self.generate_sparse_prompt(context_lbl, True, context_flip, specific_prompt_types=specific_prompts)

        # === 3. Prepare Text Tokens (reuse previously retrieved target_ids) ===
        language_tokens = None
        if use_text and target_ids is not None:
            language_tokens = self.text_adapter(target_ids, self.device)
            
        # === 4. Forward ===
        optimizer = self.optimizers()
        mask = self.forward(target_in, context_in, context_prompt, language_tokens=language_tokens, visual_mask=visual_mask)
        
        # Multiply by 0.0 to maintain the connectivity of the computation graph
        raw_loss = self.custom_loss(mask, target_out, task)
        if target_out.sum() < 5:
             loss = raw_loss * 0.0  # Filter empty samples: Generate all-zero gradients without disrupting learning
        else:
             loss = raw_loss
        
        # All processes must strictly execute backward pass without checking if loss.item() > 0
        self.manual_backward(loss)
        self.clip_gradients(optimizer, gradient_clip_val=2.5, gradient_clip_algorithm="norm")  
        optimizer.step(); optimizer.zero_grad()
        
        self.log("train_loss", loss)
        return {'loss': loss}

    # ==============================================================================
    # 3D Logic
    # ==============================================================================
    def _training_step_3d(self, batch, batch_idx):
        task = batch['task'][0]; dataset_name = batch.get('dataset', ['Unknown'])[0]
        imgs = batch['image'][0, :, None, :]; labs = batch['label'][0, :, None, :]
        flipped_scalar = batch['flipped'][0].to(self.device) if 'flipped' in batch else torch.tensor(0, device=self.device)
        flipped_status = flipped_scalar.repeat(imgs.shape[0])
        
        if np.random.rand() < self.hparams.gin_prob:
            imgs = self.gin_transform(imgs.float().to(device=self.device))
        
        num_rolls = imgs.shape[0]; optimizer = self.optimizers(); losses = []
        
        # === 0. Retrieve Target IDs early to check for unknown categories ===
        is_unknown_label = False
        all_ids = None
        if dataset_name in self.text_enabled_datasets:
            all_ids = self._get_target_ids(batch, num_rolls)
            
            # Flatten and check if all elements are the fallback 0
            def flatten_check(item):
                if isinstance(item, (list, tuple)):
                    return [x for sub in item for x in flatten_check(sub)]
                return [item]
                
            flat_ids = flatten_check(all_ids)
            # If all elements after flattening are 0 (i.e., DEFAULT_TEXT_ID), it indicates an unknown category or random
            if all(x == 0 for x in flat_ids):
                is_unknown_label = True

        # === 1. Sample strategy (Batch-level uniform strategy) ===
        if is_unknown_label:
            strategy_idx = np.random.choice(3, p=self.probs_vis_only)
        else:
            strategy_idx = self._sample_strategy(dataset_name)

        # =================================================================
        # Determine the Context Size for this Batch uniformly outside the loop
        # =================================================================
        batch_L = imgs.shape[0] - 1  # Initial total available Context count (Batch Size - 1)
        sampled_L = self._sample_dynamic_context_size(batch_L)

        for i in range(num_rolls):
            # 1. Data rolling
            imgs = torch.roll(imgs, 1, 0); labs = torch.roll(labs, 1, 0); flipped_status = torch.roll(flipped_status, 1, 0)
            target_in = imgs[:1,:]; context_in = imgs[None, 1:,:]; target_out = labs[:1,:]; context_label_dense = labs[None, 1:,:]; context_flipped = flipped_status[None, 1:]
            
            # =================================================================
            # Truncate using sampled_L determined outside the loop
            # =================================================================
            if sampled_L < batch_L:
                context_in = context_in[:, :sampled_L]
                context_label_dense = context_label_dense[:, :sampled_L]
                context_flipped = context_flipped[:, :sampled_L]
            
            L = sampled_L # Update L for subsequent use
            # =================================================================

            # === 2. Apply strategy ===
            context_in, context_label_dense, use_text, specific_prompts = self._apply_strategy_context(
                strategy_idx, target_in, target_out, context_in, context_label_dense, context_flipped, L
            )

            # === [Scheme C Modification] Physical Removal for Text Only ===
            visual_mask = None
            if strategy_idx == 3: # Text Only
                # Physical removal
                B, C, W, H, D = target_in.shape
                context_in = torch.zeros((B, 0, C, W, H, D), device=self.device)
                context_label_dense = torch.zeros((B, 0, C, W, H, D), device=self.device)
                context_prompt = torch.zeros((B, 0, C, W, H, D), device=self.device)
            else:
                context_prompt = self.generate_sparse_prompt(context_label_dense, True, context_flipped, specific_prompt_types=specific_prompts)

            # === 3. Prepare Text Tokens (reuse previously retrieved all_ids) ===
            language_tokens = None
            if use_text and all_ids is not None:
                current_idx = -(i + 1) % num_rolls
                target_ids_base = [all_ids[current_idx]]
                language_tokens = self.text_adapter(target_ids_base, self.device)
            
            # 5. Forward propagation
            mask = self.forward(target_in, context_in, context_prompt, language_tokens=language_tokens, visual_mask=visual_mask)
            
            # Include logic to prevent empty slices and compute the original loss
            raw_loss = self.custom_loss(mask, target_out, task)
            if target_out.sum() < 5:
                loss = raw_loss * 0.0
            else:
                loss = raw_loss
                
            losses.append(loss.detach()) 
            
            # 6. Backward propagation 
            # Remove the 'if loss.item() > 0' constraint
            self.manual_backward(loss)
            self.clip_gradients(optimizer, gradient_clip_val=2.5, gradient_clip_algorithm="norm")
            optimizer.step(); optimizer.zero_grad()
            
            del loss, mask, language_tokens

    # ==============================================================================
    # Prompt Generation (Robust Fix)
    # ==============================================================================
    def generate_sparse_prompt(self, dense_mask_batch: torch.Tensor, is_train: bool = True, flipped_batch: torch.Tensor = None, specific_prompt_types=None) -> torch.Tensor:
        if dense_mask_batch.ndim == 6: 
            return self._generate_sparse_prompt_3d(dense_mask_batch, is_train, flipped_batch, specific_prompt_types)
        elif dense_mask_batch.ndim == 5: 
            return self._generate_sparse_prompt_2d(dense_mask_batch, is_train, flipped_batch, specific_prompt_types)
        else: 
            raise ValueError(f"Unsupported mask shape: {dense_mask_batch.shape}")

    def _generate_sparse_prompt_2d(self, dense_mask_batch, is_train, flipped_batch, specific_prompt_types):
        B, L, C, H, W = dense_mask_batch.shape
        prompt_batch = torch.zeros_like(dense_mask_batch)
        consistency_prob = self.consistency_prob 

        for b in range(B):
            current_batch_flips = None
            if flipped_batch is not None:
                if flipped_batch.ndim > 0: current_batch_flips = flipped_batch[b]
                else: current_batch_flips = flipped_batch
            
            use_consistent_prompt = (random.random() < consistency_prob)
            global_prompt_type = random.choice(self.prompt_types_2d)

            for l in range(L):
                mask_2d = dense_mask_batch[b, l, 0]
                if not torch.any(mask_2d > 0): continue
                
                is_current_slice_flipped = False
                if current_batch_flips is not None:
                    if isinstance(current_batch_flips, torch.Tensor) and current_batch_flips.ndim > 0:
                        idx = l if l < len(current_batch_flips) else 0
                        is_current_slice_flipped = bool(current_batch_flips[idx] > 0)
                    else: is_current_slice_flipped = bool(current_batch_flips > 0)

                if specific_prompt_types is not None and l < len(specific_prompt_types) and specific_prompt_types[l] is not None:
                    chosen_type = specific_prompt_types[l]
                    if chosen_type == 'dense_slice': 
                        sparse_options = [t for t in self.prompt_types_2d if t != 'dense']
                        chosen_type = random.choice(sparse_options)
                elif is_current_slice_flipped: 
                    chosen_type = 'dense'
                elif use_consistent_prompt: 
                    chosen_type = global_prompt_type
                else: 
                    chosen_type = random.choice(self.prompt_types_2d)
                
                sparse_prompt_2d = None
                try:
                    if chosen_type == 'box': sparse_prompt_2d = randomized_rects_2d(mask_2d, max_rects=1, var=0.5)
                    elif chosen_type == 'point': sparse_prompt_2d = randomized_points_2d(mask_2d, radius=6, max_points=1)
                    elif chosen_type == 'dense': sparse_prompt_2d = randomized_dense_2d(mask_2d)
                    elif chosen_type == 'lasso': sparse_prompt_2d = interactive_lasso_2d(mask_2d)
                    elif chosen_type == 'scribble':
                        sparse_prompt_2d = generate_scribbles_2d(mask_2d, 
                            generator=lambda m: self.scribble_engine.generators[random.choice(['line', 'centerline', 'contour'])](m)
                        )
                    if sparse_prompt_2d is not None: prompt_batch[b, l, 0] = sparse_prompt_2d
                except Exception: pass
        return prompt_batch

    def _generate_sparse_prompt_3d(self, dense_mask_batch, is_train, flipped_batch, specific_prompt_types):
        B, L, C, W, H, D = dense_mask_batch.shape
        prompt_batch = torch.zeros_like(dense_mask_batch)
        consistency_prob = self.consistency_prob 

        for b in range(B):
            current_batch_flips = None
            if flipped_batch is not None:
                if flipped_batch.ndim > 0: current_batch_flips = flipped_batch[b]
                else: current_batch_flips = flipped_batch
            
            use_consistent_prompt = (random.random() < consistency_prob)
            global_prompt_type = random.choice(self.prompt_types_3d)

            for l in range(L):
                mask_3d = dense_mask_batch[b, l, 0]
                if not torch.any(mask_3d > 0): continue
                is_current_slice_flipped = False
                if current_batch_flips is not None:
                    if isinstance(current_batch_flips, torch.Tensor) and current_batch_flips.ndim > 0:
                        idx = l if l < len(current_batch_flips) else 0
                        is_current_slice_flipped = bool(current_batch_flips[idx] > 0)
                    else: is_current_slice_flipped = bool(current_batch_flips > 0)
                
                if specific_prompt_types is not None and l < len(specific_prompt_types) and specific_prompt_types[l] is not None:
                    chosen_type = specific_prompt_types[l]
                elif is_current_slice_flipped: 
                    chosen_type = 'dense'
                elif use_consistent_prompt: 
                    chosen_type = global_prompt_type
                else: 
                    chosen_type = random.choice(self.prompt_types_3d)
                
                sparse_prompt_3d = None
                try:
                    if chosen_type == 'box': sparse_prompt_3d = randomized_slice_rects_3d(mask_3d, max_slice=1, var=0.5)
                    elif chosen_type == 'point': sparse_prompt_3d = randomized_slice_points_3d(mask_3d, radius=6, max_slice=1)
                    elif chosen_type == 'dense_slice': sparse_prompt_3d = randomized_dense_slice_3d(mask_3d, max_slice=1)
                    elif chosen_type == 'dense': sparse_prompt_3d = mask_3d.clone()
                    elif chosen_type in ['lasso', 'scribble']:
                        mask_3d_cpu = (mask_3d > 0.5).float().contiguous().cpu()
                        if chosen_type == 'lasso':
                            sparse_prompt_3d_cpu, _ = interactive_lasso_3d(mask_3d_cpu.to(torch.uint8), max_prompts=1)
                            sparse_prompt_3d = sparse_prompt_3d_cpu.to(self.device)
                        elif chosen_type == 'scribble':
                            sparse_prompt_3d_cpu, _ = self.scribble_engine.generate_3d_scribbles(mask_3d_cpu, max_prompts=1)
                            sparse_prompt_3d = sparse_prompt_3d_cpu.to(self.device)
                    if sparse_prompt_3d is not None: prompt_batch[b, l, 0] = sparse_prompt_3d
                except Exception: pass
        return prompt_batch

    # ==============================================================================
    # Visualization (Auto 2D/3D)
    # ==============================================================================
    def visualize_results(self, target_img, target_gt, pred_logits, context_mask, context_prompt, context_img, batch_idx, save_prefix='val'):
        log_dir = self.logger.log_dir if self.logger and self.logger.log_dir else "logs"
        save_dir = os.path.join(log_dir, "visualizations")
        os.makedirs(save_dir, exist_ok=True)
        
        # ----------------------
        # 2D Mode
        # ----------------------
        if target_img.ndim == 4: 
            b = 0 
            def norm(t): return (t - t.min()) / (t.max() - t.min() + 1e-6)
            
            t_img = norm(target_img[b].float().cpu())
            t_gt = target_gt[b].float().cpu()
            
            if context_img.shape[1] > 0:
                c_img = norm(context_img[b, 0].float().cpu())
                c_mask = context_mask[b, 0].float().cpu()
                c_prompt = context_prompt[b, 0].float().cpu()
            else:
                c_img = torch.zeros_like(t_img); c_mask = torch.zeros_like(t_gt); c_prompt = torch.zeros_like(t_gt)
                
            pred = sigmoid_modified(pred_logits[b], scale=1, offset=0.5).float().cpu()
            pred_bin = (pred > 0.5).float()
            
            grid = torch.cat([c_img, c_mask, c_prompt, t_img, t_gt, pred, pred_bin], dim=2)
            save_image(grid, os.path.join(save_dir, f"{save_prefix}_ep{self.current_epoch}_b{batch_idx}_2D.png"))
            
        # ----------------------
        # 3D Mode
        # ----------------------
        else:
            b = 0
            # Handle Empty Context (L=0)
            if context_prompt.shape[1] == 0:
                # Mock prompt volume for visualization if empty
                prompt_vol = torch.zeros_like(target_img[b, 0])
            else:
                if context_prompt.ndim == 6: prompt_vol = context_prompt[b, 0, 0] 
                elif context_prompt.ndim == 5: prompt_vol = context_prompt[b, 0]    
                else: prompt_vol = context_prompt[b]

            # Slice selection logic optimization
            prompt_indices = torch.nonzero(prompt_vol) 
            
            if prompt_indices.shape[0] > 0:
                # Primary Plan A: Follow the Prompt (Normal mode)
                std_devs = torch.std(prompt_indices.float(), dim=0)
                best_axis_idx = torch.argmin(std_devs).item()
                slice_index = int(prompt_indices[:, best_axis_idx].float().mean().item())
            else:
                # Alternative Plan B: Follow the GT (Text Only mode)
                # Since there is no Prompt for guidance, search the GT for a non-empty slice
                gt_indices = torch.nonzero(target_gt[b, 0])
                
                if gt_indices.shape[0] > 0:
                    # Calculate the center of mass of the GT
                    std_devs = torch.std(gt_indices.float(), dim=0)
                    best_axis_idx = torch.argmin(std_devs).item()
                    slice_index = int(gt_indices[:, best_axis_idx].float().mean().item())
                else:
                    # Alternative Plan C: GT is also empty (true background). Lacking a reference, slice through the middle
                    best_axis_idx = 2 
                    slice_index = target_img.shape[2] // 2

            def get_s(t):
                if t is None: return torch.zeros(1, 128, 128)
                t_obj = t
                if t.shape[1] == 0: return torch.zeros(1, 128, 128) # Handle empty context
                if t.ndim == 6: t_obj = t[b, 0] 
                elif t.ndim == 5: t_obj = t[b]
                target_dim = best_axis_idx + 1 
                max_dim = t_obj.shape[target_dim]
                safe_idx = max(0, min(slice_index, max_dim - 1))
                s = torch.index_select(t_obj, target_dim, torch.tensor([safe_idx], device=t.device)).squeeze(target_dim)
                if s.ndim == 2: s = s.unsqueeze(0)
                return s.detach().float().cpu()

            def norm(t): return (t - t.min()) / (t.max() - t.min() + 1e-6)
            
            img_s = norm(get_s(target_img))
            gt_s = get_s(target_gt)
            ctx_img_s = norm(get_s(context_img))
            ctx_mask_s = get_s(context_mask)
            ctx_prompt_s = get_s(context_prompt)
            pred_slice_raw = get_s(pred_logits)
            pred_s = sigmoid_modified(pred_slice_raw, scale=1, offset=0.5)
            pred_bin = (pred_s > 0.5).float()

            grid = torch.cat([ctx_img_s, ctx_mask_s, ctx_prompt_s, img_s, gt_s, pred_s, pred_bin], dim=2)
            axis_names = ['W', 'H', 'D']
            fname = f"{save_prefix}_ep{self.current_epoch}_b{batch_idx}_3D_Axis{axis_names[best_axis_idx]}_idx{slice_index}.png"
            save_image(grid, os.path.join(save_dir, fname))
            
    def configure_optimizers(self):
        # 1. Separate parameter groups
        text_params = list(self.text_adapter.parameters())
        base_params = [p for n, p in self.named_parameters() if "text_adapter" not in n]
        
        # 2. Set a higher LR for the Text Adapter (e.g., 50-100x the base LR)
        adapter_lr = self.hparams.lr * 100.0 
        
        # Print confirmation
        print(f"\n[Optimizer Config] Base LR: {self.hparams.lr}, Text Adapter LR: {adapter_lr}")
        
        optimizer = torch.optim.Adam([
            {'params': base_params, 'lr': self.hparams.lr},
            {'params': text_params, 'lr': adapter_lr} # Aggressive learning rate for Text Adapter
        ])
        
        scheduler = ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=self.hparams.lr_decline_patience, 
            min_lr=self.hparams.lr*0.01, verbose=True
        )
        
        return {
            'optimizer': optimizer, 
            'lr_scheduler': {
                'scheduler': scheduler, 
                'monitor': 'val_loss', 
                'interval': 'epoch',
                'frequency': 1, 
                'strict': True
            }
        }
    
    # ==============================================================================
    # Validation Step (Metrics Integrated)
    # ==============================================================================
    def validation_step(self, batch, batch_idx):
        # 1. Extract metadata and task configuration
        dataset_name = batch.get('dataset', ['Unknown'])[0]
        task = batch['task'][0]

        def process_input(input_mod):
            if isinstance(input_mod, str): return input_mod
            elif isinstance(input_mod, torch.Tensor): return int(input_mod.item())
            else: return input_mod
        
        input_mod = process_input(batch.get('input', [''])[0])
        output_mod = process_input(batch.get('output', [''])[0])

        # Metric calculation helper function
        def _calc_and_log_metrics(mask, target_out):
            p_pred = (mask > 0.5).float().cpu().numpy()
            p_ref = target_out.float().cpu().numpy()
            
            if p_pred.sum() == 0 and p_ref.sum() == 0:
                dsc = 1.0
            else:
                try:
                    from utils.pairwise_measures import BinaryPairwiseMeasures as PM
                    bpm = PM(p_pred, p_ref, dict_args={"nsd": 1, "hd_perc": 95})
                    if task == 'Seg':
                        if 'DSC' in self.metrics:
                            dsc = bpm.dsc()
                            self.log(f'{dataset_name}_{str(input_mod)}{str(output_mod)}_{task}_DSC', float(dsc))
                    elif task in ['SupRes', 'Inp', '2D23D', 'ModTran', 'Bias','Denoi']:
                        if 'L1' in self.metrics:
                            l1 = bpm.calculate_l1(mask.float().cpu().numpy(), target_out.float().cpu().numpy())
                            self.log(f'{dataset_name}_{str(input_mod)}{str(output_mod)}_{task}_L1', l1)
                except Exception:
                    pass

        # 2. Set random seed to ensure reproducibility during validation
        rng_seed = int(self.current_epoch * 100000 + batch_idx)
        local_rng = np.random.RandomState(rng_seed)
        
        # Directly utilize the strategy sampling logic defined within the class
        # This automatically samples based on the probabilities calculated from parameters such as --prob_text, --prob_a
        # If --prob_text is 1, strategy_idx will strictly equal 3 here
        strategy_idx = self._sample_strategy(dataset_name)

        # --- 2D Validation Branch ---
        if local_rng.rand() < self.prob_2d:
             imgs_3d = batch['image'][0][:, None, :]
             labs_3d = batch['label'][0][:, None, :]
             
             flipped_scalar = batch.get('flipped', [torch.tensor(0)])[0].to(self.device)
             flipped_status = flipped_scalar.repeat(imgs_3d.shape[0])
             
             batch_data = self._prepare_batch_multi_axis(imgs_3d, labs_3d, flipped_status, batch_idx)
             if batch_data is None: return
             target_in, target_out, context_in, context_lbl, context_flip = batch_data
             
             L = context_in.shape[1]

             # Apply strategy logic (reusing training logic)
             context_in, context_lbl, use_text, specific_prompts = self._apply_strategy_context(
                 strategy_idx, target_in, target_out, context_in, context_lbl, context_flip, L
             )
             
             # Text Token retrieval
             language_tokens = None
             if use_text:
                 # Note: Ensure the dataset supports text. _sample_strategy already handles this check,
                 # but for safety, the whitelist is checked again or handled internally by _get_target_ids
                 target_ids = self._get_target_ids(batch, target_in.shape[0])
                 language_tokens = self.text_adapter(target_ids, self.device)
             
             # Visual context processing (Scheme C: Physical removal)
             if strategy_idx == 3: # Text Only Strategy
                 B, L_old, C, H, W = context_in.shape
                 context_in = torch.zeros((B, 0, C, H, W), device=self.device)
                 context_prompt = torch.zeros((B, 0, C, H, W), device=self.device)
                 vis_lbl = torch.zeros((B, 0, C, H, W), device=self.device)
             else:
                 context_prompt = self.generate_sparse_prompt(context_lbl, is_train=False, flipped_batch=context_flip, specific_prompt_types=specific_prompts)
                 vis_lbl = context_lbl
             
             # Forward propagation (visual_mask is set to None, automatically inferred via L=0)
             mask = self.forward(target_in, context_in, context_prompt, language_tokens=language_tokens, visual_mask=None)
             loss = self.custom_loss(mask, target_out, task)
             self.log('val_loss', loss)
             
             if batch_idx % 10 == 0:
                 # Dummy visualization variables
                 vis_ctx_in = context_in if context_in.shape[1] > 0 else torch.zeros_like(target_in).unsqueeze(1)
                 vis_ctx_prompt = context_prompt if context_prompt.shape[1] > 0 else torch.zeros_like(target_in).unsqueeze(1)
                 vis_ctx_lbl = vis_lbl if vis_lbl.shape[1] > 0 else torch.zeros_like(target_in).unsqueeze(1)
                 self.visualize_results(target_in, target_out, mask, vis_ctx_lbl, vis_ctx_prompt, vis_ctx_in, batch_idx)
             
             _calc_and_log_metrics(mask, target_out)
             
        # --- 3D Validation Branch ---
        else:
             imgs = batch['image'][0, :, None, :]
             labs = batch['label'][0, :, None, :]
             
             # Vision-end slicing: Batch Size is forced to 1
             target_in = imgs[:1,:] 
             context_in = imgs[None, 1:,:]
             target_out = labs[:1,:]
             context_label_dense = labs[None, 1:,:]
             flipped_status_slice = torch.zeros(1, device=self.device) # Dummy flip for 3D val

             L = context_in.shape[1]
             
             # Apply strategy logic
             context_in, context_label_dense, use_text, specific_prompts = self._apply_strategy_context(
                 strategy_idx, target_in, target_out, context_in, context_label_dense, flipped_status_slice, L
             )
             
             # Text Token retrieval (forced to batch=1)
             language_tokens = None
             if use_text:
                 target_ids = self._get_target_ids(batch, 1) 
                 language_tokens = self.text_adapter(target_ids, self.device)
             
             # Visual context processing (Scheme C: Physical removal)
             if strategy_idx == 3: # Text Only Strategy
                 B, C, W, H, D = target_in.shape
                 context_in = torch.zeros((B, 0, C, W, H, D), device=self.device)
                 context_prompt = torch.zeros((B, 0, C, W, H, D), device=self.device)
                 vis_lbl = torch.zeros((B, 0, C, W, H, D), device=self.device)
             else:
                 context_prompt = self.generate_sparse_prompt(context_label_dense, is_train=False, specific_prompt_types=specific_prompts)
                 vis_lbl = context_label_dense
             
             mask = self.forward(target_in.to(self.device), context_in.to(self.device), 
                                 context_prompt, language_tokens=language_tokens, visual_mask=None)
             
             loss = self.custom_loss(mask, target_out.to(self.device), task)
             self.log('val_loss', loss)
             
             if batch_idx % 10 == 0:
                 vis_ctx_in = context_in if context_in.shape[1] > 0 else torch.zeros_like(target_in).unsqueeze(1)
                 vis_ctx_prompt = context_prompt if context_prompt.shape[1] > 0 else torch.zeros_like(target_in).unsqueeze(1)
                 vis_ctx_lbl = vis_lbl if vis_lbl.shape[1] > 0 else torch.zeros_like(target_in).unsqueeze(1)
                 self.visualize_results(target_in, target_out, mask, vis_ctx_lbl, vis_ctx_prompt, vis_ctx_in, batch_idx)
             
             _calc_and_log_metrics(mask, target_out.to(self.device))

    def on_validation_epoch_end(self):
        sch = self.lr_schedulers()
        if sch is not None:
            try: sch.step(self.trainer.callback_metrics["val_loss"])
            except: pass

    def custom_loss(self, output, target, key):
        if key == 'Seg' or key == 'seg':
            if self.hparams.loss_seg == 'case_weight':
                loss = F.smooth_l1_loss(output, target, reduction='mean')
                scale = target.numel()*0.05/(target.sum()+5.)
                if scale > 100: scale = 100 
                loss = loss * scale
            elif self.hparams.loss_seg == 'dice_sigmoid':
                loss = self.dice_loss(sigmoid_modified(output, scale = 1, offset = 0.5), target)
            elif self.hparams.loss_seg == 'focal_loss':
                loss = self.focal_loss(output-0.5, target)
            elif self.hparams.loss_seg == 'DiceCE_loss':
                loss = self.dice_ce_loss_function(sigmoid_modified(output, scale = 1, offset = 0.5), target)
            elif self.hparams.loss_seg == 'smoothl1':
                loss = F.smooth_l1_loss(output, target, reduction='mean')
            elif self.hparams.loss_seg == 'smoothl3_l1':
                loss = 50 * self.smoothl3l1Loss(output, target)
            else:
                loss = self.dice_loss(sigmoid_modified(output, scale = 1, offset = 0.5), target)
        else:
            loss = F.smooth_l1_loss(output, target, reduction='mean')
            loss_gradient = self.gradient_loss(output, target)
            loss = 0.5 * loss_gradient + 0.5 * loss
        return loss
        
    def gin_transform(self, image):
        gin = GINGroupConv_3D(interm_channel=10, scale_pool=[1,3,1,1], n_layer=4, device = self.device).to(self.device)
        with torch.no_grad():
            image_ = gin(image)
        batch = image_.shape[0]
        min_vals = image_.view(batch, -1).min(dim=1, keepdim=True)[0].view(batch, 1, 1, 1, 1)
        max_vals = image_.view(batch, -1).max(dim=1, keepdim=True)[0].view(batch, 1, 1, 1, 1)
        normalized = (image_ - min_vals) / (max_vals - min_vals)
        return normalized
