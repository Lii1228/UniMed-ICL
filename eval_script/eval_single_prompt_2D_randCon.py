import torch, torch.nn as nn, torch.utils.data as data, torchvision as tv, torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
import os
import json
import cv2
from tqdm import tqdm
import argparse
import random
import PIL
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import sys
from datetime import datetime
import warnings
import types  

# --- 获取项目根目录并加入系统路径 ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from dataloader import MetaDataset_Multi, Meta_dataset_Sampler, MetaDataset_Multi_Extended, MetaDatasetf_transform_1channel
from config import get_parser

# ================= 配置与加载部分 =================
args = get_parser().parse_args()
print(args)

print("="*50)
print("Running UniMed-ICL 2D Random Retrieval in Open-Source Sample Mode")
print("="*50)

sample_config_path = os.path.join(PROJECT_ROOT, 'Liver', 'Liver.json')

args.data_dir = PROJECT_ROOT

try:
    with open(sample_config_path, 'r') as f:
        data_loading_config = json.load(f)
    print(f"Successfully loaded open-source config from {sample_config_path}")
    print('The current sample task is:', data_loading_config[0]['name'])
except Exception as e:
    raise RuntimeError(f"Failed to load sample config. Error: {e}")
# ----------------------------------------

assert args.checkpoint_path is not None, "The checkpoint_path should not be None!"
checkpoint_path = args.checkpoint_path
tmp = sorted([i for i in os.listdir(checkpoint_path) if i.endswith('.ckpt') or i.endswith('.pth')])[args.checkpoint_index]
checkpoint_path = os.path.join(checkpoint_path, tmp)
print('load check points from:', checkpoint_path)

# Load model
model_module = f'from {args.model_name}.lightning_model import LightningModel'
print(model_module)
exec(model_module)

warnings.filterwarnings('ignore')
model = LightningModel.load_from_checkpoint(checkpoint_path, map_location=torch.device(args.device))

# ================= 【核心修复：环境与策略控制】 =================
print("!!! Forcing Model Controls !!!")

model.prob_2d = 1  
model.num_slices = 1 

# 锁死策略：仅使用纯视觉上下文
model._sample_strategy = lambda dataset_name: 0
print("-> Evaluation Mode ON: Model strategy forced to 0 (Random Visual Context only).")

# ================= 【核心修改：随机抽取切片构建上下文】 =================
def random_extract_slices(self, img_3d_group, lab_3d_group, axis, num_samples, batch_idx):
    """
    Random ICL Context Retrieval:
    随机从包含前景的切片中抽取作为 Context，不进行形态或伪标签匹配。
    """
    G = img_3d_group.shape[0]
    selected_indices_matrix = torch.zeros((G, num_samples), dtype=torch.long, device=self.device)
    
    local_target_dim = axis + 1
    dims_to_sum = [0, 1, 2, 3]
    dims_to_sum.remove(local_target_dim)

    # --- 1. 基础元数据与前景过滤 ---
    group_metadata = []
    for g in range(G):
        lab = lab_3d_group[g]
        slice_areas = lab.sum(dim=tuple(dims_to_sum)).float()
        
        if slice_areas.max() > 0:
            valid_mask = slice_areas > 0.0
            valid_indices = torch.nonzero(valid_mask).view(-1)
            valid_weights = torch.ones_like(valid_indices, dtype=torch.float)
            
            group_metadata.append({
                'valid': True, 
                'valid_indices': valid_indices, 
                'weights': valid_weights,
                'max_dim': lab.shape[local_target_dim]
            })
        else:
            group_metadata.append({'valid': False, 'max_dim': lab.shape[local_target_dim]})

    # --- 2. 随机选取策略 ---
    for n in range(num_samples):
        valid_g_indices = [i for i, meta in enumerate(group_metadata) if meta['valid']]
        
        # 极端情况兜底：如果整个 Batch 都没有前景，完全随机抽
        if not valid_g_indices:
            for g in range(G):
                selected_indices_matrix[g, n] = torch.randint(0, group_metadata[g]['max_dim'], (1,)).item()
            continue
        
        # 2.1 随机选取 Target (Leader) 切片
        leader_g = random.choice(valid_g_indices)
        leader_meta = group_metadata[leader_g]
        
        sample_idx_local = torch.multinomial(leader_meta['weights'], 1, replacement=True).item()
        leader_abs_idx = leader_meta['valid_indices'][sample_idx_local].item()
        selected_indices_matrix[leader_g, n] = leader_abs_idx
        
        # 2.2 为 Context (Follower) 纯随机寻找切片
        for g in range(G):
            if g == leader_g: 
                continue
            follower_meta = group_metadata[g]
            
            if follower_meta['valid']:
                # 从该 Support 组的有效前景切片中随机抽取一个
                random_idx = torch.randint(0, len(follower_meta['valid_indices']), (1,)).item()
                best_abs_idx = follower_meta['valid_indices'][random_idx].item()
                selected_indices_matrix[g, n] = best_abs_idx
            else:
                # Support 无前景时，从全维度随机抽
                selected_indices_matrix[g, n] = torch.randint(0, follower_meta['max_dim'], (1,)).item()

    # --- 3. 提取图像和标签 ---
    out_img_group = []
    out_lab_group = []
    for g in range(G):
        img = img_3d_group[g]; lab = lab_3d_group[g]; indices = selected_indices_matrix[g]
        imgs_list = []; labs_list = []
        for idx in indices:
            idx_tensor = idx.view(1)
            curr_img = torch.index_select(img, local_target_dim, idx_tensor).squeeze(local_target_dim)
            curr_lab = torch.index_select(lab, local_target_dim, idx_tensor).squeeze(local_target_dim)
            imgs_list.append(curr_img); labs_list.append(curr_lab)
        out_img_group.append(torch.stack(imgs_list, dim=0))
        out_lab_group.append(torch.stack(labs_list, dim=0))
        
    return torch.stack(out_img_group, dim=0), torch.stack(out_lab_group, dim=0)

# 动态绑定：替换模型原本的切片抽取方法
model._extract_slices_single_axis = types.MethodType(random_extract_slices, model)
print("-> Slices extraction method patched: Random Valid Retrieval (Foreground Only).")
# =========================================================================

# 统计参数
total_params = sum(p.numel() for p in model.parameters())
print("Total number of parameters: ", total_params)
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Number of trainable parameters: ", trainable_params)

# ================= 数据集准备 =================
dataset_val = MetaDataset_Multi_Extended(
        dataset_dir = args.data_dir, 
        skip_resize = args.skip_resize,
        data_loading_config = data_loading_config,
        train_or_val = 'val',
        train_or_val_split_rate = args.train_or_val_split_rate,
        group_size = args.context_size+1,
        DDPM = '_DDPM' in args.model_name,
        len_divid = args.len_divid,
        cut_length = True)

dataset_val.len_divid = 4
batchsize = args.batch_size

dataloader_val = DataLoader(dataset_val, 
                            batch_size=batchsize,
                            num_workers=args.workers,
                            pin_memory=True,
                            persistent_workers=True)

# 初始化 Trainer
trainer = pl.Trainer(logger=False, enable_checkpointing=False)

# ================= 评估逻辑 =================
target_prompt_types = ['dense']
final_results = {}

print("="*50)
print(f"Starting Individual Prompt Evaluation for: {target_prompt_types}")
print("="*50)

for p_type in target_prompt_types:
    print(f"\n>>> Evaluating Prompt Type: [ {p_type} ] ...")
    
    # 强制修改模型内部的 prompt_types 列表
    model.prompt_types_2d = [p_type] 
    model.prompt_types_3d = [p_type] 
    model.visualize_results = lambda *args, **kwargs: None
    # ==================================================
    
    # 运行验证
    trainer.validate(model, dataloaders=dataloader_val)
    
    # 获取指标
    metrics = model.trainer.callback_metrics
    # 确保 Tensor 被正确转换
    metrics_dict = {key: (value.item() if isinstance(value, torch.Tensor) else value) for key, value in metrics.items()}
    final_results[p_type] = metrics_dict
    print(f">>> Result for {p_type}: {metrics_dict}")

# ================= 保存结果 =================
current_time = datetime.now().strftime("%H%M%S")

if hasattr(model, 'prob_2d') and model.prob_2d==1 :
    mode_tag = "_2D"
elif hasattr(model, 'prob_2d') and model.prob_2d==0 :
    mode_tag = "_3D"
else:
    mode_tag = "_mix"


safe_save_dir = os.path.join(PROJECT_ROOT, "opensource_eval_results")
os.makedirs(safe_save_dir, exist_ok=True)


base_ckpt_name = os.path.basename(checkpoint_path)
new_file_name = base_ckpt_name.replace(
    '.ckpt', 
    f'_FLARE22_Sample_context{str(args.context_size)}{mode_tag}_Random_Retrieval_{current_time}_ICL.json'
)

# 组合成最终的安全路径
save_filename = os.path.join(safe_save_dir, new_file_name)

print("\nFinal Results Dict:", json.dumps(final_results, indent=4))

with open(save_filename, 'w') as f:
    json.dump(final_results, f, indent=4)

print("="*50)
print(f"Detailed evaluation metrics saved to:\n{save_filename}")
print("="*50)