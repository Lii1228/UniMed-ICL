import torch, torch.nn as nn, torch.utils.data as data, torchvision as tv, torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from dataloader import MetaDataset_Multi, Meta_dataset_Sampler, MetaDataset_Multi_Extended, MetaDatasetf_transform_1channel
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

from config import get_parser
from config import data_loading_config_meta

# ================= 配置与加载部分 =================
args = get_parser().parse_args()
print(args)

task_names = list(data_loading_config_meta.keys())
task_name = task_names[args.task_idx]
data_loading_config = data_loading_config_meta[task_name]
print('All task:',task_names)
print('Length :',len(task_names))
print('The current task is:', task_name)

assert args.checkpoint_path is not None, "The checkpoint_path should not be None!"
checkpoint_path = args.checkpoint_path
tmp = sorted([i for i in os.listdir(checkpoint_path)])[args.checkpoint_index]
checkpoint_path = os.path.join(checkpoint_path, tmp)
print('load check points from:', checkpoint_path)

# Load model
model_module = f'from {args.model_name}.lightning_model import LightningModel'
print(model_module)
exec(model_module)

warnings.filterwarnings('ignore')
model = LightningModel.load_from_checkpoint(checkpoint_path, map_location=torch.device(args.device))

# ================= 【环境与策略控制】 =================
print("!!! Forcing Model Controls !!!")

# 1. 控制 2D/3D 模式
model.prob_2d = 0  # 核心修改：0 为强制 3D，如果需要 2D 请改为 1.0
model.num_slices = 1 

# 2. 锁死模型的策略采样逻辑
# 强制让 _sample_strategy 永远返回 0 (Random Visual Context only)
# 策略 0 意味着：纯视觉上下文，无文本干扰，且 specific_prompts = None (允许外层循环完全接管 Prompt)
model._sample_strategy = lambda dataset_name: 0
print("-> Evaluation Mode ON: Model strategy forced to 0 (Random Visual Context only).")
# ==========================================================

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

# ================= 分别评估每种 Prompt =================

# 1. 定义你想测试的所有 Prompt 类型
#target_prompt_types = ['lasso', 'box', 'point', 'scribble', 'dense_slice','dense']
target_prompt_types = ['dense']

# 用于存储所有结果的大字典
final_results = {}

print("="*50)
print(f"Starting Individual Prompt Evaluation for: {target_prompt_types}")
print("="*50)

for p_type in target_prompt_types:
    print(f"\n>>> Evaluating Prompt Type: [ {p_type} ] ...")
    
    # 强制修改模型内部的 prompt_types 列表
    model.prompt_types_2d = [p_type] 
    model.prompt_types_3d = [p_type] 
    
    # 运行验证
    trainer.validate(model, dataloaders=dataloader_val)
    
    # 获取指标
    metrics = model.trainer.callback_metrics
    # 转换为普通字典 (float)
    metrics_dict = {key: value.item() for key, value in metrics.items()}
    
    # 存入总结果
    final_results[p_type] = metrics_dict
    
    print(f">>> Result for {p_type}: {metrics_dict}")

# ================= 保存结果 =================

current_time = datetime.now().strftime("%H%M%S")
epoch = checkpoint_path.split('/')[-1].split('-')[0]

# 判断是 2D 还是 3D，自动添加标识
if hasattr(model, 'prob_2d') and model.prob_2d==1 :
    mode_tag = "_2D"
elif hasattr(model, 'prob_2d') and model.prob_2d==0 :
    mode_tag = "_3D"
else:
    mode_tag = "_mix"

# 在文件名中加入 mode_tag 和 random_context 标识以示区分
save_filename = checkpoint_path.replace('/checkpoints', '/eval_'+epoch).replace(
    '.ckpt', 
    f'_context{str(args.context_size)}_task{args.task_idx}{mode_tag}_random_context_{current_time}_ICL.json'
)

os.makedirs(os.path.dirname(save_filename), exist_ok=True)

# 保存包含所有 Prompt 结果的 JSON
with open(save_filename, 'w') as f:
    json.dump(final_results, f, indent=4)

print("="*50)
print(f"Detailed evaluation metrics saved to:\n{save_filename}")
print("="*50)
