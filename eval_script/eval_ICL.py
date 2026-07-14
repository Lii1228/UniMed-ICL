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

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from config import get_parser

args = get_parser().parse_args()
print(args)

sample_config_path = os.path.join(PROJECT_ROOT, 'Liver', 'Liver.json')

args.data_dir = PROJECT_ROOT

try:
    with open(sample_config_path, 'r') as f:
        data_loading_config = json.load(f)
    print(f"Successfully loaded open-source config from {sample_config_path}")
    print('The current sample task is:', data_loading_config[0]['name'])
except Exception as e:
    raise RuntimeError(f"Failed to load sample config. Please make sure sample_config.json exists in the root directory. Error: {e}")
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

# ================= Environment and Strategy Controls =================
print("!!! Forcing Model Controls !!!")

model.prob_2d = 0  
model.num_slices = 1 

model._sample_strategy = lambda dataset_name: 0
print("-> Evaluation Mode ON: Model strategy forced to 0 (Random Visual Context only).")
# =====================================================================

total_params = sum(p.numel() for p in model.parameters())
print("Total number of parameters: ", total_params)
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Number of trainable parameters: ", trainable_params)

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

trainer = pl.Trainer(logger=False, enable_checkpointing=False)

target_prompt_types = ['dense']

final_results = {}

print("="*50)
print(f"Starting Individual Prompt Evaluation for: {target_prompt_types}")
print("="*50)

for p_type in target_prompt_types:
    print(f"\n>>> Evaluating Prompt Type: [ {p_type} ] ...")
    
    model.prompt_types_2d = [p_type] 
    model.prompt_types_3d = [p_type] 
    
    model.visualize_results = lambda *args, **kwargs: None
    
    trainer.validate(model, dataloaders=dataloader_val)
    
    metrics = model.trainer.callback_metrics
    
    metrics_dict = {key: value.item() for key, value in metrics.items() if isinstance(value, torch.Tensor)}
   
    final_results[p_type] = metrics_dict
    
    print(f">>> Result for {p_type}: {metrics_dict}")

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
    f'_FLARE22_Sample_context{str(args.context_size)}{mode_tag}_random_context_{current_time}_ICL.json'
)

save_filename = os.path.join(safe_save_dir, new_file_name)

print("\nFinal Results Dict:", json.dumps(final_results, indent=4))

with open(save_filename, 'w') as f:
    json.dump(final_results, f, indent=4)

print("="*50)
print(f"Detailed evaluation metrics saved to:\n{save_filename}")
print("="*50)
