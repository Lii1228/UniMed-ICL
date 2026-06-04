import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import os
import json
import numpy as np
import types
from datetime import datetime
import warnings
import sys

# --- 获取项目根目录并加入系统路径 ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from dataloader import MetaDataset_Multi_Extended
from config import get_parser

# 尝试导入评估指标库
try:
    from utils.pairwise_measures import BinaryPairwiseMeasures as PM
except ImportError:
    print("Warning: Could not import BinaryPairwiseMeasures. Metrics might be unavailable.")
    PM = None

# ================= 1. 解析参数与开源配置加载 =================
args = get_parser().parse_args()

print("="*60)
print("Running UniMed-ICL Text-Prompt Mode in Open-Source")
print("="*60)

# 指向专门的文本配置文件
sample_config_path = os.path.join(PROJECT_ROOT, 'Brain','Brain.json')
# 强制将数据目录指向根目录 (Dataloader 会拼接 sample_data/Dataset...)
args.data_dir = PROJECT_ROOT

try:
    with open(sample_config_path, 'r') as f:
        data_loading_config = json.load(f)
    current_task_name = data_loading_config[0]['name']
    print(f"Successfully loaded open-source text config for: {current_task_name}")
    print(f"Config details: {len(data_loading_config)} dataset(s) loaded")
except Exception as e:
    raise RuntimeError(f"Failed to load sample text config. Please ensure 'sample_config_text.json' exists. Error: {e}")

# 加载 Checkpoint
assert args.checkpoint_path is not None, "The checkpoint_path should not be None!"
checkpoint_path = args.checkpoint_path
tmp = sorted([i for i in os.listdir(checkpoint_path) if i.endswith('.ckpt') or i.endswith('.pth')])
if args.checkpoint_index < 0:
    checkpoint_idx = len(tmp) + args.checkpoint_index
else:
    checkpoint_idx = args.checkpoint_index

if 0 <= checkpoint_idx < len(tmp):
    selected_ckpt = tmp[checkpoint_idx]
else:
    selected_ckpt = tmp[-1] 

checkpoint_path = os.path.join(checkpoint_path, selected_ckpt)
print(f'Loading checkpoint from: {checkpoint_path}')

# ================= 2. 加载模型 =================
model_module = f'from {args.model_name}.lightning_model import LightningModel'
print(model_module)
exec(model_module)

warnings.filterwarnings('ignore')
model = LightningModel.load_from_checkpoint(checkpoint_path, map_location=torch.device(args.device))

model.to(args.device)
model.eval()

# 强制模式设置 (强制 3D 模式用于评估)
model.prob_2d = 0      
model.num_slices = 1   

# ================= 3. 注入白名单 =================
target_dataset_names = []
if isinstance(data_loading_config, list):
    for item in data_loading_config:
        if 'name' in item:
            target_dataset_names.append(item['name'])
elif isinstance(data_loading_config, dict):
    if 'name' in data_loading_config:
        target_dataset_names.append(data_loading_config['name'])

print(f"Datasets in this task: {target_dataset_names}")

if hasattr(model, 'text_enabled_datasets'):
    updated_list = list(set(model.text_enabled_datasets + target_dataset_names))
    model.text_enabled_datasets = updated_list
    print(f"Updated Text Whitelist: {model.text_enabled_datasets}")

# ================= 4. 数据加载 =================
dataset_val = MetaDataset_Multi_Extended(
        dataset_dir = args.data_dir, 
        skip_resize = args.skip_resize,
        data_loading_config = data_loading_config, 
        train_or_val = 'val',
        train_or_val_split_rate = args.train_or_val_split_rate,
        group_size = args.context_size+1,
        cut_length = True 
)
dataset_val.len_divid = 4 

dataloader_val = DataLoader(dataset_val, 
                            batch_size=args.batch_size,
                            num_workers=args.workers,
                            pin_memory=True,
                            persistent_workers=True)

# ================= 5. 【核心修改】Monkey Patching validation_step =================

def patched_validation_step(self, batch, batch_idx):
    # 1. 获取当前验证场景配置
    config = getattr(self, 'eval_config', {})
    
    target_prompt_type = config.get('visual_prompt', 'dense') 
    enable_text = config.get('enable_text', False)
    remove_visual = config.get('remove_visual', False) 
    
    dataset_name = batch.get('dataset', ['Unknown'])[0]
    task = batch['task'][0]

    def process_input(input_mod):
        if isinstance(input_mod, str): return input_mod
        elif isinstance(input_mod, torch.Tensor): return int(input_mod.item())
        else: return input_mod
    
    input_mod = process_input(batch.get('input', [''])[0])
    output_mod = process_input(batch.get('output', [''])[0])
    
    # 2. 解析 3D 数据
    imgs = batch['image'][0, :, None, :]
    labs = batch['label'][0, :, None, :]
    
    target_in = imgs[:1, :]           
    target_out = labs[:1, :]
    context_in = imgs[None, 1:, :]    
    context_label_dense = labs[None, 1:, :]
    
    L = context_in.shape[1]
    
    # 3. 处理文本 (Text Control)
    language_tokens = None
    if enable_text and dataset_name in self.text_enabled_datasets:
        # 强制 batch_size=1
        target_ids = self._get_target_ids(batch, 1)
        language_tokens = self.text_adapter(target_ids, self.device)
        
    # 4. 处理视觉上下文 (Physical Removal / Scheme C)
    if remove_visual:
        # 严格匹配 target_in 的五维结构: (B, C, W, H, D)
        B_dim, C_dim, W_dim, H_dim, D_dim = target_in.shape
        # 创建 L=0 的空 Tensor
        context_in = torch.zeros((B_dim, 0, C_dim, W_dim, H_dim, D_dim), device=self.device)
        context_prompt = torch.zeros((B_dim, 0, C_dim, W_dim, H_dim, D_dim), device=self.device)
    else:
        # 正常生成指定的 Prompt
        context_prompt = self.generate_sparse_prompt(
            context_label_dense.to(self.device), 
            is_train=False, 
            specific_prompt_types=[target_prompt_type] * L
        )
    
    # 5. 调用模型前向传播
    mask = self.forward(
        target_in.to(self.device), 
        context_in.to(self.device), 
        context_prompt.to(self.device), 
        language_tokens=language_tokens
    )
    
    # 6. 计算 Metrics
    if PM is not None:
        try:
            p_pred = (mask > 0.5).float().cpu().numpy()
            p_ref = target_out.float().cpu().numpy()
            
            if p_pred.sum() == 0 and p_ref.sum() == 0:
                dsc = 1.0
            else:
                bpm = PM(p_pred, p_ref, dict_args={"nsd": 1, "hd_perc": 95})
                dsc = bpm.dsc()
                
            # 分别记录带前后缀的和干净的 DSC，方便主程序抓取
            self.log(f'{dataset_name}_{str(input_mod)}{str(output_mod)}_{task}_DSC', float(dsc))
            self.log(f'{dataset_name}_DSC', float(dsc))
            
        except Exception as e: 
            pass

    return torch.tensor(0.0, device=self.device) # 返回 dummy loss 防止报错

# === 动态替换模型的方法 ===
model.eval_config = {} 
model.validation_step = types.MethodType(patched_validation_step, model)
print(">> Successfully patched model.validation_step for Scheme C evaluation.")

# ================= 6. 评估循环 =================
trainer = pl.Trainer(logger=False, enable_checkpointing=False, accelerator='auto', devices=1)

# 定义测试场景 (Scheme C)
eval_scenarios = [
    # --- 1. 纯文本模式 (Text Only) ---
    # 关键点: remove_visual=True (触发 L=0 物理置空), enable_text=True
    {'name': 'Text_Only_SchemeC', 'visual_prompt': 'none', 'enable_text': True,  'remove_visual': True}, 
]

final_results = {}

print("\n" + "="*40)
print("STARTING EVALUATION LOOP (Scheme C)")
print("="*40)

for scenario in eval_scenarios:
    run_name = scenario['name']
    print(f"\n>>> Running Scenario: [{run_name}]")
    print(f"    Config: {scenario}")
    
    # 注入配置
    model.eval_config = scenario
    
    # ====== 跳过可视化，避免权限报错 ======
    model.visualize_results = lambda *args, **kwargs: None
    
    # 运行验证
    trainer.validate(model, dataloaders=dataloader_val)
    
    # 收集结果，确保安全转换
    metrics = {k: (v.item() if isinstance(v, torch.Tensor) else v) for k, v in model.trainer.callback_metrics.items()}
    
    # 尝试抓取主要的 DSC 指标
    main_dsc_key = f"{target_dataset_names[0]}_DSC" if target_dataset_names else "DSC"
    main_dsc = metrics.get(main_dsc_key, "N/A")
    if main_dsc == "N/A":
        for k, v in metrics.items():
            if k.endswith('DSC'):
                main_dsc = v
                break
                
    final_results[run_name] = metrics
    print(f">>> Scenario [{run_name}] Finished. Main DSC: {main_dsc}")

# ================= 7. 保存结果 =================
current_time = datetime.now().strftime("%H%M%S")

# --- 【核心修改 2：统一开源结果保存目录】 ---
safe_save_dir = os.path.join(PROJECT_ROOT, "opensource_eval_results")
os.makedirs(safe_save_dir, exist_ok=True)

base_ckpt_name = os.path.basename(checkpoint_path)
new_file_name = base_ckpt_name.replace(
    '.ckpt', 
    f'_TextPrompt_Sample_{current_task_name}_{current_time}_results.json'
).replace('.pth', f'_TextPrompt_Sample_{current_task_name}_{current_time}_results.json')

save_filename = os.path.join(safe_save_dir, new_file_name)

print("\nFinal Results Dict:", json.dumps(final_results, indent=4))

with open(save_filename, 'w') as f:
    json.dump(final_results, f, indent=4)

print("="*60)
print(f"Results saved to: {save_filename}")
print("="*60)