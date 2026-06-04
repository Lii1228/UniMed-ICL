#!/bin/bash

# ==============================================================
# UniMed-ICL 推理脚本 
# ==============================================================

# 1. 基础配置
CUDA_DEVICE=0
CKPT_PATH="./UniMed_ICL/checkpoints" 

# 2. 运行参数
CONTEXT_SIZE=1
TASK_IDX=0

echo "🚀 开始运行 UniMed-ICL 评估..."
echo "---------------------------------"
echo "模型权重: $CKPT_PATH"
echo "上下文数量: $CONTEXT_SIZE"
echo "---------------------------------"

# 3. 执行 Python 脚本
CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python eval/eval_text.py \
    --checkpoint_path $CKPT_PATH \
    --checkpoint_index 2 \
    --model_name UniMed_ICL \
    --context_size $CONTEXT_SIZE \
    --task_idx $TASK_IDX \
    --skip_resize True \
    --len_divid 1 \
    --train_or_val_split_rate 0.95 \
    --batch_size 1 \

echo "✅ 评估完成！详细结果已保存至 ./opensource_eval_results/ 目录。"
