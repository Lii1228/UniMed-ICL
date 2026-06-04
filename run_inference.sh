#!/bin/bash



CUDA_DEVICE=0
CKPT_PATH="./UniMed_ICL/checkpoints" 


CONTEXT_SIZE=1
TASK_IDX=0

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

