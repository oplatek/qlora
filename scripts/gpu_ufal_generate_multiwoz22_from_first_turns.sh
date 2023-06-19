#!/bin/bash
set -euo pipefail

# model_name_or_path="EleutherAI/pythia-12b"
# model_name_or_path="EleutherAI/pythia-70m"
model_name_or_path="$1"
checkpoint_dir="$2"

dataset=multi_woz_v22
dataset_format=multi_woz_v22_turns
num_samples=10
# Let's store the output dir based on the model name
# If the PEFT weights are saved in checkpoint_dir,
#  store the output in the checkpoint subdirectory.
output_dir=output/"$(echo $checkpoint_dir | sed 's:output/::')"/pred_${dataset_format}_${num_samples}_$(date +'%s')/

gpu-python --gpu-mem 40 \
  qlora.py \
    --dataloader_num_workers 0 \
    --max_eval_samples $num_samples \
    --model_name_or_path "$model_name_or_path" \
    --checkpoint_dir "$checkpoint_dir" \
    --output_dir "$output_dir" \
    --do_train False \
    --do_eval False \
    --do_predict True \
    --predict_with_generate \
    --per_device_eval_batch_size 4 \
    --dataset $dataset \
    --dataset_format $dataset_format \
    --source_max_len 256 \
    --target_max_len 288 \
    --max_new_tokens 32 \
    --do_sample \
    --top_p 0.9 \
    --num_beams 1 \

# you may want to evaluate your models using 
INP_DIALOG=$output_dir/predictions_dialogs.json
python evaluate.py --richness --bleu --success --input $INP_DIALOG  --output ${INP_DIALOG%jsonl}results.json
