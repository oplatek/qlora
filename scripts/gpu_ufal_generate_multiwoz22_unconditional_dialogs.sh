#!/bin/bash
set -euo pipefail

# model_name_or_path="EleutherAI/pythia-12b"
# model_name_or_path="EleutherAI/pythia-70m"
model_name_or_path="$1"
checkpoint_dir="$2"

dataset=multi_woz_v22
dataset_format=multi_woz_v22_dialogs
num_samples=10
# Let's store the output dir based on the model name
# If the PEFT weights are saved in checkpoint_dir,
#  store the output in the checkpoint subdirectory.
output_dir=output/"$(echo $checkpoint_dir | sed 's:output/::')"/pred_${dataset_format}_${num_samples}_$(date +'%s')/

# Either login to a machine with GPU and use python or submit your script to a gpu-machine e.g. using https://github.com/oplatek/shellgit-ufal/blob/master/bin/gpu-python
# python \
gpu-python --gpu-mem 40 \
  qlora.py \
    --dataloader_num_workers 3 \
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
    --source_max_len 2 \
    --target_max_len 256 \
    --max_new_tokens 256 \
    --do_sample \
    --top_p 0.9 \
    --num_beams 1 \
