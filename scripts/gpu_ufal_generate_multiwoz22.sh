#!/bin/bash
# model_name_or_path="EleutherAI/pythia-12b"
# model_name_or_path="EleutherAI/pythia-70m"
model_name_or_path="$1"

dataset=multi_woz_v22
dataset_format=multi_woz_v22_dialogs
# dataset_format=multi_woz_v22_turns  # this fails
output_dir=output/"$(echo $model_name_or_path | sed 's:output/::')"/pred_${dataset_format}/

gpu-python --gpu-mem 40 \
  qlora.py \
    --model_name_or_path "$model_name_or_path" \
    --output_dir "$output_dir" \
    --do_train False \
    --do_eval False \
    --do_predict True \
    --predict_with_generate \
    --per_device_eval_batch_size 4 \
    --dataset $dataset \
    --dataset_format $dataset_format \
    --source_max_len 512 \
    --target_max_len 128 \
    --max_new_tokens 64 \
    --do_sample \
    --top_p 0.9 \
    --num_beams 1 \
