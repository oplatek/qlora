#!/bin/bash
# model_name_or_path="EleutherAI/pythia-12b"
# model_name_or_path="EleutherAI/pythia-70m"
model_name_or_path="$1"
python qlora.py \
    --model_name_or_path "$model_name_or_path" \
    --output_dir ./output \
    --do_train False \
    --do_eval False \
    --do_predict True \
    --predict_with_generate \
    --per_device_eval_batch_size 4 \
    --dataset alpaca \
    --source_max_len 512 \
    --target_max_len 128 \
    --max_new_tokens 64 \
    --do_sample \
    --top_p 0.9 \
    --num_beams 1 \
