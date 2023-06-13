#!/bin/bash
# gpu-python
#  is script for slurm
#  see https://github.com/oplatek/shellgit-ufal/blob/master/bin/gpu-python
timestamp="$(date +'%s')"
export WANDB_ENTITY=metric
export WANDB_PROJECT=llm_finetune_multiwoz22.sh
# based on https://aclanthology.org/D18-1547.pdf
# table 1
# avg turns per dialogues 13.68
# avg tokens per turn     13.18
# --> avg dialogue has ~ 200 tokens

# python \
gpu-python --gpu-mem 40 \
  qlora.py \
    --do_predict true \
    --model_name_or_path huggyllama/llama-7b \
    --source_max_len 512 \
    --target_max_len 64 \
    --output_dir ./output/multiwoz22-7b_${timestamp}_$$ \
    --report_to wandb \
    --dataset multi_woz_v22 \
    --dataset_format multi_woz_v22_dialogs \
    --logging_steps 10 \
    --save_strategy steps \
    --data_seed 42 \
    --save_steps 500 \
    --save_total_limit 40 \
    --evaluation_strategy steps \
    --eval_dataset_size 1024 \
    --max_eval_samples 1000 \
    --per_device_eval_batch_size 1 \
    --max_new_tokens 32 \
    --dataloader_num_workers 3 \
    --logging_strategy steps \
    --remove_unused_columns False \
    --do_train \
    --do_eval \
    --lora_r 64 \
    --lora_alpha 16 \
    --lora_modules all \
    --double_quant \
    --quant_type nf4 \
    --bf16 \
    --bits 4 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type constant \
    --gradient_checkpointing \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --max_steps 1875 \
    --eval_steps 187 \
    --learning_rate 0.0002 \
    --adam_beta2 0.999 \
    --max_grad_norm 0.3 \
    --lora_dropout 0.1 \
    --weight_decay 0.0 \
    --seed 0 \
    --group_by_length false

    # Commented out options needs to at the end
    # without group_by_length it will be less efficient
    # but 1) I do not get how it affects the training if batch_size is 1
    #     2) without it learns first, second, third, etc turns  together  
    # --group_by_length \
    # --do_mmlu_eval \

