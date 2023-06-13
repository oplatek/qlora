#!/bin/bash
# gpu-python
#  is script for slurm
#  see https://github.com/oplatek/shellgit-ufal/blob/master/bin/gpu-python
timestamp="$(date +'%s')"
gpu-python --gpu-mem 48 qlora.py \
    --model_name_or_path huggyllama/llama-7b \
    --output_dir ./output/guanaco-7b_${timestamp}_$$ \
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
    --group_by_length \
    --logging_strategy steps \
    --remove_unused_columns False \
    --do_train \
    --do_eval \
    --do_mmlu_eval \
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
    --dataset oasst1 \
    --source_max_len 16 \
    --target_max_len 512 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --max_steps 1875 \
    --eval_steps 187 \
    --learning_rate 0.0002 \
    --adam_beta2 0.999 \
    --max_grad_norm 0.3 \
    --lora_dropout 0.1 \
    --weight_decay 0.0 \
    --seed 0

# oplatek@dll-8gpu2
# ***** train metrics *****
#   epoch                    =        3.4
#   train_loss               =     0.9947
#   train_runtime            = 8:10:21.54
#   train_samples_per_second =       1.02
#   train_steps_per_second   =      0.064
# 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1000/1000 [03:46<00:00,  4.42it/s]
# 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1531/1531 [09:13<00:00,  2.76it/s]
# ***** eval metrics *****
#   epoch                   =        3.4
#   eval_loss               =     1.7491
#   eval_runtime            = 0:03:46.51
#   eval_samples_per_second =      4.415
#   eval_steps_per_second   =      4.415
#
# {'loss': 0.4886, 'learning_rate': 0.0002, 'epoch': 3.39}                                                                                                        
# {'eval_loss': 1.735099196434021, 'eval_runtime': 226.555, 'eval_samples_per_second': 4.414, 'eval_steps_per_second': 4.414, 'epoch': 3.39}                     
# 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1531/1531 [09:13<00:00,  2.76it/s]
# {'mmlu_loss': 2.9475739900774616, 'mmlu_eval_accuracy_high_school_microeconomics': 0.2692307692307692, 'mmlu_eval_accuracy_moral_disputes': 0.3157894736842105, 'mmlu_eval_accurac
# y_high_school_computer_science': 0.3333333333333333, 'mmlu_eval_accuracy_medical_genetics': 0.18181818181818182, 'mmlu_eval_accuracy_us_foreign_policy': 0.2727272727272727, 'mmlu
# _eval_accuracy_high_school_geography': 0.4090909090909091, 'mmlu_eval_accuracy_college_chemistry': 0.375, 'mmlu_eval_accuracy_human_aging': 0.2608695652173913, 'mmlu_eval_accurac
# y_philosophy': 0.38235294117647056, 'mmlu_eval_accuracy_professional_accounting': 0.3225806451612903, 'mmlu_eval_accuracy_college_computer_science': 0.18181818181818182, 'mmlu_ev
# al_accuracy_high_school_government_and_politics': 0.42857142857142855, 'mmlu_eval_accuracy_high_school_world_history': 0.19230769230769232, 'mmlu_eval_accuracy_elementary_mathema
# tics': 0.3170731707317073, 'mmlu_eval_accuracy_logical_fallacies': 0.5, 'mmlu_eval_accuracy_sociology': 0.45454545454545453, 'mmlu_eval_accuracy_astronomy': 0.3125, 'mmlu_eval_ac
# curacy_high_school_mathematics': 0.2413793103448276, 'mmlu_eval_accuracy_professional_law': 0.27058823529411763, 'mmlu_eval_accuracy_professional_psychology': 0.2608695652173913,
#  'mmlu_eval_accuracy_formal_logic': 0.2857142857142857, 'mmlu_eval_accuracy_public_relations': 0.5, 'mmlu_eval_accuracy_abstract_algebra': 0.36363636363636365, 'mmlu_eval_accurac
# y_high_school_physics': 0.47058823529411764, 'mmlu_eval_accuracy_machine_learning': 0.18181818181818182, 'mmlu_eval_accuracy_business_ethics': 0.2727272727272727, 'mmlu_eval_accu
# racy_professional_medicine': 0.2903225806451613, 'mmlu_eval_accuracy_nutrition': 0.30303030303030304, 'mmlu_eval_accuracy_global_facts': 0.3, 'mmlu_eval_accuracy_world_religions'
# : 0.3684210526315789, 'mmlu_eval_accuracy_high_school_statistics': 0.30434782608695654, 'mmlu_eval_accuracy_high_school_us_history': 0.45454545454545453, 'mmlu_eval_accuracy_coll
# ege_medicine': 0.18181818181818182, 'mmlu_eval_accuracy_management': 0.5454545454545454, 'mmlu_eval_accuracy_high_school_biology': 0.53125, 'mmlu_eval_accuracy_conceptual_physics
# ': 0.2692307692307692, 'mmlu_eval_accuracy_anatomy': 0.42857142857142855, 'mmlu_eval_accuracy_college_mathematics': 0.2727272727272727, 'mmlu_eval_accuracy_jurisprudence': 0.4545
# 4545454545453, 'mmlu_eval_accuracy_marketing': 0.44, 'mmlu_eval_accuracy_college_biology': 0.3125, 'mmlu_eval_accuracy_high_school_macroeconomics': 0.2558139534883721, 'mmlu_eval
# _accuracy_security_studies': 0.25925925925925924, 'mmlu_eval_accuracy_clinical_knowledge': 0.3103448275862069, 'mmlu_eval_accuracy_computer_security': 0.36363636363636365, 'mmlu_
# eval_accuracy_human_sexuality': 0.08333333333333333, 'mmlu_eval_accuracy_high_school_psychology': 0.5333333333333333, 'mmlu_eval_accuracy_international_law': 0.3076923076923077, 
# 'mmlu_eval_accuracy_moral_scenarios': 0.22, 'mmlu_eval_accuracy_econometrics': 0.08333333333333333, 'mmlu_eval_accuracy_prehistory': 0.37142857142857144, 'mmlu_eval_accuracy_high
# _school_european_history': 0.3888888888888889, 'mmlu_eval_accuracy_virology': 0.3333333333333333, 'mmlu_eval_accuracy_college_physics': 0.45454545454545453, 'mmlu_eval_accuracy_h
# igh_school_chemistry': 0.3181818181818182, 'mmlu_eval_accuracy_electrical_engineering': 0.125, 'mmlu_eval_accuracy_miscellaneous': 0.4883720930232558, 'mmlu_eval_accuracy': 0.328
# 24898613704895, 'epoch': 3.39}                                                           
# {'train_runtime': 29421.5443, 'train_samples_per_second': 1.02, 'train_steps_per_second': 0.064, 'train_loss': 0.9947136127471924, 'epoch': 3.4}   
