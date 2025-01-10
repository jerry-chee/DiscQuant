#!/bin/bash 

model_id="microsoft/Phi-3-mini-4k-instruct"
dataset="red_concat"
output_path="phi3/greedy"
wbits="3"
groupsize="32"

mkdir -p $output_path
python discq.py \
    --model_id $model_id \
    --dataset $dataset \
    --early_savedir "$output_path/checkpoints" \
    --wbits $wbits --groupsize $groupsize --no-quip \
    --early_save_mode greedy

function single_eval {
  local load_name=$1
  local model_id=$2
  local output_name=$3
  local batch_size=$4
  local task=$5
  local num_shot=$6
  lm_eval \
    --model hf \
    --model_args pretrained=$load_name,tokenizer=$model_id,trust_remote_code=True,dtype=float16 \
    --tasks $task \
    --num_fewshot $num_shot \
    --batch_size $batch_size \
    --output_path "${output_name}/${task}.json"
}

batch_size=32
mmlu_bs=8
if [ ! -f "${log_dir}/gsm8k_cot.json" ] || [ $use_ckpt = "false" ]; then
    single_eval "$output_path/checkpoints" $model_id "$output_path/logs" $batch_size gsm8k_cot 8
fi
if [ ! -f "${log_dir}/wikitext.json" ] || [ $use_ckpt = "false" ]; then
    single_eval "$output_path/checkpoints" $model_id "$output_path/logs" $batch_size wikitext 0
fi
if [ ! -f "${log_dir}/piqa.json" ] || [ $use_ckpt = "false" ]; then
    single_eval "$output_path/checkpoints" $model_id "$output_path/logs" $batch_size piqa 0
fi
if [ ! -f "${log_dir}/arc_challenge.json" ] || [ $use_ckpt = "false" ]; then
    single_eval "$output_path/checkpoints" $model_id "$output_path/logs" $batch_size arc_challenge 0
fi
if [ ! -f "${log_dir}/mmlu.json" ] || [ $use_ckpt = "false" ]; then
    single_eval "$output_path/checkpoints" $model_id "$output_path/logs" $mmlu_bs mmlu 5
fi
if [ ! -f "${log_dir}/hellaswag.json" ] || [ $use_ckpt = "false" ]; then
    single_eval "$output_path/checkpoints" $model_id "$output_path/logs" $batch_size hellaswag 0
fi
if [ ! -f "${log_dir}/winogrande.json" ] || [ $use_ckpt = "false" ]; then
    single_eval "$output_path/checkpoints" $model_id "$output_path/logs" $batch_size winogrande 0
fi