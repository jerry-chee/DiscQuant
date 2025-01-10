#!/bin/bash

model_id="microsoft/Phi-3-mini-4k-instruct"
dataset="red_concat"
output_path="phi3/discq"
wbits="3"
groupsize="32"
clamp="1.0"
rhof="200"
lr="0.1"
grad_accum="8"

mkdir -p $output_path
python discq.py \
    --model_id $model_id \
    --dataset $dataset \
    --output $output_path \
    --wbits $wbits --groupsize $groupsize --no-quip \
    --clamp $clamp --rho_factor $rhof --learning_rate $lr \
    --grad_accum $grad_accum --save_model

python lmeval.py \
    --model_id $model_id \
    --dataset $dataset \
    --output $output_path \
    --wbits $wbits --groupsize $groupsize --no-quip \
    --clamp $clamp --rho_factor $rhof --learning_rate $lr \
    --grad_accum $grad_accum --save_model