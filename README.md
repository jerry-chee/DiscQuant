# DiscQuant: A Quantization Method for Neural Networks Inspired by Discrepancy Theory

This is a rounding method which produces synthetic quantized weights. 
The code currently supports Phi-3 and Llama-3 models, but can easily be extended to other transformer architectures.

---
### Setup
To reproduce the results of our paper, we use 2 conda environments, one for the Phi-3 class of models, and the other for the Llama-3.1 class of models.

```bash
conda create --name discquant_phi3 python=3.10 -y
conda run --no-capture-output -n discquant_phi3 python -m pip install sentencepiece protobuf langdetect immutabledict matplotlib wandb torch==2.2.1 transformers==4.39.0 datasets==2.17.1 numpy==1.26.4 lm-eval==0.4.2 peft==0.10.0
conda run --no-capture-output -n discquant_phi3 python -m pip install packaging ninja flash-attn==2.6.1 --no-build-isolation fast-hadamard-transform
```

```bash
conda create --name discquant_lam3 pip python=3.10 -y
conda run --no-capture-output -n discquant_lam3 pip install sentencepiece protobuf langdetect immutabledict matplotlib wandb torch==2.2.1 transformers==4.44.0 datasets==2.17.1 numpy==1.26.4 lm-eval==0.4.2 peft==0.10.0
conda run --no-capture-output -n discquant_lam3 pip install packaging ninja flash-attn==2.6.1 --no-build-isolation fast-hadamard-transform
```

---

### Description
Our method operates in the continuous weight space, and aims to find solutions with minimal quantization error and contain a high number of rounded parameters.
We use knowledge distillation to ensure the solution has low quantization error.
We add a linear regularization term which encourages many parameters to be rounded.
At the end of our procedure, the remaining few unrounded parameters are greedily rounded.
Please see our paper [TBD Link]() for further details.
Our method is inspired by theoretical insights we derive using discrepancy theory.

Our code contains a wrapper linear class which holds a frozen copy of the original weights, and an unfrozen `[0,1]` version of the weights which interpolate between the nearest down and up points in the quantization grid.
We wrap all relevant linear layers in the original model with this wrapper class in order to run our method.
Our method is agnostic to the quantization grid used. 
We implement a base quantization grid class which can be used to wrap any quantization grid, and only requires that `round_down(W)` and `round_up(W)` functions are implemented which return the element-wise closest down or up quantization grid points.
Currently we are performing synthetic quantization, where the weights are quantized with the correct number of unique elements, but saved in the original datatype. This means that we do not save the quantization scale factors separately.

We also copy and modify the [GPTQ code](https://github.com/IST-DASLab/marlin) in order to run this baseline using the same quantization grid code.

- `discq.py`         : The main file, containing training code. Run this to produce a quantized model.
- `lmeval.py`        : Runs lm-evaluation-harness on a pre-determined set of tasks. Call with the same arguments as discquant.py
- `linearutils.py`   : Contains wrapper quantized linear class, each linear layer to be quantized will be wrapped.
- `quantutils.py`    : Quantization base class and wrapped implementation of block scaling from gptq.
- `quiputils.py`     : Mostly copied implementation of the Random Hadamard Transform to perform incoherence processing from [QuIP#](https://github.com/Cornell-RelaxML/quip-sharp).
- `utils.py`         : Various helper functions, including `parse_args()` which specifices all arguments.
- `svd.py`           : Script to compute singular values of LLM gradients.
- `gptq/datautils.py`: Creates quantization dataset. We've added several datasets.
- `gptq/quant.py`    : (Unmodified) GPTQ block scaling quantization.
- `gptq/gptq.py`     : (Unmodified) GPTQ algorithm.
- `gptq/llama.py`    : Running GPTQ with Llama model class.
- `gptq/phi3.py`     : We wrote this code, adapting llama.py to work with the Phi-3 model class.

---
### How to Run the Code
#### DiscQuant
To run our method, run `discq.py` with the below arguments. 
```bash
python discq.py  \
    --model_id <HF/path/to/model> \
    --dataset <dataset> \
    --output <output> \
    --wbits <wbits> --groupsize <groupsize> \
    --number_of_iterations <num_iter> --grad_accum 8 \
    --learning_rate <lr> --clamp <clamp> --rhof 200 \
    --save <savepath> \
    --dtype bfloat16 \
```

Evaluate with `lmeval.py`, using the same arguments.
To use incoherence processing, add the `--quip` argument.

To replicate our experiments, here are the following hyperparameters. `scripts/discq.sh` is an exmaple script of how to run for a single bit setting of `wbits`, `groupsize`.

**Phi3-mini-4k-instruct Block Scaling**

| wbits | groupsize | lr  | clamp |
| ----- | --------- | --- | ----- |
| 3---- | -1        | 0.1 | 1.0   | 
| 3---- | 64        | 0.1 | 0.5   | 
| 3---- | 32        | 0.1 | 1.0   | 
| 4---- | -1        | 0.1 | 1.0   | 
| 4---- | 64        | 0.1 | 0.5   | 
| 4---- | 32        | 0.05| 0.5   |

**Lama3.1-8b-instruct Block Scaling**

| wbits | groupsize | lr  | clamp |
| ----- | --------- | --- | ----- |
| 3---- | -1        | 0.05| 0.5   | 
| 3---- | 64        | 0.05| 1.0   | 
| 3---- | 32        | 0.1 | 0.5   | 
| 4---- | -1        | 0.05| 1.0   | 
| 4---- | 64        | 0.05| 1.0   | 
| 4---- | 32        | 0.1 | 0.5   |

**Phi3-mini-4k-instruct QuIP**

| wbits | groupsize | lr  | clamp |
| ----- | --------- | --- | ----- |
| 3---- | -1        | 0.05| 0.05  | 
| 3---- | 64        | 0.01| 0.05  | 
| 3---- | 32        | 0.01| 0.05  | 
| 4---- | -1        | 0.05| 0.05  | 
| 4---- | 64        | 0.01| 0.05  |
| 4---- | 32        | 0.05| 0.05  |

**Lama3.1-8b-instruct QuIP**

| wbits | groupsize | lr  | clamp |
| ----- | --------- | --- | ----- |
| 3---- | -1        | 0.05| 0.05  | 
| 3---- | 64        | 0.05| 0.05  | 
| 3---- | 32        | 0.01| 0.05  |
| 4---- | -1        | 0.05| 0.05  |
| 4---- | 64        | 0.01| 0.05  |
| 4---- | 32        | 0.01| 0.05  |


#### Greedy
To run greedy rounding, run `discq.py` with `--early_save_mode greedy` and the appropriate ``--wbits, --groupsize` arguments.

#### GPTQ
To run GPTQ, run `gptq/phi3.py` or `gptq/llama.py` with the following arguments.
```bash
python gptq/phi3.py \
    <HF/path/to/model> \
    --dataset <dataset> \
    --nsamples <nsamples> \
    --wbits <wbits> --groupsize <groupsize> \
    --true_sequential --actorder \
    --save <save>
```
We evaluate using the lm-evlauation-harness.
We tune the number of samples (1024 4096, 8192), here are the parameters we used:

**Phi3-mini-4k-instruct Block Scaling**

| wbits | groupsize | nsamples |
| ----- | --------- | -------- |
| 3     | -1        | 4096     |
| 3     | 32        | 4096     | 
| 3     | 64        | 4096     |
| 4     | -1        | 1024     |
| 4     | 32        | 1024     |
| 4     | 64        | 1024     |

**Lama3.1-8b-instruct Block Scaling**

| wbits | groupsize | nsamples |
| ----- | --------- | -------- |
| 3     | -1        | 4096     |
| 3     | 32        | 8192     | 
| 3     | 64        | 8192     |
| 4     | -1        | 4096     |
| 4     | 32        | 8192     |
| 4     | 64        | 1024     |

**Phi3-mini-4k-instruct QuIP**

| wbits | groupsize | nsamples |
| ----- | --------- | -------- |
| 3     | -1        | 8192     |
| 3     | 32        | 8192     | 
| 3     | 64        | 1024     |
| 4     | -1        | 8192     |
| 4     | 32        | 1024     |
| 4     | 64        | 4096     |

**Lama3.1-8b-instruct QuIP**

| wbits | groupsize | nsamples |
| ----- | --------- | -------- |
| 3     | -1        | 1024     |
| 3     | 32        | 8192     | 
| 3     | 64        | 8192     |
| 4     | -1        | 8192     |
| 4     | 32        | 1024     |
| 4     | 64        | 8192     |


### Arguments
#### Arguments for discq.py, lmeval.py
- `model_id`         : HuggingFace model path.
- `device_map`       : Transformers from_pretrained() device_map argument, choices=['auto','balanced','balanced_low_0'].
- `dataset`          : Quantization dataset, choices=['wikitext2','wikitext2_partition','red','red_concat','gsm','gsm_concat']. `red_concat` is default.
- `output`           : Output folder; saves quantized model in `checkpoints/` subfolder, and logs and evaluation outputs in `logs/` subfolder.
- `seed`             : Seed for reproducibility.
- `init_x`           : Initialization method, default='rand', choices=['rand','orig'].
- `wbits`            : Number of bits to quantize to.
- `groupsize`        : Every <> parameters shares the same scale factor, default=-1 meaning that each row receives a unique scale factor.
- `optimizer`        : Default='AdamW', choices=['SGD','AdamW'].
- `lr_sched`         : Default='cosine', choices=['linear','cosine'].
- `number_of_samples`: Default=1024.
- `number_of_samples_val`: Default=64.
- `number_of_warmup' : Default=128.
- `batch_size`       : Default=1. 
- `grad_accum`       : Gradient accumulation. Also influences total batch size.
- `number_of_epochs` : Default=1.
- `bs_iter_fixed`    : Default=False, action=argparse.BooleanOptionalAction, help='keeps iterations fixed when increase bs')
- `window_size`      : Sequence length for quantization dataset, default=2048.
- `rho_factor`       : Scales the distillation gradient.
- `clamp_value`      : Entry-wise gradient clamping for distillation portion of the loss.
- `learning_rate`    : Learning rate.
- `save_model`       : Whether to save the quantized model.
- `dtype`            : Datatype to run the method in, choices=['float16','bfloat16','float32'].
- `log_interval`     : Logging interval for training.
- `out_interval`     : Interval for saving output.
- `grad_ckpt`        : Whether to use gradient checkpointing.
- `early_save_mode`  : Diagnostic option to save an alternative version of the weights, choices=['greedy','y','orig']. 'greedy' produces greedily rounded weight, 'y' produces the original model in our new parameterization, and `orig` produces the original model.
- `wandb`            : Whether to use wandb logging.
- `use_train_ckpt`   : If true, checks if quantized model already output. If so, does not run method.
- `use_eval_ckpt`    : If true, checks if eval already output. If os, does not run eval.
- `quip`             : Whether to use rotations.
- `half_gsm_data`    : swaps half of dataset with gsm
- `optimize_quant_scales`: optimization the quantization scales


#### GPTQ
- `model`: Huggingface model path.
- `dataset`: Quantization dataset, choices=['wikitext2','wikitext2_partition','red','red_concat','gsm','gsm_concat']. `red_concat` is default.
- `seed`: Random seed.
- `nsamples`
- `wbits`            : Number of bits to quantize to.
- `groupsize`        : Every <> parameters shares the same scale factor, default=-1 meaning that each row receives a unique scale factor.
- `true_sequential`: hueristic from GPTQ
- `actorder`: heuristics from GPTQ
- `inpus_cpu`: whether to store intermediate outputs in CPU memory
- `quip`: whether to use incoherence processing


---
### Extending to Other Models
See the top of `discq.py:train()`. 
For a new model, need to specify a `quantlist=[]` containing layer names to be quantized.
Also specify a unique `model_str` at the top of `utils.py:name_session()` for logging.
