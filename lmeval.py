import os
import json
import queue
import lm_eval
import multiprocessing as mp

from utils import *

device_list=['cuda:0']
def lmeval_single(load_name, output_name, task, num_fewshot, batch_size, use_wandb):
    try:
        results = lm_eval.simple_evaluate(
            model='hf', model_args=f'pretrained={load_name},trust_remote_code=True,dtype=float16', #,parallelize=True
            tasks=task, num_fewshot=num_fewshot, batch_size=batch_size, 
            write_out=True, log_samples=False)
        with open(f'{output_name}/{task}.json', 'w') as f:
            json.dump(results, f, indent=2)
    except Exception as e:
        print(f"lmeval {task} didn't work:")
        print(e)
        pass

def main():
    args = parse_args()
    print(args)
    display_memory(device_list)
    reset_seeds(args.seed)

    session = name_session(args)

    # evaluate via lm-eval
    load_name = os.path.join(args.output, f'checkpoints/{session}/quantized_model')
    output_name = os.path.join(args.output, f'logs/{session}')
    batch_size = 32
    mmlu_bs = 8

    if (os.path.isfile(f'{output_name}/gsm8k_cot.json') is False) or (args.use_eval_ckpt is False):
        lmeval_single(load_name, output_name, 'gsm8k_cot', 8, batch_size, args.wandb)

    if (os.path.isfile(f'{output_name}/wikitext.json') is False) or (args.use_eval_ckpt is False):
        lmeval_single(load_name, output_name, 'wikitext', 0, batch_size, args.wandb)

    if (os.path.isfile(f'{output_name}/piqa.json') is False) or (args.use_eval_ckpt is False):
        lmeval_single(load_name, output_name, 'piqa', 0, batch_size, args.wandb)

    if (os.path.isfile(f'{output_name}/arc_challenge.json') is False) or (args.use_eval_ckpt is False):
        lmeval_single(load_name, output_name, 'arc_challenge', 0, batch_size, args.wandb)

    if (os.path.isfile(f'{output_name}/mmlu.json') is False) or (args.use_eval_ckpt is False):
        lmeval_single(load_name, output_name, 'mmlu', 5, mmlu_bs, args.wandb)

    if (os.path.isfile(f'{output_name}/hellaswag.json') is False) or (args.use_eval_ckpt is False):
        lmeval_single(load_name, output_name, 'hellaswag', 0, batch_size, args.wandb)

    if (os.path.isfile(f'{output_name}/winogrande.json') is False) or (args.use_eval_ckpt is False):
        lmeval_single(load_name, output_name, 'winogrande', 0, batch_size, args.wandb)


if __name__ == '__main__':
    main()