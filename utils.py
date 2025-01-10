import gc
import argparse
import random
import torch
import numpy as np
import transformers
import torch
import os


def parse_args():
    parser = argparse.ArgumentParser(description='quantization')
    
    parser.add_argument('--model_id', type=str, default='microsoft/phi-2')
    parser.add_argument('--device_map', type=str, default='auto', choices=['auto','balanced','balanced_low_0'], help='from_pretrained()')
    parser.add_argument('--dataset', type=str, choices=['wikitext2','wikitext2_partition','red','red_concat','gsm','gsm_concat'])
    parser.add_argument('--output', type=str, default='..', help='output folder')
    parser.add_argument('--seed', type=int, default=42, help='seed')
    parser.add_argument('--init_x', type=str, default='rand', choices=['rand','orig'])
    parser.add_argument('--cache_up', default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument('--cache_down', default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument('--wbits', type=int, default=4, help='wbits')
    parser.add_argument('--groupsize', type=int, default=-1, help='groupsize')
    parser.add_argument('--optimizer', type=str, default='AdamW', choices=['SGD','AdamW'])
    parser.add_argument('--lr_sched', type=str, default='cosine', choices=['linear','cosine'])
    parser.add_argument('--number_of_iterations', type=int, default=1024, help='number of samples')
    parser.add_argument('--number_of_samples_val', type=int, default=64, help='number of validation samples')
    parser.add_argument('--number_of_warmup', type=int, default=128, help='number of warmup samples')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('--grad_accum', type=int, default=1, help='gradient accumulation')
    parser.add_argument('--number_of_epochs', type=int, default=1, help='number of epochs')
    parser.add_argument('--bs_iter_fixed', default=True, action=argparse.BooleanOptionalAction, help='keeps iterations fixed when increase bs')
    parser.add_argument('--window_size', type=int, default=2048, help='batch size')
    parser.add_argument('--clamp_value', type=float, default=0.1, help='clamp value')
    parser.add_argument('--rho_factor', type=float, default=10, help='rho factor')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate')
    parser.add_argument('--save_model', action=argparse.BooleanOptionalAction)
    parser.add_argument('--val_save', default=False, action=argparse.BooleanOptionalAction, help='saves if xrd_val lower')
    parser.add_argument('--dtype', type=str, default='bfloat16', choices=['float16','bfloat16','float32'], help='data type: float16, float32, bfloat16')
    parser.add_argument('--wGT', type=float, default=0.0, help='weight for GT loss')
    parser.add_argument('--wKL', type=float, default=1.0, help='weight for KL loss')
    parser.add_argument('--wI', type=float, default=0.0, help='weight for intermediate representations')
    parser.add_argument('--intermediate', type=str, default='layer', choices=['None','layer','linear'])
    parser.add_argument('--log_interval', type=int, default=64)
    parser.add_argument('--out_interval', type=int, default=256)
    parser.add_argument('--grad_ckpt', default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument('--early_save_mode', default=None, type=str, choices=['greedy','y','orig'])
    parser.add_argument('--early_savedir', type=str)
    parser.add_argument('--wandb', default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument('--use_train_ckpt', default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument('--use_eval_ckpt', default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument('--quip', default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument('--half_gsm_data', default=False, action=argparse.BooleanOptionalAction, help='replaces half of dataset with GSM')

    args = parser.parse_args()

    if args.intermediate == 'None':
        args.wI = 0.0
    if args.wI == 0.0:
        args.intermediate = 'None'

    world_size = int(os.environ.get('WORLD_SIZE', 1))
    if args.bs_iter_fixed:
        print(f"keeping iters fixed: number_of_samples = number_of_iterations * batch_size * grad_accum")
        args.number_of_iterations = int( args.number_of_iterations * args.batch_size * args.grad_accum * world_size)
    
    # assert args.number_of_samples % args.log_interval == 0
    # assert args.number_of_samples % args.out_interval == 0
    assert args.log_interval % (args.batch_size * args.grad_accum) == 0
    assert args.out_interval % (args.batch_size * args.grad_accum) == 0
    if not args.bs_iter_fixed:
        print(f"re-adjusting log, out interval for bs")
        args.log_interval = args.log_interval // (args.batch_size * args.grad_accum * world_size)
        args.out_interval = args.out_interval // (args.batch_size * args.grad_accum * world_size)
    assert (args.number_of_iterations // (args.batch_size * args.grad_accum * world_size)) % args.log_interval == 0
    assert (args.number_of_iterations // (args.batch_size * args.grad_accum * world_size)) % args.out_interval == 0

    return args

def name_session(args):
    if args.model_id == 'microsoft/phi-2':
        model_str='phi2'
    elif args.model_id == 'microsoft/Phi-3-mini-4k-instruct':
        model_str='phi3mini4k'
    elif args.model_id == 'meta-llama/Meta-Llama-3.1-8B-Instruct':
        model_str='llama3.1_8b_instruct'
    elif args.model_id == 'meta-llama/Meta-Llama-3.1-8B':
        model_str='llama3.1_8b'
    elif args.model_id == 'meta-llama/Meta-Llama-3.1-70B-Instruct':
        model_str='llama3.1_70b_instruct'
    elif args.model_id == 'meta-llama/Meta-Llama-3.1-70B':
        model_str='llama3.1_70b'
    elif args.model_id == 'meta-llama/Llama-2-7b-hf':
        model_str='llama2_7b'
    else:
        raise ValueError(f'Need to specify model_str for {args.model_id}')
    
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    N_adj = args.number_of_iterations
    if args.bs_iter_fixed:
        N_adj = args.number_of_iterations // (args.batch_size * args.grad_accum * world_size)
    bs_adj = args.batch_size * args.grad_accum * world_size

    session = f'{model_str}_wKL_{args.wKL}_wGT_{args.wGT}_wI_{args.wI}_lr_{args.learning_rate}_rhof_{args.rho_factor:.1f}_clamp_{args.clamp_value}_wbits_{args.wbits}_group_{args.groupsize}_intermed_{args.intermediate}_dtype_{args.dtype}_optim_{args.optimizer}_lrsched_{args.lr_sched}_{args.dataset}_N_{N_adj}_Warm_{args.number_of_warmup}_W_{args.window_size}_E_{args.number_of_epochs}_BS_{bs_adj}_np_{world_size}_initX_{args.init_x}'
    return session

def display_memory(devices='cuda',printstr=''):
    #if device is not a list, make it a list
    if not isinstance(devices, list):
        devices = [devices]
    for device in devices:
        gc.collect()
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
        #torch.cuda.memory_cache()
        memory = torch.cuda.memory_allocated(device)
        print(device, "{:.3f} GB".format(memory / 1024 ** 3), printstr)


def reset_seeds(seed):
    # Set the seed for PyTorch
    torch.manual_seed(seed)

    # Set the seed for other libraries
    np.random.seed(seed)
    random.seed(seed)


def cast_batch_encoding(sample, dtype):
    casted_sample = {}
    casted_sample['attention_mask'] = sample['attention_mask']
    casted_sample['input_ids'] = sample['input_ids'].to(dtype)
    return transformers.tokenization_utils_base.BatchEncoding(casted_sample)

def disable_dropout_in_model(model: torch.nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0

class TokensDataset(torch.utils.data.Dataset):
    def __init__(self, tokens):
        self.tokens = tokens

    def __len__(self):
        return len(self.tokens)
    
    def __getitem__(self, idx):
        return self.tokens[idx]

def swap_half_gsm(token_windows, token_windows_val,
        number_of_samples, number_of_samples_val, model_id, args):
    from gptq.datautils import get_gsm_concat
    half_train = number_of_samples // 2
    gsm_train, gsm_val = get_gsm_concat(half_train, args.seed, args.window_size, model_id)
    token_windows[half_train:] = gsm_train
    half_val = number_of_samples_val // 2
    token_windows_val[half_val:] = gsm_val[:half_val]
    return token_windows, token_windows_val