import os
import math
import argparse
import transformers
import datasets
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

# devices=['cuda:0','cuda:1','cuda:2','cuda:3']
devices=['cuda:0']
torch.set_float32_matmul_precision('high')

parser = argparse.ArgumentParser(description='gradient rank')
parser.add_argument('--model_id', type=str, default='microsoft/phi-2')
parser.add_argument('--dataset', type=str, default='wikitext')
parser.add_argument('--layer_ids', nargs='+', default=[0,1,2,10,20,31])
parser.add_argument('--num_samples', type=int, default=1024)
parser.add_argument('--out_dim', type=int, default=2048)
parser.add_argument('--seed', type=int, default=12345)
parser.add_argument('--jl_stride', type=int, default=2)
parser.add_argument('--sampl_stride', type=int, default=2)
parser.add_argument('--savedir', type=str)
parser.add_argument('--mode', type=str, choices=['grad', 'plot_only'])
parser.add_argument('--lwd', type=float, default=1.0, help='plot linewidth')
args = parser.parse_args()

def count_params(model):
    cnt = 0
    for _, module in model.model.layers[0].named_modules():
        if isinstance(module, torch.nn.Linear):
            cnt += module.weight.numel()
    return cnt

@torch.no_grad()
@torch.compile
def online_JL(x, stride=2):
    assert x.dtype == torch.float32
    assert args.out_dim % stride == 0
    n = x.shape[1]
    m = x.shape[0]
    torch.manual_seed(args.seed)
    y = torch.zeros(m, args.out_dim, dtype=x.dtype, device=x.device)
    for k in range( int(args.out_dim / stride) ):
        z = torch.normal(0, 1, size=(n, stride), device=x.device) / math.sqrt(args.out_dim)
        i = stride * k
        j = stride * (k+1)
        y[:,i:j] = x @ z
    return y 

def main():
    dtype = torch.bfloat16
    model = transformers.AutoModelForCausalLM.from_pretrained(
        args.model_id, torch_dtype=dtype, device_map=devices[0],
        trust_remote_code=True, attn_implementation='flash_attention_2')
    # model = torch.compile(model)
    tokenizer=transformers.AutoTokenizer.from_pretrained(args.model_id, padding_side='left')
    tokenizer.pad_token = tokenizer.eos_token
    if args.dataset == 'wikitext':
        dataset = datasets.load_dataset('wikitext', 'wikitext-2-v1', split='train')
        dataset = dataset.filter(lambda x: len(x['text']) >= 1)
    elif args.dataset == 'red':
        dataset = datasets.load_dataset('togethercomputer/RedPajama-Data-1T-Sample', split='train')
        dataset = dataset.filter(lambda x: len(x['text']) >= 1)
        dataset = dataset.filter(lambda x: len(x['text']) <= 4096)
    dataset = dataset.shuffle(args.seed)
    dataset_iter = iter(dataset)

    os.makedirs(f'{args.savedir}', exist_ok=True)
    if os.path.isfile(f'{args.savedir}/ckpt.pt'):
        ckpt = torch.load(f'{args.savedir}/ckpt.pt')
        start = ckpt['i'] + 1
        grads = torch.load(f'{args.savedir}/grads_dict.pt')
        print(f"found checkpoint, starting at iteration {start}")
        for _ in range(start):
            next(dataset_iter)
    else:
        grads = {layer_id: [] for layer_id in args.layer_ids}
        start = 0
    grad_tmp  = {layer_id: [] for layer_id in args.layer_ids}

    assert args.num_samples % args.sampl_stride == 0
    for i in tqdm( range(start, args.num_samples), initial=start):
        sample = next(dataset_iter)
        model.zero_grad()
        encodings = tokenizer.encode(
            sample['text'], return_tensors='pt').to('cuda', non_blocking=True)
        outputs = model(encodings, labels=encodings)
        outputs.loss.backward()

        for layer_id in args.layer_ids:
            tmp = []
            for _, module in model.model.layers[layer_id].named_modules():
                if isinstance(module, torch.nn.Linear):
                    tmp.append( module.weight.grad.detach().flatten() )
            grad_tmp[layer_id].append(
                torch.cat(tmp).unsqueeze(0).to(torch.float32)
            )
            if (i+1) % args.sampl_stride == 0:
                tmp = torch.cat(grad_tmp[layer_id])
                grad_tmp[layer_id] = None
                grad_tmp[layer_id] = []
                tmp = online_JL(tmp, args.jl_stride).to('cpu', non_blocking=True)
                grads[layer_id].append(tmp)

                if layer_id == args.layer_ids[-1]:
                    torch.save({'i': i}, f'{args.savedir}/ckpt.pt')
                    torch.save(grads, f'{args.savedir}/grads_dict.pt')

    for layer_id in args.layer_ids:
        grads[layer_id] = torch.cat(grads[layer_id]).cuda()
        S = torch.linalg.svdvals(grads[layer_id]).cpu()
        S /= (grads[layer_id].shape[0] ** 0.5)
        plt.plot(S.numpy(), label=f'layer {layer_id}', linewidth=args.lwd)
        grads[layer_id] = grads[layer_id].to('cpu', non_blocking=True)

    torch.save(grads, f'{args.savedir}/grads_dict.pt')

    plt.yscale('log')
    plt.ylim(0.0001,10)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{args.savedir}/grad.pdf')


def plot_only():
    os.makedirs(f'{args.savedir}', exist_ok=True)
    assert os.path.isfile(f'{args.savedir}/ckpt.pt')
    ckpt = torch.load(f'{args.savedir}/ckpt.pt')
    start = ckpt['i'] + 1
    grads = torch.load(f'{args.savedir}/grads_dict.pt')
    print(f"found checkpoint, starting at iteration {start}")
    for key in grads:
        S = torch.linalg.svdvals(grads[key].cuda()).cpu()
        S /= (grads[key].shape[0] ** 0.5)
        plt.plot(S.numpy(), label=f'layer {key}', linewidth=args.lwd)
    plt.yscale('log')
    plt.ylim(1e-1,1e3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{args.savedir}/grad2.pdf')



if __name__ == "__main__":
    if args.mode == 'grad':
        main()
    elif args.mode == 'plot_only':
        plot_only()