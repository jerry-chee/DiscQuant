import time
from transformers import AutoModelForCausalLM
import torch
import torch.nn as nn

from gptq import *
from quant import *
# import marlin
from datautils import *


DEV = torch.device('cuda:0')

def find_layers(module, layers=[nn.Linear], name=''):
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res

def get_phi(name):
    import torch
    def skip(*args, **kwargs):
        pass
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    # from transformers import PhiForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        name, torch_dtype='auto', trust_remote_code=True, attn_implementation='flash_attention_2')
    model.seqlen = 2048
    return model

@torch.no_grad()
def phi_sequential(model, dataloader, dev, inps_dev):
    print('Starting ...')

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    model.model.norm = model.model.norm.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = []
    attention_masks = []
    position_ids = []

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps.append(inp.to(inps_dev, non_blocking=True))
            attention_masks.append(kwargs['attention_mask'])
            position_ids.append(kwargs['position_ids'])
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch.unsqueeze(0).to(dev))
            # model(batch.to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    model.model.norm = model.model.norm.cpu()
    torch.cuda.empty_cache()

    print('Ready.')

    quantizers = {}
    for i in range(len(layers)):
        layer = layers[i].to(dev)
        full = find_layers(layer)

        if args.true_sequential:
            # Jerry 2024-05-23: best guess at ordering based on llama
            sequential = [
                ['self_attn.qkv_proj'],
                ['self_attn.o_proj'],
                ['mlp.gate_up_proj'],
                ['mlp.down_proj']
            ]
        else:
            sequential = [list(full.keys())]

        for names in sequential:
            # Jerry: don't implement skip_gq
            # if model.config.num_attention_heads != model.config.num_key_value_heads and args.skip_gq:
            #     names.remove('self_attn.k_proj')
            #     names.remove('self_attn.v_proj')

            subset = {n: full[n] for n in names}

            gptq = {}
            for name in subset:
                gptq[name] = GPTQ(subset[name])
                gptq[name].quantizer = Quantizer()
                gptq[name].quantizer.configure(args.wbits)

            def add_batch(name):
                def tmp(_, inp, out):
                    gptq[name].add_batch(inp[0].data, out.data)
                return tmp
            handles = []
            for name in subset:
                handles.append(subset[name].register_forward_hook(add_batch(name)))
            for j in range(args.nsamples):
                layer(inps[j].to(dev, non_blocking=True), attention_mask=attention_masks[j], position_ids=position_ids[j])
            for h in handles:
                h.remove()

            for name in subset:
                print(i, name)
                print('Quantizing ...')
                res = gptq[name].fasterquant(
                    # percdamp=args.percdamp, groupsize=args.groupsize, clip=not args.no_clip, baseline=args.nearest
                    percdamp=args.percdamp, groupsize=args.groupsize, clip=False,
                    quip=args.quip, baseline=args.nearest, actorder=args.actorder
                )
                # res = list(res)
                # res[0] = res[0].cpu()
                # res[1] = res[1].cpu()
                # quantizers['model.layers.%d.%s' % (i, name)] = res

        for j in range(args.nsamples):
            inps[j] = layer(inps[j].to(dev, non_blocking=True), attention_mask=attention_masks[j], position_ids=position_ids[j])[0]
            inps[j] = inps[j].to(inps_dev, non_blocking=True)

        layers[i] = layer.cpu()
        del layer
        del gptq 
        torch.cuda.empty_cache()

    model.config.use_cache = use_cache
    return quantizers

@torch.no_grad()
def phi_eval(model, dataloader, dev):
    print('Evaluating ...')

    nsamples = len(dataloader) 

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = []
    attention_masks = []
    position_ids = []

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps.append(inp)
            attention_masks.append(kwargs['attention_mask'])
            position_ids.append(kwargs['position_ids'])
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch.to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    torch.cuda.empty_cache()

    for i in range(len(layers)):
        print(i)
        layer = layers[i].to(dev)
        for j in range(nsamples):
            inps[j] = layer(inps[j], attention_mask=attention_masks[j], position_ids=position_ids[j])[0]
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()

    if model.model.norm is not None:
        model.model.norm = model.model.norm.to(dev)
    model.lm_head = model.lm_head.to(dev)

    nlls = []
    for i in range(nsamples):
        hidden_states = inps[i]
        if model.model.norm is not None:
            hidden_states = model.model.norm(hidden_states)
        lm_logits = model.lm_head(hidden_states)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = (dataloader[i].to(dev))[:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        neg_log_likelihood = loss.float() * model.seqlen
        nlls.append(neg_log_likelihood)
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    print(ppl.item())

    model.config.use_cache = use_cache


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        'model', type=str,
        help='Phi model to load; pass location of hugginface converted checkpoint.'
    )
    parser.add_argument(
        '--dataset', type=str, default='red', choices=['red','red_concat','wikitext2','wikitext2_partition'],
        help='Where to extract calibration data from.'
    )
    parser.add_argument(
        '--seed',
        type=int, default=0, help='Seed for sampling the calibration data.'
    )
    parser.add_argument(
        '--nsamples', type=int, default=256,
        help='Number of calibration data samples.'
    )
    parser.add_argument(
        '--percdamp', type=float, default=.1,
        help='Percent of the average Hessian diagonal to use for dampening.'
    )
    parser.add_argument(
        '--nearest', action='store_true',
        help='Whether to run the RTN baseline.'
    ) 
    parser.add_argument(
        '--wbits', type=int, default=16, choices=[2, 3, 4, 16],
        help='#bits to use for quantization; use 16 for evaluating base model.'
    )
    parser.add_argument(
        '--groupsize', type=int, default=128, choices=[-1, 32, 64, 128],
        help='Groupsize to use for quantization; default is 128.'
    )
    parser.add_argument(
        '--true-sequential', action='store_true',
        help='Whether to run in true sequential model.'
    )
    parser.add_argument(
        '--actorder', action='store_true',
        help='Reorder W heuristic based on H.'
    )
    parser.add_argument(
        '--save', type=str, default='',
        help='Whether and where to save the quantized model in. Takes precedence over marlin'
    )
    parser.add_argument(
        '--inps_cpu', action='store_true',
        help='Whether to store intermediate outputs in cpu.'
    )
    parser.add_argument(
        '--quip', action='store_true',
        help='Whether to use incoherence processing.'
    )

    args = parser.parse_args()
    if args.inps_cpu:
        inps_dev = 'cpu'
    else:
        inps_dev = DEV

    if args.nearest:
        args.nsamples = 0

    model = get_phi(args.model)
    model.eval()


    if args.wbits < 16:
        dataloader, testloader = get_loaders(
            args.dataset, nsamples=args.nsamples, seed=args.seed, model=args.model, seqlen=model.seqlen)
        tick = time.time()
        quantizers = phi_sequential(model, dataloader, DEV, inps_dev)
        print(time.time() - tick)

    if args.save:
        model.save_pretrained(args.save)

    # datasets = ['wikitext2', 'red'] 
    datasets = ['wikitext2', 'wikitext2_partition', 'red', 'red_concat'] 
    for dataset in datasets:
        dataloader, testloader = get_loaders(
            dataset, seed=args.seed, model=args.model, seqlen=model.seqlen
        )
        print(dataset)
        phi_eval(model, testloader, DEV)

