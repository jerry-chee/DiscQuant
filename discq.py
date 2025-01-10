import transformers
import torch
import os
import json
import argparse
import datasets
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from contextlib import nullcontext
import lm_eval
import wandb

from utils import *
from quantutils import *
from linearutils import *
from gptq.datautils import *
from quiputils import RHT_inv

from torch.utils.data import DataLoader, DistributedSampler

torch.autograd.set_detect_anomaly(True)


@torch.compile(dynamic=True)
def KLdivergence(p, q):
    #p,q are a batch of probability distributions
    return (torch.sum(p * (torch.log(p+1e-10) - torch.log(q+1e-10))) / p.shape[0])

def avg_delta2(model):
    delta2 = 0.0
    n = 0
    for name, m in model.named_modules():
        if isinstance(m, quantize_linearlayer_multimode):
            delta2 += m.delta2.float().item()
            n += 1
    print(f'avg_delta2: {delta2/n} = {delta2} / {n}')
    return delta2 / n

def compute_losses(model, token_window, args, quantlist, mode='x'):
    '''
    Computes either ground truth (GT) or distillation losses KL, Intermed (I).
    The set_mode() function changes which weights are used: original, greedy, or our parameterization.
    '''
    engine = model

    GT_loss = torch.tensor(0.0, device=model.device, dtype=model.dtype)
    KL_loss = torch.tensor(0.0, device=model.device, dtype=model.dtype)
    Iloss = torch.tensor(0.0, device=model.device, dtype=model.dtype)
    outputs = None

    # token_window = token_window.reshape(1, -1)
    assert len(token_window.shape) == 2
    if token_window.shape[0] != 1:
        assert token_window.shape[-1] == args.window_size
    sample = transformers.tokenization_utils_base.BatchEncoding(
        {'input_ids':token_window,'attention_mask':torch.ones_like(token_window,dtype=torch.int8)})
    sample = sample.to(model.device)

    if args.wI > 0.0:
        activations_original = []
        hooks_original = []
        def get_activation_original(id):
            def hook(model, input, output):
                activations_original.append(output[0]) #.detach()
            return hook
        activations_x = []
        hooks_x = []
        def get_activation_x(id):
            def hook(model, input, output):
                activations_x.append(output[0]) #.detach()
            return hook

        if args.intermediate == 'layer':
            set_mode(model, 'original')
            for id, layer in enumerate(model.model.layers):
                hooks_original.append(layer.register_forward_hook(get_activation_original(id)))
            engine(**sample, labels=sample.input_ids)
            for hook in hooks_original:
                hook.remove()

            set_mode(model, mode)
            for id, layer in enumerate(model.model.layers):
                hooks_x.append(layer.register_forward_hook(get_activation_x(id)))
            outputs = engine(**sample, labels=sample.input_ids)
            for hook in hooks_x:
                hook.remove()
        
        elif args.intermediate == 'linear':
            set_mode(model, 'original')
            for id, layer in enumerate(model.model.layers):
                for name in quantlist:
                    hooks_original.append(mygetattr(layer, name).register_forward_hook(get_activation_original(id)))
            outputs = engine(**sample, labels=sample.input_ids)
            for hook in hooks_original:
                hook.remove()

            set_mode(model, mode)
            for id, layer in enumerate(model.model.layers):
                for name in quantlist:
                    hooks_x.append(mygetattr(layer, name).register_forward_hook(get_activation_x(id)))
            outputs = engine(**sample, labels=sample.input_ids)
            for hook in hooks_x:
                hook.remove()

        for a_o, a_x in zip(activations_original, activations_x):
            Iloss += (a_o - a_x).square().sum() / a_o.square().sum()
    
    set_mode(model, mode)

    if args.wGT > 0.0:
        if outputs is None:
            outputs = engine(**sample, labels=sample.input_ids)
        GT_loss = outputs.loss
    
    if args.wKL > 0.0:
        if outputs is None:
            outputs = engine(**sample, labels=sample.input_ids)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=-1)

        set_mode(model, 'original')
        with torch.no_grad():
            target_outputs = engine(**sample, labels=sample.input_ids)
            target_logits = target_outputs.logits
            target_probs = torch.nn.functional.softmax(target_logits, dim=-1)
        set_mode(model, mode)

        KL_loss = KLdivergence(
            target_probs.view(-1, target_probs.size(-1)),
            probs.view(-1, probs.size(-1)))

    if mode!='x':
        set_mode(model, 'x')

    return GT_loss, KL_loss, Iloss

def loss_wrap(data_iter, mode, model, args, quantlist, delta2, bs=1, use_grad=False):
    '''
    Computes losses using batch size and gradient accumulation.
    The bs argument controls the batch size, and then len(data_iter)//bs is the grad accumulation.
    '''
    GTloss = 0
    KLloss = 0
    Iloss  = 0

    inner_model = model

    assert len(data_iter) % bs == 0
    N = len(data_iter) // bs
    N_I = 1
    delta2_scale = delta2
    if args.wI > 0:
        if args.intermediate == 'layer':
            N_I = N * len(inner_model.model.layers)
        elif args.intermediate == 'linear':
            N_I = N * len(inner_model.model.layers) * len(quantlist)
        else:
            raise NotImplementedError
    for i in range(N):
        token_window = torch.vstack(data_iter[i * bs:(i+1) * bs])
        gt, kl, interm = compute_losses(model, token_window, args, quantlist, mode=mode)
        if use_grad:
            loss = (args.wKL * kl)/(N * delta2_scale) + \
                (args.wI * interm)/(N_I * delta2_scale) + \
                (args.wGT * gt)/(N)
            loss.backward()
        GTloss += gt.item() / N
        KLloss += kl.item() / N
        if args.wI > 0:
            Iloss  += interm.item() / N_I

    Loss = (args.wKL * KLloss)/(delta2_scale) + (args.wI * Iloss)/(delta2_scale) \
        + (args.wGT * GTloss)
    return Loss, GTloss, KLloss, Iloss

def plot_wrap(loss_greedy, loss_orig, results_keys, results_dict, args, savename, savename2):
    '''
    Plots train and validation loss against original and greedy rounding.
    '''
    plt.axhline(y = loss_greedy, label=f'greedy val={loss_greedy:.3f}', color='blue', ls='--')
    plt.axhline(y = loss_orig, label=f'original val={loss_orig:.3f}', color='gray', ls='--')
    for key in results_keys:
        plt.plot(*zip(*results_dict[key]), label=key)
    plt.legend()
    plt.grid()
    gap=(loss_greedy - loss_orig)
    plt.ylim(loss_orig-gap*2, loss_orig+gap*4)
    plt.savefig(os.path.join(args.output, savename))
    if args.wandb: wandb.log({savename2: plt.gcf()})
    plt.close()

def save_model(args, dtype, model, tokenizer, savename, mode='rdx'):
    '''
    Saves model. The mode argument is used to determine which is saved: original, our rounded weights, etc.
    '''
    model_to_save = transformers.AutoModelForCausalLM.from_pretrained(
        args.model_id, trust_remote_code=True, torch_dtype=dtype)

    for name, m in model.named_modules():
        if isinstance(m, quantize_linearlayer_multimode):
            if mode =='rdx':
                value = m._unquant(m.x, rd=True)
            elif mode == 'y':
                value = m._unquant(m._gen_y(), rd=False)
            elif mode =='greedy':
                value = m._unquant(m._gen_y(), rd=True)
            elif mode == 'original':
                value = m.linear.weight
            else:
                raise NotImplementedError
            if (mode == 'rdx') or (mode == 'greedy'):
                m.grid._test_compression( value )
            if args.quip:
                value = RHT_inv(value, m.S_in, m.S_out).contiguous()
            mysetattr(model_to_save, 
                        name+'.weight.data', value)
    model_to_save.eval()
    model_to_save.save_pretrained(savename)
    tokenizer.save_pretrained(savename)
    del model_to_save

def train(args, devices):
    ## For each new model, need to define a quantlist which specifies the layers to be quantized.
    model_id = args.model_id
    if model_id == 'microsoft/phi-2':
        quantlist=['self_attn.q_proj','self_attn.k_proj','self_attn.v_proj','self_attn.dense', 'mlp.fc1','mlp.fc2']
    elif model_id == 'microsoft/Phi-3-mini-4k-instruct':
        quantlist = ['self_attn.o_proj','self_attn.qkv_proj','mlp.gate_up_proj','mlp.down_proj']
    elif (model_id == 'meta-llama/Meta-Llama-3.1-8B') or (model_id == 'meta-llama/Meta-Llama-3.1-8B-Instruct') \
        or (model_id == 'meta-llama/Meta-Llama-3.1-70B') or (model_id == 'meta-llama/Meta-Llama-3.1-70B-Instruct'):
        quantlist=['self_attn.q_proj','self_attn.k_proj','self_attn.v_proj','self_attn.o_proj','mlp.gate_proj','mlp.up_proj','mlp.down_proj']
    elif (model_id == 'meta-llama/Llama-2-7b-hf'):
        quantlist=['self_attn.q_proj','self_attn.k_proj','self_attn.v_proj','self_attn.o_proj','mlp.gate_proj','mlp.up_proj','mlp.down_proj']
    else:
        raise ValueError(f'Need to specify quantlist for {model_id}')
    if args.dtype == 'bfloat16':
        dtype = torch.bfloat16
    elif args.dtype == 'float32':
        dtype = torch.float32
    elif args.dtype == 'float16':
        dtype = torch.float16
    else:
        raise NotImplementedError
    
    ## Load model. Flash Attention is optional
    device_map = args.device_map
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=dtype, device_map=device_map, trust_remote_code=True, 
        attn_implementation='flash_attention_2')
    disable_dropout_in_model(model)
    ## Wraps all linear layers specified in quantlist with our quantized linear class
    quantize_model(model, quantlist, args)
    tokenizer=transformers.AutoTokenizer.from_pretrained(model_id,padding_side='left')
    tokenizer.pad_token = tokenizer.eos_token

    ## Can also save other modes. Training is then not done.
    if args.early_save_mode == 'greedy':
        save_model(args, dtype, model, tokenizer, args.early_savedir, 'greedy')
        with open(f'{args.early_savedir}/quant_args.json', 'w') as f:
            json.dump(args.__dict__, f, indent=2)
        return
    elif args.early_save_mode == 'y':
        save_model(args, dtype, model, tokenizer, args.early_savedir, 'y')
        with open(f'{args.early_savedir}/quant_args.json', 'w') as f:
            json.dump(args.__dict__, f, indent=2)
        return
    elif args.early_save_mode == 'orig':
        save_model(args, dtype, model, tokenizer, args.early_savedir, 'original')
        with open(f'{args.early_savedir}/quant_args.json', 'w') as f:
            json.dump(args.__dict__, f, indent=2)
        return

    ## training dataset is a list of tensors, each tensor a tokenized sample.
    window_size           = args.window_size
    number_of_samples     = args.number_of_iterations
    number_of_samples_val = args.number_of_samples_val
    token_windows, token_windows_val = get_loaders(
        args.dataset, number_of_samples, args.seed, args.window_size, model_id)
    token_windows_val = token_windows_val[:number_of_samples_val]
    # swaps half of dataset with gsm
    if args.half_gsm_data:
        token_windows, token_windows_val = swap_half_gsm(
            token_windows, token_windows_val,
            number_of_samples, number_of_samples_val, 
            model_id, args)

    display_memory(devices)
    tot_batch_size       = args.batch_size * args.grad_accum
    number_of_epochs = args.number_of_epochs
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    num_iterations_per_epoch = number_of_samples // tot_batch_size // world_size
    num_iterations   = number_of_epochs * num_iterations_per_epoch
    num_warmup       = args.number_of_warmup
    num_params=np.sum([p.numel() for p in model.parameters() if p.requires_grad])
    w_KL = args.wKL
    w_GT = args.wGT
    w_I  = args.wI
    delta2 = avg_delta2(model)
    log_interval = args.log_interval
    out_interval = args.out_interval

    ## unique experiment name 
    session = name_session(args)
    print(f'session={session}')
    os.makedirs(os.path.join(args.output, f'logs/{session}'), exist_ok=True)
    os.makedirs(os.path.join(args.output, f'checkpoints/{session}'), exist_ok=True)
    with open(f'{args.output}/logs/{session}/args.json', 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    if args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate)
    elif args.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999))
    if args.lr_sched == 'linear':
        scheduler = transformers.optimization.get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=num_warmup, num_training_steps=num_iterations)
    elif args.lr_sched == 'cosine':
        scheduler = transformers.optimization.get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=num_warmup, num_training_steps=num_iterations)
    best_loss_val, best_xrd_val = np.inf, np.inf

    def get_data_loader(dataset, batch_size, shuffle, args):
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        return dataloader

    train_loader = get_data_loader(token_windows, batch_size=tot_batch_size, shuffle=True, args=args)

    ## gradient checkpointing reduces gpu memory usage.
    if args.grad_ckpt:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={'use_reentrant': False})

    ## evaluating basline models: original and greedy rounding.
    with torch.no_grad():
        loss_orig, GTloss_orig, KLloss_orig, Iloss_orig = loss_wrap(
            token_windows_val, 'original', model, args, quantlist, delta2)
        loss_greedy, GTloss_greedy, KLloss_greedy, Iloss_greedy = loss_wrap(
            token_windows_val, 'greedy', model, args, quantlist, delta2)
        print(f'original loss: {loss_orig}, greedy loss: {loss_greedy}')

    results_dict={
        'KLloss_train':[], 'GTloss_train':[], 'Iloss_train':[],
        'KLloss_val':[], 'GTloss_val':[], 'Iloss_val':[],
        'KLloss_xrd_val':[],'GTloss_xrd_val':[], 'Iloss_xrd_val':[],
        'loss_train':[],'loss_val':[], 'loss_xrd_val':[],
        'lr':[],'num_rounded':[], 'best_loss_val':[], 'best_xrd_val':[],
        'done':False}
    results_dict['loss_orig']   = loss_orig
    results_dict['KLloss_orig'] = KLloss_orig
    results_dict['GTloss_orig'] = GTloss_orig
    results_dict['Iloss_orig']  = Iloss_orig
    results_dict['loss_greedy']   = loss_greedy
    results_dict['KLloss_greedy'] = KLloss_greedy
    results_dict['GTloss_greedy'] = GTloss_greedy
    results_dict['Iloss_greedy']  = Iloss_greedy

    for epoch in range(number_of_epochs):
        print(f'Epoch: {epoch}')

        pbar = tqdm(total=num_iterations_per_epoch)
        for i, batch in enumerate(tqdm(train_loader)):
            iteration = epoch * num_iterations_per_epoch + i
            if iteration==0:
                display_memory(devices)

            optimizer.zero_grad()
            loss_train, GTloss_train, KLloss_train, Iloss_train = loss_wrap(
                torch.chunk(batch, args.batch_size), 'x', model, args, quantlist, delta2, bs=args.batch_size, use_grad=True)

            ## Modify gradient for linear loss term. also rescale gradient, clip also rescale gradient, clip.
            for _, m in model.named_modules():
                if isinstance(m, quantize_linearlayer_multimode):
                    c_grad = m._c_grad()                   
                    m.x.grad = -c_grad + (m.x.grad * args.rho_factor).clamp(-args.clamp_value, args.clamp_value)

            optimizer.step()
            scheduler.step()

            for _, m in model.named_modules():
                if isinstance(m, quantize_linearlayer_multimode):
                    m._projection_step()

            ## Logging and eval on validation
            with torch.no_grad():
                pbar.update(1)
                if w_KL> 0: results_dict['KLloss_train'].append( (iteration, KLloss_train) )
                if w_GT> 0: results_dict['GTloss_train'].append( (iteration, GTloss_train) )
                if w_I > 0: results_dict['Iloss_train'].append( (iteration, Iloss_train) )
                results_dict['loss_train'].append( (iteration, loss_train) )
                results_dict['lr'].append((iteration,scheduler.get_last_lr()[0]))
                if args.wandb:
                    wandb.log({
                        'iteration': iteration, 'loss_train': loss_train, 
                        'lr': scheduler.get_last_lr()[0]})
                    if w_KL > 0: wandb.log({'iteration': iteration, 'KLloss_train': KLloss_train})
                    if w_GT > 0: wandb.log({'iteration': iteration, 'GTloss_train': GTloss_train})
                    if w_I > 0: wandb.log({'iteration': iteration, 'Iloss_train': Iloss_train})

                if iteration==0 or (iteration+1) % log_interval == 0:
                    pbar.set_description(
                        f'Train. GT={GTloss_train:.2f}, KL={KLloss_train:.2f}, I={Iloss_train:.2f}' )

                    loss_val, GTloss_val, KLloss_val, Iloss_val = loss_wrap(
                        token_windows_val, 'x', model, args, quantlist, delta2)
                    loss_xrd_val, GTloss_xrd_val, KLloss_xrd_val, Iloss_xrd_val = loss_wrap(
                        token_windows_val, 'rdx', model, args, quantlist, delta2)

                    if w_KL> 0: results_dict['KLloss_val'].append( (iteration, KLloss_val) )
                    if w_GT> 0: results_dict['GTloss_val'].append( (iteration, GTloss_val) )
                    if w_I > 0: results_dict['Iloss_val'].append( (iteration, Iloss_val) )
                    if w_KL> 0: results_dict['KLloss_xrd_val'].append( (iteration, KLloss_xrd_val) )
                    if w_GT> 0: results_dict['GTloss_xrd_val'].append( (iteration, GTloss_xrd_val) )
                    if w_I > 0: results_dict['Iloss_xrd_val'].append( (iteration, Iloss_xrd_val) )
                    results_dict['loss_val'].append( (iteration, loss_val) )
                    results_dict['loss_xrd_val'].append( (iteration, loss_xrd_val) )

                    num_rounded=0
                    for _, m in model.named_modules():
                        if isinstance(m, quantize_linearlayer_multimode):
                            num_rounded += m._num_rounded()
                    results_dict['num_rounded'].append( (iteration, num_rounded) )

                    if args.wandb:
                        wandb.log({'iteration': iteration, 'loss_val': loss_val, 
                                    'loss_xrd_val': loss_xrd_val, 'num_rounded': num_rounded})
                        if w_KL > 0: wandb.log({'iteration': iteration,
                            'KLloss_val': KLloss_val, 'KLloss_xrd_val': KLloss_xrd_val})
                        if w_GT > 0: wandb.log({'iteration': iteration,
                            'GTloss_val': GTloss_val, 'GTloss_xrd_val': GTloss_xrd_val})
                        if w_I > 0: wandb.log({'iteration': iteration,
                            'Iloss_val': Iloss_val, 'Iloss_xrd_val': Iloss_xrd_val})

                if (iteration+1) % out_interval == 0:
                    plot_wrap(
                        loss_greedy, loss_orig, ['loss_train','loss_val','loss_xrd_val'], 
                        results_dict, args, f'logs/{session}/loss.png', 'loss')
                    if w_GT> 0: plot_wrap(
                        GTloss_greedy, GTloss_orig, ['GTloss_train','GTloss_val','GTloss_xrd_val'], 
                        results_dict, args, f'logs/{session}/GTloss.png', 'GTloss')
                    if w_KL> 0: plot_wrap(
                        KLloss_greedy, KLloss_orig, ['KLloss_train','KLloss_val','KLloss_xrd_val'], 
                        results_dict, args, f'logs/{session}/KLloss.png', 'KLloss')
                    if w_I > 0: plot_wrap(
                        Iloss_greedy, Iloss_orig, ['Iloss_train','Iloss_val','Iloss_xrd_val'], 
                        results_dict, args, f'logs/{session}/Iloss.png', 'Iloss')

                    plt.plot(*zip(*results_dict['num_rounded']),label='num_rounded')
                    plt.axhline(y=num_params,label='num_params',color='gray',ls='--')
                    plt.grid()
                    plt.savefig(os.path.join(args.output, f'logs/{session}/rounded.png'))
                    if args.wandb:
                        wandb.log({'rounded': plt.gcf()})
                    plt.close()

                    if args.val_save and (loss_xrd_val < best_xrd_val):
                        save_name = os.path.join(
                            args.output, f'checkpoints/{session}/quantized_model')
                        save_model(args, dtype, model, tokenizer, save_name)
                    if loss_val < best_loss_val:
                        print(f"best_loss_val: {best_loss_val} -> {loss_val}")
                        best_loss_val = loss_val
                    if loss_xrd_val < best_xrd_val:
                        print(f"best_loss_xrd_val: {best_xrd_val} -> {loss_xrd_val}")
                        best_xrd_val = loss_xrd_val
                    results_dict['best_loss_val'].append( (iteration, best_loss_val))
                    results_dict['best_xrd_val'].append( (iteration, best_xrd_val))
                    with open(os.path.join(args.output, f'logs/{session}/results.json'), 'w') as f:
                        json.dump(results_dict, f, indent=2)
                    
                    if args.wandb:
                        wandb.log({'iteration': iteration, 'best_loss_val': best_loss_val, 
                                    'best_xrd_val': best_xrd_val})

        pbar.close()

    ## Save model.
    if args.save_model:
        save_name = os.path.join(args.output, f'checkpoints/{session}/quantized_model')
        save_model(args, dtype, model, tokenizer, save_name)
    results_dict['done'] = True 

    with open(os.path.join(args.output, f'logs/{session}/results.json'), 'w') as f:
        json.dump(results_dict, f, indent=2)

def main():
    args = parse_args()

    devices= ['cuda:0']
    print(args)
    display_memory(devices)
    reset_seeds(args.seed)

    session = name_session(args)

    ## Checks if there is already a completed run in the save directory. 
    ## If true, and if use_train_ckpt==False, then we don't run the code.
    pgdfKL_done = False
    results_fpath = os.path.join(args.output, f'logs/{session}/results.json')
    if os.path.isfile(results_fpath): 
        with open(results_fpath, 'r') as f:
            results_dict = json.load(f)
        pgdfKL_done = results_dict['done']
    if (pgdfKL_done is False) or (args.use_train_ckpt is False):
        if args.wandb:
            wandb.init(project=f"DiscQuant", group=f"{args.output}", config=args, save_code=True)
        train(args, devices)
    else:
        print(f'skipping pgdf_KL training, ckpt found and is done')


if __name__ == '__main__':
    main()