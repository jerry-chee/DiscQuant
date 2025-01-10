import os
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer


def get_wikitext2_partition(nsamples, seed, seqlen, model):
    """original wikitext2 processing. filters out short samples, partitions"""
    traindata = load_dataset('wikitext', 'wikitext-2-v1', split='train')
    traindata = traindata.filter(lambda x: len(x['text']) >= 100)
    testdata  = load_dataset('wikitext', 'wikitext-2-v1', split='test')

    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)
    trainenc = tokenizer(' '.join(traindata['text']), return_tensors='pt', 
                         truncation=False, padding=False)['input_ids'][0]
    testenc = tokenizer(' '.join(testdata['text']), return_tensors='pt', 
                         truncation=False, padding=False)
    trainenc = trainenc[0:seqlen*int(len(trainenc)/seqlen)]
    train_dataloader = trainenc.view(-1,seqlen)

    import random
    random.seed(seed)
    random.shuffle(train_dataloader)
    trainloader = []
    for i in range(nsamples):
        trainloader.append(train_dataloader[i])
    testloader = []
    for i in range(0, testenc.input_ids.shape[1] - seqlen, seqlen):
        testloader.append(testenc.input_ids[:, i:(i + seqlen)])

    return trainloader, testloader


def get_wikitext2(nsamples, seed, seqlen, model):
    traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')

    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)
    trainenc = tokenizer("\n\n".join(traindata['text']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')

    import random
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        # inp = trainenc.input_ids[:, i:j]
        inp = trainenc.input_ids[0][i:j]
        # tar = inp.clone()
        # tar[:, :-1] = -100
        # trainloader.append((inp, tar))
        trainloader.append(inp)
    testloader = []
    for i in range(0, testenc.input_ids.shape[1] - seqlen, seqlen):
        testloader.append(testenc.input_ids[:, i:(i + seqlen)])

    return trainloader, testloader 


def get_red(nsamples, seed, seqlen, model):
    VALSAMPLES = 1024
    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)
    traindata = load_dataset('togethercomputer/RedPajama-Data-1T-Sample', split='train')

    np.random.seed(seed)
    perm = np.random.permutation(len(traindata))

    dataloader = []
    for i in perm:
        tokens = tokenizer(traindata[int(i)]['text'], return_tensors='pt').input_ids
        if not (1 < tokens.shape[1] <= seqlen):
            continue
        dataloader.append(tokens)
        if len(dataloader) == nsamples + VALSAMPLES:
            break
    trainloader = dataloader[VALSAMPLES:]
    testloader = dataloader[:VALSAMPLES]
    for i in range(len(trainloader)):
        trainloader[i] = trainloader[i][0]
    return trainloader, testloader


# def get_red_concat(nsamples, seed, seqlen, model):
#     VALSAMPLES = 1024
#     tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)
#     traindata = load_dataset('togethercomputer/RedPajama-Data-1T-Sample', split='train')
# 
#     np.random.seed(seed)
#     perm = np.random.permutation(len(traindata))
#     
#     dataloader = []
#     for k in range(len(traindata) // nsamples):
#         i = k * nsamples
#         j = (k+1) * nsamples
#         tokens = tokenizer(traindata[perm[i:j]]['text'], return_tensors='pt', 
#                            truncation=True, padding=True, max_length=seqlen)
#         lens = tokens.attention_mask.sum(dim=-1)
#         good = torch.where(lens == seqlen)[0]
#         if len(good) > 0:
#             if len(dataloader) + len(good) > nsamples + VALSAMPLES:
#                 good = good[:nsamples - len(dataloader) - VALSAMPLES]
#             for g in good:
#                 dataloader.append(tokens.input_ids[g])
#             if len(dataloader) == nsamples + VALSAMPLES:
#                 break
#     trainloader = dataloader[VALSAMPLES:]
#     testloader = dataloader[:VALSAMPLES]
#     for i in range(len(trainloader)):
#         trainloader[i] = trainloader[i][0]
#     return trainloader, testloader


def get_red_concat(nsamples, seed, seqlen, model):
    VALSAMPLES = 1024
    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)
    traindata = load_dataset('togethercomputer/RedPajama-Data-1T-Sample', split='train')

    np.random.seed(seed)
    perm = np.random.permutation(len(traindata))

    dataloader = []
    size = 0
    tmp = []
    for i in perm:
        tokens = tokenizer(traindata[int(i)]['text'], return_tensors='pt').input_ids
        if size + tokens.shape[1] > seqlen:
            # tokens_tail = tokens[:, seqlen - size :]
            tokens = tokens[:, : seqlen - size]
            tmp.append(tokens)
            dataloader.append( torch.cat(tmp, dim=1) )
            size = 0
            tmp = []
            # if tokens_tail.shape[1] > seqlen:
            #     print('t')
        else:
            tmp.append(tokens)
            size += tokens.shape[1]
        if len(dataloader) == nsamples + VALSAMPLES:
            break
    trainloader = dataloader[VALSAMPLES:]
    testloader = dataloader[:VALSAMPLES]
    for i in range(len(trainloader)):
        trainloader[i] = trainloader[i][0]
    return trainloader, testloader

def get_c4_new(nsamples, seed, seqlen, model):
    from datasets import load_dataset
    # new c4
    traindata = load_dataset(
        'allenai/c4',
        data_files={'train': 'en/c4-train.00000-of-01024.json.gz'},
        split='train')
    valdata = load_dataset(
        'allenai/c4',
        data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'},
        split='validation')
    # traindata = load_dataset(
    #     'allenai/c4', 'allenai--c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train'
    # )
    # valdata = load_dataset(
    #     'allenai/c4', 'allenai--c4', data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split='validation'
    # )

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)

    import random
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] >= seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        # tar = inp.clone()
        # tar[:, :-1] = -100
        # trainloader.append((inp[0], tar))
        trainloader.append(inp[0])

    valenc = tokenizer(' '.join(valdata[:1100]['text']), return_tensors='pt')
    # valenc = valenc.input_ids[:, :(256 * seqlen)]
    # return trainloader, valenc

    testloader = []
    for i in range(0, valenc.input_ids.shape[1] - seqlen, seqlen):
        testloader.append(valenc.input_ids[:, i:(i + seqlen)])
    return trainloader, testloader

    # class TokenizerWrapper:
    #     def __init__(self, input_ids):
    #         self.input_ids = input_ids
    # valenc = TokenizerWrapper(valenc)

def get_gsm(nsamples, seed, seqlen, model):
    VALSAMPLES=128
    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)
    GSM_TEMPLATE = "Question: {question}\nAnswer:"
    def apply_template(sample):
        text = GSM_TEMPLATE.format(question=sample["question"])
        if "answer" in sample:
            text += " " + sample["answer"]
        return {"text": text}

    traindata = load_dataset("gsm8k", "main", split='train')
    traindata = traindata.map(
        apply_template,
        remove_columns=list(traindata.features))

    np.random.seed(seed)
    dataloader = []
    perm = np.random.permutation(len(traindata))

    for i in perm:
        trainenc = tokenizer(traindata[int(i)]['text'], return_tensors='pt')
        # if long, truncate from the end like in Eldar's code. not sure how much matters
        inp = trainenc.input_ids[:, :seqlen]
        dataloader.append(inp)
        if len(dataloader) == nsamples + VALSAMPLES:
            break
    trainloader = dataloader[VALSAMPLES:]
    testloader = dataloader[:VALSAMPLES]
    for i in range(len(trainloader)):
        trainloader[i] = trainloader[i][0]
    return trainloader, testloader

def get_gsm_concat(nsamples, seed, seqlen, model):
    VALSAMPLES = 128
    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)
    GSM_TEMPLATE = "Question: {question}\nAnswer:"
    def apply_template(sample):
        text = GSM_TEMPLATE.format(question=sample["question"])
        if "answer" in sample:
            text += " " + sample["answer"]
        return {"text": text}

    traindata = load_dataset("gsm8k", "main", split='train')
    traindata = traindata.map(
        apply_template,
        remove_columns=list(traindata.features))

    np.random.seed(seed)
    perm = np.random.permutation(len(traindata))
    dataloader = []
    size = 0
    tmp = []
    cnt = 0
    for i in perm:
        cnt += 1
        tokens = tokenizer(traindata[int(i)]['text'], return_tensors='pt').input_ids
        if size + tokens.shape[1] < seqlen:
            pass
        else:
            dataloader.append( torch.cat(tmp, dim=1) )
            size = 0
            tmp = []
        size += tokens.shape[1]
        tmp.append(tokens)

        if len(dataloader) == nsamples + VALSAMPLES:
            break
    trainloader = dataloader[VALSAMPLES:]
    testloader = dataloader[:VALSAMPLES]
    for i in range(len(trainloader)):
        trainloader[i] = trainloader[i][0]
    print(f'gsm_concat counter: {cnt}')
    return trainloader, testloader

def get_loaders(
    name, nsamples=256, seed=0, seqlen=2048, model=''
):
    if name == 'wikitext2':
        return get_wikitext2(nsamples, seed, seqlen, model)
    if name == 'wikitext2_partition':
        return get_wikitext2_partition(nsamples, seed, seqlen, model)
    if name=='red':
        return get_red(nsamples, seed, seqlen, model)
    if name=='red_concat':
        return get_red_concat(nsamples, seed, seqlen, model)
    if name=='c4':
        return get_c4_new(nsamples, seed, seqlen, model)
    if name == 'gsm':
        return get_gsm(nsamples, seed, seqlen, model)
    if name == 'gsm_concat':
        return get_gsm_concat(nsamples, seed, seqlen, model)


# depreciated
# def proc_data(args, model, tokenizer, dataset, dataset_val):
#     #create a directory called 'data' if it doesn't exist
#     if not os.path.exists(os.path.join(args.output, 'data')):
#         os.makedirs(os.path.join(args.output, 'data'))
# 
#     if not os.path.exists(os.path.join(args.output, 'data/token_windows.pt')):
#         long_text = ' '.join(dataset['text'])
#         # Tokenize the concatenated text
#         tokens = tokenizer(long_text, return_tensors='pt', truncation=False, padding=False)['input_ids'][0]
# 
#         # Split the tokens into windows of length context_size. I am dividing by 2 because of OOM, if you have a A100 or H100, you can use full context_size
#         # window_size = model.config.max_position_embeddings//2
#         window_size = args.window_size
#         #break tokens tensor into windows of size window_size
#         tokens = tokens[0:window_size*int(len(tokens)/window_size)]
#         token_windows = tokens.view(-1,window_size)
#         #save the token_windows
#         torch.save(token_windows, os.path.join(args.output, 'data/token_windows.pt'))
#     else:
#         token_windows = torch.load(os.path.join(args.output, 'data/token_windows.pt'))
# 
#     if not os.path.exists(os.path.join(args.output, 'data/token_windows_val.pt')):
#         long_text = ' '.join(dataset_val['text'])
# 
#         # Tokenize the concatenated text
#         tokens = tokenizer(long_text, return_tensors='pt', truncation=False, padding=False)['input_ids'][0]
# 
#         # Split the tokens into windows of length context_size. I am dividing by 2 because of OOM, if you have a A100 or H100, you can use full context_size
#         window_size = model.config.max_position_embeddings//2
#         tokens = tokens[0:window_size*int(len(tokens)/window_size)]
#         token_windows_val = tokens.view(-1,window_size)
#         torch.save(token_windows_val, os.path.join(args.output, 'data/token_windows_val.pt'))
#     else:
#         token_windows_val = torch.load(os.path.join(args.output, 'data/token_windows_val.pt'))
# 
#     return token_windows, token_windows_val
