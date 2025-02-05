import copy
import math
import time

import torch
import torch.nn as nn
import transformers

from quant import *
import os
import sys
# sys.path.append(os.path.abspath('..'))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# TODO fix
from quiputils import RHT, RHT_inv, RHT_H

DEBUG = False 

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


class GPTQ:

    def __init__(self, layer, stable=False):
        self.layer = layer
        self.dev = self.layer.weight.device
        W = layer.weight.data.clone()
        self.rows = W.shape[0]
        self.columns = W.shape[1]
        self.H = torch.zeros((self.columns, self.columns), device=self.dev)
        self.nsamples = 0
        self.stable = stable
        self.mean = torch.zeros((self.columns, 1), device=self.dev)

    def add_batch(self, inp, out):
        if DEBUG:
            self.inp1 = inp
            self.out1 = out
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if len(inp.shape) == 3:
            inp = inp.reshape((-1, inp.shape[-1]))
        inp = inp.t()

        if self.stable:
            inp = inp.float()
            delta = torch.mean(inp, 1, keepdims=True) - self.mean
            self.H += inp.matmul(inp.t()) + delta.matmul(delta.t()) * self.nsamples * tmp / (self.nsamples + tmp)
            self.nsamples += tmp
            self.mean += delta * tmp / self.nsamples
        else:
            self.H *= self.nsamples / (self.nsamples + tmp)
            self.nsamples += tmp
            inp = math.sqrt(2 / self.nsamples) * inp.float()
            self.H += inp.matmul(inp.t())

    def fasterquant(
        self, blocksize=128, percdamp=.1, groupsize=-1, quip=False, 
        clip=False, baseline=False, actorder=False
    ):
        W = self.layer.weight.data.clone()
        W = W.float()

        tick = time.time()

        if self.stable:
            self.H /= self.nsamples
            self.H += self.mean.matmul(self.mean.t())
            self.H *= 2
        H = self.H
        del self.H

        Losses = torch.zeros_like(W)
        Q = torch.zeros_like(W)

        # baseline is RTN
        if quip:
            S_in  = torch.randint(0, 2, (self.layer.in_features,)) * 2 - 1
            S_in  = S_in.to(self.layer.weight.device).to(self.layer.weight.dtype)
            S_out = torch.randint(0, 2, (self.layer.out_features,)) * 2 - 1
            S_out = S_out.to(self.layer.weight.device).to(self.layer.weight.dtype)

        if not baseline:
            try:
                damp = percdamp * torch.mean(torch.diag(H))
                diag = torch.arange(self.columns, device=self.dev)
                H[diag, diag] += damp

                if quip:
                    W = RHT(W, S_in, S_out) # BEFORE find_params()
                    H = RHT_H(H, S_in)
            
                if actorder:
                    perm = torch.argsort(torch.diag(H), descending=True)
                    W = W[:, perm]
                    H = H[perm][:, perm]
                    invperm = torch.argsort(perm)

                H = torch.linalg.cholesky(H)
                H = torch.cholesky_inverse(H)
                H = torch.linalg.cholesky(H, upper=True)
                Hinv = H
            except:
                print('Singularity.')
                baseline = True
        if baseline:
            if quip:
                W = RHT(W, S_in, S_out) # BEFORE find_params()
            del H
            Hinv = torch.eye(self.columns, device=self.dev)

        groups = []
        if groupsize == -1:
            self.quantizer.find_params(W)

        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1

            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]

                if groupsize != -1:
                    if (i1 + i) % groupsize == 0:
                        self.quantizer.find_params(
                            W[:, (i1 + i):(i1 + i + groupsize)], solve=Hinv1 if clip else None)
                        groups.append(copy.deepcopy(self.quantizer))
                q = quantize(
                    w.unsqueeze(1), self.quantizer.scale, self.quantizer.zero, self.quantizer.maxq
                ).flatten()

                Q1[:, i] = q
                Losses1[:, i] = (w - q) ** 2 / d ** 2

                err1 = (w - q) / d
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1

            Q[:, i1:i2] = Q1
            Losses[:, i1:i2] = Losses1 / 2
            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

            if DEBUG:
                self.layer.weight.data[:, :i2] = Q[:, :i2]
                self.layer.weight.data[:, i2:] = W[:, i2:]
                print(torch.sum((self.layer(self.inp1) - self.out1) ** 2))
                print(torch.sum(Losses))

        if (not baseline) and actorder:
            Q = Q[:, invperm]

        if quip:
            Q = RHT_inv(Q.reshape(self.layer.weight.shape), S_in, S_out).contiguous()

        torch.cuda.synchronize()
        print('time %.2f' % (time.time() - tick))
        print('error', torch.sum(Losses).item(), flush=True)

        self.layer.weight.data = Q.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)
        if DEBUG:
            print(torch.sum((self.layer(self.inp1) - self.out1) ** 2))

        if groups:
            scale = torch.cat([q.scale for q in groups], dim=1)
            zero = torch.cat([q.zero for q in groups], dim=1)
            return scale, zero
        else:
            return self.quantizer.scale, self.quantizer.zero
