import numpy as np
import torch
from contextlib import nullcontext


class BaseGrid(torch.nn.Module):
    def __init__(self, cache_down=False, cache_up=False,
        with_grad=False):
        super().__init__()
        self.cache_down = cache_down
        self.cache_up   = cache_up
        self.grid_down  = None
        self.grid_up    = None
        self.with_grad  = with_grad
        self.ctx_grad   = nullcontext() if self.with_grad else torch.no_grad()

    @torch.compile(dynamic=True)
    def round_down(self, W):
        with self.ctx_grad:
            if self.cache_down:
                assert self.grid_down is not None
                return self.grid_down
            else:
                return self._round_down(W)
        
    @torch.compile(dynamic=True)
    def round_up(self, W):
        with self.ctx_grad:
            if self.cache_up:
                assert self.grid_up is not None
                return self.grid_up
            else:
                return self._round_up(W)


class DeltaInteger(BaseGrid):
    @torch.no_grad()
    def __init__(self, W, delta, cache_down=False, cache_up=False, with_grad=False):
        assert with_grad == False, "with_grad not implemented for DeltaInteger!"
        super().__init__(cache_down=cache_down, cache_up=cache_up)
        self.delta = delta
        if self.cache_down:
            self.grid_down = self._round_down(W)
        if self.cache_up:
            self.grid_up   = self._round_up(W)
        self.num_gridpts = np.inf
    
    @torch.no_grad()
    def _test_compression(self, Wx):
        "input is unquant weights"
        return
    
    def _round_down(self, W):
        return torch.floor(W / self.delta) * self.delta
    
    def _round_up(self, W):
        return torch.ceil(W / self.delta) * self.delta

class FloorSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        # ctx.save_for_backward(input)
        return torch.floor(input)

    @staticmethod
    def backward(ctx, grad_output):
        # input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        return grad_input

class CeilSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        # ctx.save_for_backward(input)
        return torch.ceil(input)

    @staticmethod
    def backward(ctx, grad_output):
        # input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        return grad_input

class ClampSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, min, max):
        # ctx.save_for_backward(input)
        return torch.clamp(input, min, max)

    @staticmethod
    def backward(ctx, grad_output):
        # input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        return grad_input


class GroupFinite(BaseGrid):
    @torch.no_grad()
    def __init__(self, W, wbits, groupsize, cache_down=False, cache_up=False, with_grad=False):
        super().__init__(cache_down=cache_down, cache_up=cache_up, with_grad=with_grad)
        self.wbits     = wbits
        self.groupsize = groupsize
        from gptq.quant import Quantizer
        self.quantizer = Quantizer()
        self.quantizer.configure(self.wbits)
        self.rows = W.shape[0]
        self.cols = W.shape[1]

        if self.groupsize == -1:
            self.quantizer.find_params(W)
            self.scales = self.quantizer.scale.to(W.dtype)
            self.zeros = self.quantizer.zero.to(W.dtype)
        else:
            assert groupsize > 0
            assert self.cols % groupsize == 0
            col_iter = int(self.cols / self.groupsize)
            self.scales = torch.zeros(
                self.rows, col_iter, dtype=W.dtype, device=W.device)
                # self.rows, col_iter, dtype=W.dtype)
            self.zeros = torch.zeros_like(self.scales)
            for k in range(col_iter):
                i = self.groupsize * k
                j = self.groupsize * (k+1)
                w = W[:, i:j]
                self.quantizer.find_params(w)
                self.scales[:, k] = self.quantizer.scale.squeeze().to(W.dtype)
                self.zeros[:, k]  = self.quantizer.zero.squeeze().to(W.dtype)
            self.scales = self.scales.unsqueeze(2)
            self.zeros = self.zeros.unsqueeze(2)
        self.maxq = self.quantizer.maxq 
        self.num_gridpts = self.quantizer.maxq + 1

        # if with_grad, make scales Parameter() so gradients activated
        if self.with_grad:
            assert cache_down == False
            assert cache_up == False
            self.scales = torch.nn.Parameter(self.scales)

        if self.cache_down:
            self.grid_down = self._round_down(W)
        if self.cache_up:
            self.grid_up   = self._round_up(W)

    @torch.no_grad()
    def _test_compression(self, Wx):
        "input is unquant weights. sorts and checks consecutive weights per group"
        if self.groupsize == -1:
            gs = self.cols
        else:
            gs = self.groupsize
        Wx = Wx.view(
            self.rows, self.cols//gs, gs)
        Wx_sorted, _ = torch.sort(Wx, dim=-1)
        unique_mask = torch.cat([
            torch.ones((Wx_sorted.shape[0], Wx_sorted.shape[1], 1), device=Wx_sorted.device), 
            Wx_sorted[:, :, 1:] != Wx_sorted[:, :, :-1]], dim=-1)
        unique_counts = torch.sum(unique_mask, dim=-1)
        assert (unique_counts <= self.num_gridpts).all(), "Some groups have more unique elements than allowed"

    def _same_dev(self, W):
        if W.device != self.scales.device:
            self.scales = self.scales.to(W.device, non_blocking=True)
        if W.device != self.zeros.device:
            self.zeros = self.zeros.to(W.device, non_blocking=True)

    # def _round_down(self, W):
    #     self._same_dev(W)
    #     if self.groupsize != -1:
    #         W = W.view(self.rows, self.cols//self.groupsize, self.groupsize)
    #     Wq = ClampSTE.apply(FloorSTE.apply(W / self.scales) + self.zeros, 0, self.maxq)
    #     Wq = self.scales * (Wq - self.zeros)
    #     if self.groupsize != -1:
    #         Wq = Wq.view(self.rows, self.cols)
    #     return Wq

    # def _round_up(self, W):
    #     self._same_dev(W)
    #     if self.groupsize != -1:
    #         W = W.view(self.rows, self.cols//self.groupsize, self.groupsize)
    #     Wq = ClampSTE.apply(CeilSTE.apply(W / self.scales) + self.zeros, 0, self.maxq)
    #     Wq = self.scales * (Wq - self.zeros)
    #     if self.groupsize != -1:
    #         Wq = Wq.view(self.rows, self.cols)
    #     return Wq

    # @torch.no_grad()
    def _round_down(self, W):
        self._same_dev(W)
        if self.groupsize != -1:
            W = W.view(self.rows, self.cols//self.groupsize, self.groupsize)
        Wq = torch.clamp(torch.floor(W / self.scales) + self.zeros, 0, self.maxq)
        Wq = self.scales * (Wq - self.zeros)
        if self.groupsize != -1:
            Wq = Wq.view(self.rows, self.cols)
        return Wq

    # @torch.no_grad()
    def _round_up(self, W):
        self._same_dev(W)
        if self.groupsize != -1:
            W = W.view(self.rows, self.cols//self.groupsize, self.groupsize)
        Wq = torch.clamp(torch.ceil(W / self.scales) + self.zeros, 0, self.maxq)
        Wq = self.scales * (Wq - self.zeros)
        if self.groupsize != -1:
            Wq = Wq.view(self.rows, self.cols)
        return Wq