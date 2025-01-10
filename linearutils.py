import torch
from quantutils import DeltaInteger, GroupFinite
from quiputils import RHT, quant_linear

import contextlib
import copy

class quantize_linearlayer_multimode(torch.nn.Module):
    '''
    Wrapper class to quantize linear layer. Contains original layer weights, 
    as well as "x" parameters which interpolate between the closest up and down
    quantization grid for each weight. 
    self.grid abstracts the quantization grid, and exposes generic round_down and round_up
    functions.
    ''' 
    @torch.no_grad()
    def __init__(self, linear, args):
        super().__init__()
        self.mode = 'x' # 'x', 'rdx', 'original', 'greedy'
        self.linear = linear
        if args.quip:
            self.S_in  = torch.randint(0, 2, (linear.in_features,)) * 2 - 1
            self.S_out = torch.randint(0, 2, (linear.out_features,)) * 2 - 1
            self.S_in  = self.S_in.to(linear.weight.device).to(linear.weight.dtype)
            self.S_out = self.S_out.to(linear.weight.device).to(linear.weight.dtype)
            self.linear.weight.data = RHT(self.linear.weight.data, self.S_in, self.S_out)
            
        self.linear = torch.compile(linear)

        self.grid = GroupFinite(
            linear.weight, args.wbits, args.groupsize, args.cache_down, args.cache_up, 
            with_grad=False)

        if args.init_x == 'rand':
            init_val = torch.rand_like(self.linear.weight)
        elif args.init_x == 'orig':
            # init_val = self._gen_y().to(self.linear.weight.dtype).to(self.linear.weight.device).detach().clone()
            init_val = self._gen_y().to(self.linear.weight.dtype).detach().clone()
        else:
            raise NotImplementedError
        self.x = torch.nn.Parameter(init_val)
        self.quant_args = args

        y = self._gen_y()
        assert torch.all(0 - 1e-6 <= y) and torch.all(y <= 1 + 1e-6)
        greedy = self._unquant(y, rd=True) - self.linear.weight
        self.delta2 = 12 * greedy.square().sum() / y.numel()
        self.grid._test_compression( self._unquant(self.x, rd=True) )

    def _unquant(self, a, rd=False):
        '''
        entry-wise interpolation between self.grid.round_down and round_up.
        '''
        W = self.linear.weight
        if rd:
            a = torch.round(a)
        return (1-a) * self.grid.round_down(W) + a * self.grid.round_up(W)
    
    @torch.no_grad()
    @torch.compile(dynamic=True)
    def _gen_y(self):
        '''
        y is the interpolation between the closest down and up quantizatoin
        gridpoints which gives the original weights
        '''
        W = self.linear.weight
        up = self.grid.round_up(W)
        down = self.grid.round_down(W)
        y = (W - down) / (up - down)
        y = torch.where(torch.isclose(up, down), 0.5, y)
        return y

    @torch.compile(dynamic=True)
    def _d_sigmoid(self, a):
        return torch.exp( -a.abs() ) / (1 + torch.exp( -a.abs() )).square()

    @torch.no_grad()
    @torch.compile(dynamic=True)
    def _c_grad(self):
        '''
        gradient of linear term which encourages rounding
        '''
        W = self.linear.weight
        delta2_i = 1
        c = (2 * self._gen_y() - 1)
        return delta2_i * c

    @torch.no_grad()
    def _projection_step(self):
        self.x.data.clamp_(0, 1)
    
    @torch.no_grad()
    @torch.compile(dynamic=True)
    def _num_rounded(self):
        x_val = self.x
        return torch.sum( 
                torch.isclose( 
                    x_val, 
                    torch.tensor(0.0, dtype=x_val.dtype, device=x_val.device), 
                    atol=1e-2
                ) | torch.isclose(
                    x_val, 
                    torch.tensor(1.0, dtype=x_val.dtype, device=x_val.device), 
                    atol=1e-2)
                ).item()

    def forward(self, input):
        '''
        Each quantized linear layer contains multiple modes, or multiple models.
        "x" is the non-rounded version of our quantization parameter.
        "rdx" rounds our quantization parameter.
        "original" is the original weights.
        "greedy" is the original weights greedily quantized.
        '''
        bias = 0
        if self.linear.bias is not None: bias = self.linear.bias
        if self.mode == 'x':
            Wq = self._unquant(self.x, rd=False)
        elif self.mode == 'rdx':
            Wq = self._unquant(self.x, rd=True)
        elif self.mode == 'original':
            Wq = self.linear.weight
        elif self.mode == 'greedy':
            Wq = self._unquant(self._gen_y(), rd=True)
        else:
            raise ValueError('Invalid Mode')

        # no_grad() if not x
        ctx_mgr = contextlib.nullcontext() if self.mode=='x' else torch.no_grad()
        with ctx_mgr:
            if self.quant_args.quip:
                return quant_linear(input, Wq, bias, self.S_in, self.S_out)
            else:
                return input @ Wq.T + bias


def set_mode(model,mode):
    if mode not in ['x','rdx','original','greedy']:
        raise ValueError('Invalid mode')
    for module in model.modules():
        #if isinstance(module,quantize_linearlayer_multimode):
        if str(type(module))==str(quantize_linearlayer_multimode):
            module.mode=mode


def mygetattr(obj, attr_path):
    """
    Recursively get an attribute from an object using a dotted path.
    
    :param obj: The base object to get attributes from.
    :param attr_path: A string containing the dotted path of attribute names.
    :param default: A default value to return if the attribute is not found.
    :return: The value of the nested attribute, or the default if not found.
    """
    attributes = attr_path.split('.')
    for attr in attributes:
        obj = getattr(obj, attr)
    return obj


def mysetattr(obj, attr_path, value):
    """
    Recursively set an attribute on an object using a dotted path.
    
    :param obj: The base object to set attributes on.
    :param attr_path: A string containing the dotted path of attribute names.
    :param value: The value to set on the final attribute.
    """
    attributes = attr_path.split('.')
    for attr in attributes[:-1]:
        # Retrieve the next level object 
        obj = getattr(obj, attr)
    # Set the final attribute
    setattr(obj, attributes[-1], value)


def quantize_model(model, quantlist, args):
    for param in model.parameters():
        param.requires_grad = False
    
    for layer in model.model.layers:
        for name in quantlist:
            mysetattr(layer, name, quantize_linearlayer_multimode(mygetattr(layer, name), args))