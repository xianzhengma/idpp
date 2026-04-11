import math
from inspect import isfunction
import torch
from torch import nn
import torch.distributed as dist

def gather_data(data, return_np=True):
    data_list = [torch.zeros_like(data) for _ in range(dist.get_world_size())]
    dist.all_gather(data_list, data)
    if return_np:
        data_list = [data.cpu().numpy() for data in data_list]
    return data_list

def autocast(f):

    def do_autocast(*args, **kwargs):
        with torch.cuda.amp.autocast(enabled=True, dtype=torch.get_autocast_gpu_dtype(), cache_enabled=torch.is_autocast_cache_enabled()):
            return f(*args, **kwargs)
    return do_autocast

def extract_into_tensor(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *(1,) * (len(x_shape) - 1))

def noise_like(shape, device, repeat=False):
    repeat_noise = lambda: torch.randn((1, *shape[1:]), device=device).repeat(shape[0], *(1,) * (len(shape) - 1))
    noise = lambda: torch.randn(shape, device=device)
    return repeat_noise() if repeat else noise()

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

def exists(val):
    return val is not None

def identity(*args, **kwargs):
    return nn.Identity()

def uniq(arr):
    return {el: True for el in arr}.keys()

def mean_flat(tensor):
    return tensor.mean(dim=list(range(1, len(tensor.shape))))

def ismap(x):
    if not isinstance(x, torch.Tensor):
        return False
    return len(x.shape) == 4 and x.shape[1] > 3

def isimage(x):
    if not isinstance(x, torch.Tensor):
        return False
    return len(x.shape) == 4 and (x.shape[1] == 3 or x.shape[1] == 1)

def max_neg_value(t):
    return -torch.finfo(t.dtype).max

def shape_to_str(x):
    shape_str = 'x'.join([str(x) for x in x.shape])
    return shape_str

def init_(tensor):
    dim = tensor.shape[-1]
    std = 1 / math.sqrt(dim)
    tensor.uniform_(-std, std)
    return tensor
ckpt = torch.utils.checkpoint.checkpoint

def checkpoint(func, inputs, params, flag):
    if flag:
        return ckpt(func, *inputs, use_reentrant=False)
    else:
        return func(*inputs)
