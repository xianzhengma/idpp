import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import os

class LoRALinearLayer(nn.Module):

    def __init__(self, in_features, out_features, rank=4, alpha=1.0, dropout=0.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.linear = nn.Linear(in_features, out_features, bias=False)
        for param in self.linear.parameters():
            param.requires_grad = False
        self.lora_down = nn.Linear(in_features, rank, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.lora_up = nn.Linear(rank, out_features, bias=False)
        nn.init.normal_(self.lora_down.weight, std=1 / rank)
        nn.init.zeros_(self.lora_up.weight)
        self.scale = alpha / rank

    def forward(self, x):
        base_output = self.linear(x)
        lora_output = self.lora_up(self.dropout(self.lora_down(x)))
        return base_output + lora_output * self.scale

class LoRAConv2dLayer(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, rank=4, alpha=1.0, dropout=0.0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.rank = rank
        self.alpha = alpha
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        for param in self.conv.parameters():
            param.requires_grad = False
        self.lora_down = nn.Conv2d(in_channels, rank, 1, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.lora_up = nn.Conv2d(rank, out_channels, kernel_size, stride, padding, bias=False)
        nn.init.normal_(self.lora_down.weight, std=1 / rank)
        nn.init.zeros_(self.lora_up.weight)
        self.scale = alpha / rank

    def forward(self, x):
        base_output = self.conv(x)
        lora_output = self.lora_up(self.dropout(self.lora_down(x)))
        return base_output + lora_output * self.scale

class LoRAConv3dLayer(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, rank=4, alpha=1.0, dropout=0.0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.rank = rank
        self.alpha = alpha
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        for param in self.conv.parameters():
            param.requires_grad = False
        self.lora_down = nn.Conv3d(in_channels, rank, 1, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.lora_up = nn.Conv3d(rank, out_channels, kernel_size, stride, padding, bias=False)
        nn.init.normal_(self.lora_down.weight, std=1 / rank)
        nn.init.zeros_(self.lora_up.weight)
        self.scale = alpha / rank

    def forward(self, x):
        base_output = self.conv(x)
        lora_output = self.lora_up(self.dropout(self.lora_down(x)))
        return base_output + lora_output * self.scale

def replace_unet_modules_with_lora(model, rank=16, alpha=32, dropout=0.1, target_replace_module=None, exact_matches=None):
    if target_replace_module is None:
        target_replace_module = ['to_q', 'to_k', 'to_v', 'to_out.0', 'proj_in', 'proj_out', 'temopral_conv.conv1.2', 'temopral_conv.conv2.3', 'temopral_conv.conv3.3', 'temopral_conv.conv4.3']
    exact_matches = exact_matches or []
    lora_module_names = []
    for name, module in model.named_modules():
        if name in exact_matches or any((target_name in name for target_name in target_replace_module)):
            if isinstance(module, nn.Linear):
                parent_name = name.rsplit('.', 1)[0]
                child_name = name.rsplit('.', 1)[1]
                parent_module = model.get_submodule(parent_name)
                lora_layer = LoRALinearLayer(module.in_features, module.out_features, rank=rank, alpha=alpha, dropout=dropout)
                lora_layer.linear.weight.data = module.weight.data.clone()
                if hasattr(module, 'bias') and module.bias is not None:
                    lora_layer.linear.bias = nn.Parameter(module.bias.data.clone())
                setattr(parent_module, child_name, lora_layer)
                lora_module_names.append(name)
            elif isinstance(module, nn.Conv2d):
                parent_name = name.rsplit('.', 1)[0]
                child_name = name.rsplit('.', 1)[1]
                parent_module = model.get_submodule(parent_name)
                lora_layer = LoRAConv2dLayer(module.in_channels, module.out_channels, module.kernel_size[0], stride=module.stride[0], padding=module.padding[0], rank=rank, alpha=alpha, dropout=dropout)
                lora_layer.conv.weight.data = module.weight.data.clone()
                if hasattr(module, 'bias') and module.bias is not None:
                    lora_layer.conv.bias = nn.Parameter(module.bias.data.clone())
                setattr(parent_module, child_name, lora_layer)
                lora_module_names.append(name)
            elif isinstance(module, nn.Conv3d):
                parent_name = name.rsplit('.', 1)[0]
                child_name = name.rsplit('.', 1)[1]
                parent_module = model.get_submodule(parent_name)
                lora_layer = LoRAConv3dLayer(module.in_channels, module.out_channels, module.kernel_size, stride=module.stride, padding=module.padding, rank=rank, alpha=alpha, dropout=dropout)
                lora_layer.conv.weight.data = module.weight.data.clone()
                if hasattr(module, 'bias') and module.bias is not None:
                    lora_layer.conv.bias = nn.Parameter(module.bias.data.clone())
                setattr(parent_module, child_name, lora_layer)
                lora_module_names.append(name)
    print(f'Replaced {len(lora_module_names)} modules with LoRA layers')
    print(f'Including {len(exact_matches)} exact name matches')
    return model

def get_lora_parameters(model):
    lora_parameters = []
    for name, param in model.named_parameters():
        if 'lora_' in name and param.requires_grad:
            lora_parameters.append(param)
    return lora_parameters

def save_lora_weights(model, path):
    lora_state_dict = {}
    for name, param in model.named_parameters():
        if 'lora_' in name and param.requires_grad:
            lora_state_dict[name] = param.data.clone()
    torch.save(lora_state_dict, path)
    print(f'LoRA weights saved to {path}')

def load_lora_weights(model, path):
    lora_state_dict = torch.load(path)
    for name, param in model.named_parameters():
        if name in lora_state_dict and param.requires_grad:
            param.data = lora_state_dict[name]
    print(f'Loaded LoRA weights from {path}')

def get_pdecoder_parameters(model):
    pdecoder_parameters = []
    for name, param in model.named_parameters():
        if 'p_decoder' in name and param.requires_grad:
            pdecoder_parameters.append(param)
    return pdecoder_parameters

def save_pdecoder_weights(model, path):
    pdecoder_state_dict = {}
    for name, param in model.named_parameters():
        if 'p_decoder' in name and param.requires_grad:
            pdecoder_state_dict[name] = param.data.clone()
    torch.save(pdecoder_state_dict, path)
    print(f'PDecoder weights saved to {path}')

def load_pdecoder_weights(model, path):
    pdecoder_state_dict = torch.load(path)
    for name, param in model.named_parameters():
        if name in pdecoder_state_dict and param.requires_grad:
            param.data = pdecoder_state_dict[name]
    print(f'Loaded PDecoder weights from {path}')

def save_trained_parameters(model, prefix_path, training_config=None):
    os.makedirs(os.path.dirname(prefix_path), exist_ok=True)
    trained_state_dict = {'lora_params': {}, 'pdecoder_params': {}, 'training_config': training_config or {}}
    trained_state_dict['version'] = '1.0'
    import time
    trained_state_dict['timestamp'] = time.time()
    for name, param in model.named_parameters():
        if 'lora_' in name and param.requires_grad:
            trained_state_dict['lora_params'][name] = param.data.clone()
        elif 'p_decoder' in name and param.requires_grad:
            trained_state_dict['pdecoder_params'][name] = param.data.clone()
    full_path = f'{prefix_path}.pt'
    torch.save(trained_state_dict, full_path)
    print(f'All trainable parameters saved to {full_path}')
    return full_path

def load_trained_parameters(model, path, load_lora=True, load_pdecoder=True):
    print(f'[LoRA] Loading trained parameters from {path}')
    print(f'[LoRA] Options: load_lora={load_lora}, load_pdecoder={load_pdecoder}')
    try:
        trained_state_dict = torch.load(path)
        print(f'[LoRA] Loaded checkpoint file, size: {os.path.getsize(path) / 1024 / 1024:.2f} MB')
        keys = list(trained_state_dict.keys())
        print(f'[LoRA] Checkpoint keys: {keys}')
        if 'version' in trained_state_dict:
            print(f"[LoRA] Checkpoint version: {trained_state_dict['version']}")
        if 'timestamp' in trained_state_dict:
            import datetime
            ts = datetime.datetime.fromtimestamp(trained_state_dict['timestamp'])
            print(f'[LoRA] Checkpoint created at: {ts}')
        if 'version' in trained_state_dict and trained_state_dict['version'] != '1.0':
            print(f"[LoRA] Warning: checkpoint version {trained_state_dict['version']} may be incompatible")
        if load_lora and 'lora_params' in trained_state_dict:
            lora_params = trained_state_dict['lora_params']
            print(f'[LoRA] Found {len(lora_params)} LoRA tensors in checkpoint')
            loaded_count = 0
            for name, param in model.named_parameters():
                if name in lora_params and param.requires_grad:
                    param.data = lora_params[name]
                    loaded_count += 1
            print(f'[LoRA] Loaded {loaded_count} LoRA tensors ({loaded_count}/{len(lora_params)} matched in model)')
        elif load_lora:
            print(f'[LoRA] Warning: no LoRA tensors in checkpoint')
        if load_pdecoder and 'pdecoder_params' in trained_state_dict:
            pdecoder_params = trained_state_dict['pdecoder_params']
            print(f'[LoRA] Found {len(pdecoder_params)} PDecoder tensors in checkpoint')
            loaded_count = 0
            for name, param in model.named_parameters():
                if name in pdecoder_params and param.requires_grad:
                    param.data = pdecoder_params[name]
                    loaded_count += 1
            print(f'[LoRA] Loaded {loaded_count} PDecoder tensors ({loaded_count}/{len(pdecoder_params)} matched in model)')
        elif load_pdecoder:
            print(f'[LoRA] Warning: no PDecoder tensors in checkpoint')
        return trained_state_dict.get('training_config', {})
    except Exception as e:
        import traceback
        print(f'[LoRA] Error loading checkpoint: {e}')
        print(traceback.format_exc())
        return {}
