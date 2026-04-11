"""
LoRA (Low-Rank Adaptation) 模块
用于给UNet添加LoRA层，实现参数高效微调
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import os


class LoRALinearLayer(nn.Module):
    """
    LoRA线性层，用于替换普通线性层
    
    参数:
        in_features (int): 输入特征维度
        out_features (int): 输出特征维度
        rank (int): LoRA的秩，决定了额外参数的数量
        alpha (float): 缩放因子，用于控制LoRA的影响
        dropout (float): dropout概率
    """
    def __init__(self, in_features, out_features, rank=4, alpha=1.0, dropout=0.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        
        # 原始线性层，参数冻结
        self.linear = nn.Linear(in_features, out_features, bias=False)
        # 冻结原始参数
        for param in self.linear.parameters():
            param.requires_grad = False
            
        # LoRA层
        self.lora_down = nn.Linear(in_features, rank, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.lora_up = nn.Linear(rank, out_features, bias=False)
        
        # 初始化LoRA层
        nn.init.normal_(self.lora_down.weight, std=1 / rank)
        nn.init.zeros_(self.lora_up.weight)
        
        # 缩放因子
        self.scale = alpha / rank
        
    def forward(self, x):
        # 原始线性层输出
        base_output = self.linear(x)
        
        # LoRA路径
        lora_output = self.lora_up(self.dropout(self.lora_down(x)))
        
        # 合并输出
        return base_output + lora_output * self.scale


class LoRAConv2dLayer(nn.Module):
    """
    LoRA卷积层，用于替换普通2D卷积层
    
    参数:
        in_channels (int): 输入通道数
        out_channels (int): 输出通道数
        kernel_size (int): 卷积核大小
        stride (int): 步长
        padding (int): 填充
        rank (int): LoRA的秩
        alpha (float): 缩放因子
        dropout (float): dropout概率
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, rank=4, alpha=1.0, dropout=0.0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.rank = rank
        self.alpha = alpha
        
        # 原始卷积层，参数冻结
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        # 冻结原始参数
        for param in self.conv.parameters():
            param.requires_grad = False
            
        # LoRA层 - 使用1x1卷积降维，然后使用kxk卷积升维
        self.lora_down = nn.Conv2d(in_channels, rank, 1, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.lora_up = nn.Conv2d(rank, out_channels, kernel_size, stride, padding, bias=False)
        
        # 初始化LoRA层
        nn.init.normal_(self.lora_down.weight, std=1 / rank)
        nn.init.zeros_(self.lora_up.weight)
        
        # 缩放因子
        self.scale = alpha / rank
        
    def forward(self, x):
        # 原始卷积层输出
        base_output = self.conv(x)
        
        # LoRA路径
        lora_output = self.lora_up(self.dropout(self.lora_down(x)))
        
        # 合并输出
        return base_output + lora_output * self.scale


class LoRAConv3dLayer(nn.Module):
    """
    LoRA 3D卷积层，用于替换普通3D卷积层，主要用于时间卷积
    
    参数:
        in_channels (int): 输入通道数
        out_channels (int): 输出通道数
        kernel_size (tuple): 卷积核大小 (t, h, w)
        stride (tuple): 步长
        padding (tuple): 填充
        rank (int): LoRA的秩
        alpha (float): 缩放因子
        dropout (float): dropout概率
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, rank=4, alpha=1.0, dropout=0.0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.rank = rank
        self.alpha = alpha
        
        # 原始卷积层，参数冻结
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        # 冻结原始参数
        for param in self.conv.parameters():
            param.requires_grad = False
            
        # LoRA层 - 使用1x1x1卷积降维，然后使用原始卷积核大小升维
        self.lora_down = nn.Conv3d(in_channels, rank, 1, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.lora_up = nn.Conv3d(rank, out_channels, kernel_size, stride, padding, bias=False)
        
        # 初始化LoRA层
        nn.init.normal_(self.lora_down.weight, std=1 / rank)
        nn.init.zeros_(self.lora_up.weight)
        
        # 缩放因子
        self.scale = alpha / rank
        
    def forward(self, x):
        # 原始卷积层输出
        base_output = self.conv(x)
        
        # LoRA路径
        lora_output = self.lora_up(self.dropout(self.lora_down(x)))
        
        # 合并输出
        return base_output + lora_output * self.scale


def replace_unet_modules_with_lora(model, rank=16, alpha=32, dropout=0.1, target_replace_module=None, exact_matches=None):
    """
    将UNet模型中的目标模块替换为带有LoRA层的模块
    
    参数:
        model (nn.Module): UNet模型
        rank (int): LoRA的秩
        alpha (float): 缩放因子
        dropout (float): dropout概率
        target_replace_module (list): 要替换的模块名称列表
        exact_matches (list): 要精确匹配的完整模块路径列表
        
    返回:
        nn.Module: 带有LoRA层的UNet模型
    """
    if target_replace_module is None:
        # 默认替换的模块
        target_replace_module = [
            # 时间注意力中的QKV投影
            "to_q", "to_k", "to_v", "to_out.0",
            # 空间注意力中的QKV投影
            "proj_in", "proj_out",
            # 时间卷积层
            "temopral_conv.conv1.2", "temopral_conv.conv2.3", 
            "temopral_conv.conv3.3", "temopral_conv.conv4.3"
        ]

    # 确保exact_matches是列表
    exact_matches = exact_matches or []

    # 记录所有替换的模块
    lora_module_names = []
    
    # 递归遍历模型的所有模块
    for name, module in model.named_modules():
        # 检查是否应该应用LoRA - 子字符串匹配或精确路径匹配
        if name in exact_matches or any(target_name in name for target_name in target_replace_module):
            # 对于Linear层
            if isinstance(module, nn.Linear):
                parent_name = name.rsplit(".", 1)[0]
                child_name = name.rsplit(".", 1)[1]
                parent_module = model.get_submodule(parent_name)
                
                # 创建LoRA层替换原始线性层
                lora_layer = LoRALinearLayer(
                    module.in_features, 
                    module.out_features,
                    rank=rank,
                    alpha=alpha,
                    dropout=dropout
                )
                # 复制原始权重
                lora_layer.linear.weight.data = module.weight.data.clone()
                if hasattr(module, 'bias') and module.bias is not None:
                    lora_layer.linear.bias = nn.Parameter(module.bias.data.clone())
                
                # 替换模块
                setattr(parent_module, child_name, lora_layer)
                lora_module_names.append(name)
                
            # 对于Conv2d层
            elif isinstance(module, nn.Conv2d):
                parent_name = name.rsplit(".", 1)[0]
                child_name = name.rsplit(".", 1)[1]
                parent_module = model.get_submodule(parent_name)
                
                # 创建LoRA层替换原始卷积层
                lora_layer = LoRAConv2dLayer(
                    module.in_channels, 
                    module.out_channels,
                    module.kernel_size[0],
                    stride=module.stride[0],
                    padding=module.padding[0],
                    rank=rank,
                    alpha=alpha,
                    dropout=dropout
                )
                # 复制原始权重
                lora_layer.conv.weight.data = module.weight.data.clone()
                if hasattr(module, 'bias') and module.bias is not None:
                    lora_layer.conv.bias = nn.Parameter(module.bias.data.clone())
                
                # 替换模块
                setattr(parent_module, child_name, lora_layer)
                lora_module_names.append(name)
                
            # 对于Conv3d层
            elif isinstance(module, nn.Conv3d):
                parent_name = name.rsplit(".", 1)[0]
                child_name = name.rsplit(".", 1)[1]
                parent_module = model.get_submodule(parent_name)
                
                # 创建LoRA层替换原始3D卷积层
                lora_layer = LoRAConv3dLayer(
                    module.in_channels, 
                    module.out_channels,
                    module.kernel_size,
                    stride=module.stride,
                    padding=module.padding,
                    rank=rank,
                    alpha=alpha,
                    dropout=dropout
                )
                # 复制原始权重
                lora_layer.conv.weight.data = module.weight.data.clone()
                if hasattr(module, 'bias') and module.bias is not None:
                    lora_layer.conv.bias = nn.Parameter(module.bias.data.clone())
                
                # 替换模块
                setattr(parent_module, child_name, lora_layer)
                lora_module_names.append(name)
    
    # 打印替换的模块信息
    print(f"替换了 {len(lora_module_names)} 个模块为LoRA层")
    print(f"其中包括 {len(exact_matches)} 个精确匹配的模块")
    
    return model


def get_lora_parameters(model):
    """
    获取模型中所有LoRA参数
    
    参数:
        model (nn.Module): 包含LoRA层的模型
        
    返回:
        list: LoRA参数列表
    """
    lora_parameters = []
    
    # 遍历模型参数
    for name, param in model.named_parameters():
        # 只选择LoRA参数
        if "lora_" in name and param.requires_grad:
            lora_parameters.append(param)
            
    return lora_parameters


def save_lora_weights(model, path):
    """
    保存模型中的LoRA权重
    
    参数:
        model (nn.Module): 包含LoRA层的模型
        path (str): 保存路径
    """
    # 创建保存字典
    lora_state_dict = {}
    
    # 遍历模型参数
    for name, param in model.named_parameters():
        # 只保存LoRA参数
        if "lora_" in name and param.requires_grad:
            lora_state_dict[name] = param.data.clone()
            
    # 保存权重
    torch.save(lora_state_dict, path)
    print(f"LoRA权重已保存到 {path}")


def load_lora_weights(model, path):
    """
    加载LoRA权重到模型
    
    参数:
        model (nn.Module): 包含LoRA层的模型
        path (str): 权重文件路径
    """
    # 加载权重
    lora_state_dict = torch.load(path)
    
    # 加载权重到模型
    for name, param in model.named_parameters():
        if name in lora_state_dict and param.requires_grad:
            param.data = lora_state_dict[name]
            
    print(f"已从 {path} 加载LoRA权重")


def get_pdecoder_parameters(model):
    """
    获取模型中所有PDecoder参数
    
    参数:
        model (nn.Module): 包含PDecoder的模型
        
    返回:
        list: PDecoder参数列表
    """
    pdecoder_parameters = []
    
    # 遍历模型参数
    for name, param in model.named_parameters():
        # 只选择PDecoder参数
        if "p_decoder" in name and param.requires_grad:
            pdecoder_parameters.append(param)
            
    return pdecoder_parameters


def save_pdecoder_weights(model, path):
    """
    保存模型中的PDecoder权重
    
    参数:
        model (nn.Module): 包含PDecoder的模型
        path (str): 保存路径
    """
    # 创建保存字典
    pdecoder_state_dict = {}
    
    # 遍历模型参数
    for name, param in model.named_parameters():
        # 只保存PDecoder参数
        if "p_decoder" in name and param.requires_grad:
            pdecoder_state_dict[name] = param.data.clone()
            
    # 保存权重
    torch.save(pdecoder_state_dict, path)
    print(f"PDecoder权重已保存到 {path}")


def load_pdecoder_weights(model, path):
    """
    加载PDecoder权重到模型
    
    参数:
        model (nn.Module): 包含PDecoder的模型
        path (str): 权重文件路径
    """
    # 加载权重
    pdecoder_state_dict = torch.load(path)
    
    # 加载权重到模型
    for name, param in model.named_parameters():
        if name in pdecoder_state_dict and param.requires_grad:
            param.data = pdecoder_state_dict[name]
            
    print(f"已从 {path} 加载PDecoder权重")


def save_trained_parameters(model, prefix_path, training_config=None):
    """
    保存模型中所有训练的参数（LoRA和PDecoder）
    
    参数:
        model (nn.Module): 包含LoRA和PDecoder的模型
        prefix_path (str): 保存路径前缀，不包含扩展名
        training_config (dict, optional): 训练配置信息，将一并保存
    """
    # 创建目录（如果不存在）
    os.makedirs(os.path.dirname(prefix_path), exist_ok=True)
    
    # 创建保存字典
    trained_state_dict = {
        "lora_params": {},
        "pdecoder_params": {},
        "training_config": training_config or {}
    }
    
    # 保存版本信息和时间戳
    trained_state_dict["version"] = "1.0"
    import time
    trained_state_dict["timestamp"] = time.time()
    
    # 遍历模型参数
    for name, param in model.named_parameters():
        # 保存LoRA参数
        if "lora_" in name and param.requires_grad:
            trained_state_dict["lora_params"][name] = param.data.clone()
        # 保存PDecoder参数
        elif "p_decoder" in name and param.requires_grad:
            trained_state_dict["pdecoder_params"][name] = param.data.clone()
    
    # 保存所有训练参数
    full_path = f"{prefix_path}.pt"
    torch.save(trained_state_dict, full_path)
    print(f"所有训练参数已保存到 {full_path}")
    
    # 返回保存的路径
    return full_path


def load_trained_parameters(model, path, load_lora=True, load_pdecoder=True):
    """
    从文件加载训练参数到模型
    
    参数:
        model (nn.Module): 要加载参数的模型
        path (str): 参数文件路径
        load_lora (bool): 是否加载LoRA参数
        load_pdecoder (bool): 是否加载PDecoder参数
        
    返回:
        dict: 加载的训练配置信息
    """
    # 增强日志
    print(f"[LoRA模块] 开始从 {path} 加载训练参数")
    print(f"[LoRA模块] 加载选项: load_lora={load_lora}, load_pdecoder={load_pdecoder}")
    
    try:
        # 加载保存的字典
        trained_state_dict = torch.load(path)
        print(f"[LoRA模块] 成功加载参数文件，文件大小: {os.path.getsize(path)/1024/1024:.2f} MB")
        
        # 打印训练参数文件的基本信息
        keys = list(trained_state_dict.keys())
        print(f"[LoRA模块] 参数文件包含的键: {keys}")
        if "version" in trained_state_dict:
            print(f"[LoRA模块] 参数文件版本: {trained_state_dict['version']}")
        if "timestamp" in trained_state_dict:
            import datetime
            ts = datetime.datetime.fromtimestamp(trained_state_dict["timestamp"])
            print(f"[LoRA模块] 参数文件创建时间: {ts}")
        
        # 检查版本兼容性
        if "version" in trained_state_dict and trained_state_dict["version"] != "1.0":
            print(f"[LoRA模块] 警告：参数文件版本 {trained_state_dict['version']} 可能与当前代码不兼容")
        
        # 加载LoRA参数
        if load_lora and "lora_params" in trained_state_dict:
            lora_params = trained_state_dict["lora_params"]
            print(f"[LoRA模块] 找到 {len(lora_params)} 个LoRA参数")
            loaded_count = 0
            for name, param in model.named_parameters():
                if name in lora_params and param.requires_grad:
                    param.data = lora_params[name]
                    loaded_count += 1
            print(f"[LoRA模块] 已加载 {loaded_count} 个LoRA参数 (在模型中找到 {loaded_count}/{len(lora_params)} 个匹配参数)")
        elif load_lora:
            print(f"[LoRA模块] 警告：参数文件中未找到LoRA参数")
        
        # 加载PDecoder参数
        if load_pdecoder and "pdecoder_params" in trained_state_dict:
            pdecoder_params = trained_state_dict["pdecoder_params"]
            print(f"[LoRA模块] 找到 {len(pdecoder_params)} 个PDecoder参数")
            loaded_count = 0
            for name, param in model.named_parameters():
                if name in pdecoder_params and param.requires_grad:
                    param.data = pdecoder_params[name]
                    loaded_count += 1
            print(f"[LoRA模块] 已加载 {loaded_count} 个PDecoder参数 (在模型中找到 {loaded_count}/{len(pdecoder_params)} 个匹配参数)")
        elif load_pdecoder:
            print(f"[LoRA模块] 警告：参数文件中未找到PDecoder参数")
        
        # 返回训练配置
        return trained_state_dict.get("training_config", {})
    except Exception as e:
        import traceback
        print(f"[LoRA模块] 加载参数文件时发生错误: {e}")
        print(traceback.format_exc())
        return {} 