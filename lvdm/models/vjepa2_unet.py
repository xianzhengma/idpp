import torch
import torch.nn as nn
from transformers import AutoModel, AutoVideoProcessor
import math
import re
from einops import repeat

# Helper function from LDM codebase
def timestep_embedding(timesteps, dim, max_period=10000, repeat_only=False):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    if not repeat_only:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=timesteps.device)
        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    else:
        embedding = repeat(timesteps, 'b -> b d', d=dim)
    return embedding


class VJEPA2_UNet_Wrapper(nn.Module):
    """
    VJEPA2_UNet_Wrapper 作为一个UNet的替代品，其内部使用VJEPA2作为骨干网络。
    它被设计成与LatentVisualDiffusionv84的接口兼容，接收潜在噪声、时间步和条件，
    并输出预测的噪声(model_output)和中间特征列表(feat_list)。

    输入:
        - x (torch.Tensor): 噪声化的潜在表示, [B, C_in, T, H, W]
        - timesteps (torch.Tensor): 时间步, [B]
        - context (dict): 条件字典, 包含 'c_crossattn' 和 'c_concat'
    
    输出:
        - tuple[torch.Tensor, list[torch.Tensor]]:
            - model_output (torch.Tensor): 预测的噪声，形状与输入x相同
            - feat_list (list[torch.Tensor]): 模拟UNet输出的中间特征层列表
    """
    def __init__(
        self,
        # --- UNet compatibility params ---
        in_channels: int,
        model_channels: int,
        out_channels: int,
        context_dim: int,
        # --- VJEPA2 specific params ---
        vjepa_model_size: str = 'giant',
        vjepa_checkpoint_path: str = None,
        # --- Compatibility params for v84 ---
        v84_feat_dims: list = None
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.context_dim = context_dim
        self.model_channels = model_channels

        # 1. 加载 VJEPA2 主干网络
        hf_repo = self._get_hf_repo(vjepa_model_size)
        print(f"Loading VJEPA2 model from: {hf_repo}")
        try:
            model_path = vjepa_checkpoint_path if vjepa_checkpoint_path else hf_repo
            self.vjepa = AutoModel.from_pretrained(model_path, output_hidden_states=True)
        except Exception as e:
            raise IOError(f"Failed to load VJEPA2 model: {e}")

        self.vjepa_hidden_dim = self.vjepa.config.hidden_size
        
        # 冻结VJEPA2的所有参数，因为我们只将其用作特征提取器
        for param in self.vjepa.parameters():
            param.requires_grad = False
        
        # 2. 时间步嵌入模块
        # 将时间步嵌入并投影到VJEPA2的隐藏维度
        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, self.vjepa_hidden_dim),
        )

        # 3. 输入投影模块
        # VJEPA2原生的patch embedding是为3通道RGB图像设计的。
        # 我们需要创建一个新的3D卷积层来处理来自VAE的4通道潜在变量。
        vjepa_patch_embed = self.vjepa.embeddings.patch_embeddings.projection
        self.input_proj = nn.Conv3d(
            in_channels=in_channels,
            out_channels=self.vjepa_hidden_dim,
            kernel_size=vjepa_patch_embed.kernel_size,
            stride=vjepa_patch_embed.stride
        )
        self.pos_embed = self.vjepa.embeddings.position_embeddings

        # 4. 条件上下文投影模块
        self.context_proj = nn.Linear(context_dim, self.vjepa_hidden_dim)

        # 5. 输出和特征列表解码器
        self.v84_feat_dims = v84_feat_dims or [320, 320, 320, 320, 640, 640, 640, 1280, 1280, 1280, 1280, 1280, 1280, 1280, 1280, 1280, 1280, 1280, 1280, 640, 640, 640, 320, 320, 320]
        
        # VJEPA2-giant有40个transformer层，我们需要从中选择25个来生成feat_list
        num_vjepa_layers = len(self.vjepa.encoder.layer)
        self.vjepa_layer_indices = torch.linspace(0, num_vjepa_layers - 1, len(self.v84_feat_dims)).long().tolist()

        # 为每个需要的特征创建一个线性投影层
        self.feat_list_projectors = nn.ModuleList([
            nn.Linear(self.vjepa_hidden_dim, c_out) for c_out in self.v84_feat_dims
        ])

        # 最终输出投影层，使用转置卷积进行上采样
        self.out_proj = nn.ConvTranspose3d(
            self.vjepa_hidden_dim, 
            out_channels, 
            kernel_size=vjepa_patch_embed.kernel_size, 
            stride=vjepa_patch_embed.stride
        )

    def _get_hf_repo(self, model_size: str) -> str:
        """根据模型大小构建HuggingFace仓库名称"""
        size_map = {'large': 'vitl', 'huge': 'vith', 'giant': 'vitg'}
        if model_size not in size_map:
            raise ValueError(f"Unsupported VJEPA2 model size: {model_size}.")
        # 假设使用256分辨率和64帧的预训练模型
        return f"facebook/vjepa2-{size_map[model_size]}-fpc64-256"

    def forward(self, x, timesteps, context, **kwargs):
        B, C_in, T_in, H_in, W_in = x.shape

        # 1. 将输入潜在变量x投影为patch嵌入
        x_embed = self.input_proj(x)
        B, D, T_p, H_p, W_p = x_embed.shape
        x_embed = x_embed.flatten(2).transpose(1, 2)  # [B, T_p*H_p*W_p, D]
        
        # 2. 添加位置嵌入
        # VJEPA2的位置嵌入可能与我们的patch数量不匹配，需要检查
        if x_embed.shape[1] != self.pos_embed.shape[1]:
             # 如果patch数量不匹配，可能需要插值或截断位置嵌入
             # 这里我们简单地重复或截断以匹配
            num_patches = x_embed.shape[1]
            if num_patches > self.pos_embed.shape[1]:
                pos_embed = torch.cat(
                    [self.pos_embed] * (num_patches // self.pos_embed.shape[1]) + [self.pos_embed[:, :num_patches % self.pos_embed.shape[1]]], 
                    dim=1
                )
            else:
                pos_embed = self.pos_embed[:, :num_patches]
        else:
            pos_embed = self.pos_embed

        x_embed = x_embed + pos_embed
        
        # 3. 准备并添加时间步嵌入
        t_embed = self.time_embed(timestep_embedding(timesteps, self.model_channels)) # [B, D]
        t_embed = t_embed.unsqueeze(1) # [B, 1, D]
        x_embed = x_embed + t_embed

        # 4. 准备并添加条件上下文嵌入
        c_crossattn = context['c_crossattn'][0]
        c_embed = self.context_proj(c_crossattn)  # [B, N_ctx, D]

        # 5. 拼接序列并通过VJEPA2
        full_seq = torch.cat([c_embed, x_embed], dim=1)
        
        vjepa_outputs = self.vjepa(
            inputs_embeds=full_seq,
            output_hidden_states=True
        )
        
        all_hidden_states = vjepa_outputs.hidden_states # 元组，包含(embedding_output, layer_1_output, ...)
        
        # 分离内容和条件的token
        content_hidden_states = [
            h[:, c_embed.shape[1]:, :] for h in all_hidden_states
        ]

        # 6. 生成 feat_list
        feat_list = []
        for i, proj_layer in enumerate(self.feat_list_projectors):
            vjepa_layer_idx = self.vjepa_layer_indices[i]
            # +1 是因为 all_hidden_states 的第一个元素是初始嵌入
            hidden_state = content_hidden_states[vjepa_layer_idx + 1] # [B, N_content, D]

            # 投影到期望的通道维度
            projected_feat = proj_layer(hidden_state)  # [B, N_content, C_feat]
            C_feat = projected_feat.shape[-1]
            
            # 重塑为5D张量: B x T x C x H x W
            # 首先恢复通道维度: [B, C_feat, N_content]
            projected_feat_reshaped = projected_feat.transpose(1, 2)
            # 然后恢复空间/时间维度: [B, C_feat, T_p, H_p, W_p]
            projected_feat_reshaped = projected_feat_reshaped.view(B, C_feat, T_p, H_p, W_p)
            # 最后permute到v84期望的格式: [B, T_p, C_feat, H_p, W_p]
            # 注意：这里的维度顺序是 T, C, H, W
            final_feat = projected_feat_reshaped.permute(0, 2, 1, 3, 4)
            feat_list.append(final_feat)

        # 7. 生成 model_output
        # 使用最后一层的特征来预测噪声
        last_hidden_state = content_hidden_states[-1] # [B, N_content, D]
        # 恢复空间/时间维度
        last_hidden_state_reshaped = last_hidden_state.transpose(1, 2).view(B, D, T_p, H_p, W_p)
        
        # 上采样并投影到输出通道
        model_output = self.out_proj(last_hidden_state_reshaped)

        # 检查并修正输出尺寸，以防上采样产生偏差
        if model_output.shape != x.shape:
            # 使用插值来精确匹配尺寸
            model_output = nn.functional.interpolate(model_output, size=x.shape[2:], mode='trilinear', align_corners=False)

        return model_output, feat_list


class VJEPA2_Direct_Wrapper(nn.Module):
    """
    一个直接处理原始视频的VJEPA2包装器。
    它接收原始视频和条件，并输出用于物理损失计算的特征列表。
    这个版本绕过了VAE和扩散过程。此版本输出统一维度的特征，以匹配
    LatentVisualDiffusionv84_VJEPA的需求。

    输入:
        - x (torch.Tensor): 原始视频帧, [B, C, T, H, W] (C=3)
        - context (dict): 条件字典

    输出:
        - tuple[torch.Tensor, list[torch.Tensor]]:
            - model_output (None): 占位符，与UNet接口兼容。
            - feat_list (list[torch.Tensor]): 从VJEPA2中间层提取的特征列表，
                                           所有特征具有统一的维度。
    """
    def __init__(
        self,
        context_dim: int,
        vjepa_model_size: str = 'giant',
        vjepa_checkpoint_path: str = None,
        num_output_features: int = 16,
        temporal_length: int = 16,
        image_cross_attention: bool = True,
        **kwargs, # Accept and ignore unused arguments from the config
    ):
        super().__init__()
        self.context_dim = context_dim
        self.num_output_features = num_output_features
        self.temporal_length = temporal_length
        self.image_cross_attention = image_cross_attention

        # 1. 加载 VJEPA2 主干网络和其配套的预处理器
        hf_repo = self._get_hf_repo(vjepa_model_size)
        print(f"Loading VJEPA2 model for direct video processing from: {hf_repo}")
        try:
            model_path = vjepa_checkpoint_path if vjepa_checkpoint_path else hf_repo
            self.vjepa = AutoModel.from_pretrained(model_path, output_hidden_states=True)
            self.processor = AutoVideoProcessor.from_pretrained(model_path)
        except Exception as e:
            raise IOError(f"Failed to load VJEPA2 model or processor: {e}")

        self.vjepa_hidden_dim = self.vjepa.config.hidden_size
        
        # 冻结VJEPA2的所有参数
        for param in self.vjepa.parameters():
            param.requires_grad = False
        
        # 2. 条件上下文投影模块 (如果需要的话，但在这个直接模型中可能不使用)
        self.context_proj = nn.Linear(context_dim, self.vjepa_hidden_dim)

        # 3. 选择要输出的VJEPA层
        # 我们不再将特征投影到不同的维度。
        # 相反，我们直接从VJEPA中选择N个层并输出它们的hidden states。
        num_vjepa_layers = len(self.vjepa.encoder.layer)
        self.vjepa_layer_indices = torch.linspace(
            0, num_vjepa_layers - 1, self.num_output_features
        ).long().tolist()

        print(f"VJEPA2 Wrapper will output {self.num_output_features} features.")
        print(f"Selected VJEPA layer indices: {self.vjepa_layer_indices}")

        # `feat_list_projectors` 已被移除，因为我们输出统一维度的特征。
        self.feat_list_projectors = None


    def _get_hf_repo(self, model_size: str) -> str:
        size_map = {'large': 'vitl', 'huge': 'vith', 'giant': 'vitg'}
        if model_size not in size_map:
            raise ValueError(f"Unsupported VJEPA2 model size: {model_size}.")
        return f"facebook/vjepa2-{size_map[model_size]}-fpc64-256"

    def forward(self, x, t=None, timesteps=None, cond=None, context=None, **kwargs):
        # timesteps 在这个直接处理模型中被忽略。
        # 为了兼容性，接受多种参数名
        final_context = cond if cond is not None else context

        # 输入 x 的形状是 [B, C, T, H, W]
        # AutoVideoProcessor 期望的是一个 [B, T, C, H, W] 的列表
        # 所以我们需要先调整维度并转换为列表
        B, C, T_in, H, W = x.shape
        x_permuted = x.permute(0, 2, 1, 3, 4) # -> [B, T, C, H, W]
        video_list = [frame for frame in x_permuted]
        
        # 1. 使用官方的processor处理视频
        inputs = self.processor(video_list, return_tensors="pt").to(x.device)

        # 2. 直接用VJEPA2处理视频
        vjepa_outputs = self.vjepa(
            **inputs,
            output_hidden_states=True
        )
        
        # all_hidden_states[0] 是初始的patch嵌入
        # all_hidden_states[1:] 是每个Transformer层的输出
        all_hidden_states = vjepa_outputs.hidden_states[1:] 

        # 2. 从中间层生成feat_list
        feat_list = []
        
        # 终极解决方案：结合processor的输出和模型hidden_state来反解出所有维度
        # 1. 从处理后的视频中获取真实的 H 和 W
        processed_video = inputs['pixel_values_videos']
        _, _, _, H, W = processed_video.shape

        # 2. 从config中获取可靠的patch_size来计算空间patch数
        cfg = self.vjepa.config
        patch_size = cfg.patch_size
        H_p = H // patch_size
        W_p = W // patch_size
        
        # 3. 从模型输出中获取真实的patch总数
        temp_hidden_state = all_hidden_states[0]
        B, N_patches, D = temp_hidden_state.shape

        # 4. 反解出时间patch数
        assert N_patches % (H_p * W_p) == 0, f"Total patches {N_patches} not divisible by spatial patches {H_p}x{W_p}"
        T_p = N_patches // (H_p * W_p)

        for vjepa_layer_idx in self.vjepa_layer_indices:
            hidden_state = all_hidden_states[vjepa_layer_idx] # [B, N_patches, D]
            hidden_state_final = all_hidden_states[-1] # [B, N_patches, D]


            
            # 无需投影。直接使用hidden_state。
            # C_feat现在是VJEPA的隐藏维度。
            # No projection. Use the hidden_state directly.
            # C_feat is now the VJEPA hidden dimension.
            C_feat = self.vjepa_hidden_dim
            
            # 重塑为5D张量: B x T x C x H x W
            # Reshape to 5D tensor: B x T x C x H x W
            feat_reshaped = hidden_state.transpose(1, 2)
            feat_reshaped = feat_reshaped.view(B, C_feat, T_p, H_p, W_p)
            final_feat = feat_reshaped.permute(0, 2, 1, 3, 4)
            feat_list.append(final_feat)
        
        # 3. model_output 不是必需的，返回None作为占位符，final_feat作为model_output
        model_output = hidden_state_final   

        return model_output, feat_list