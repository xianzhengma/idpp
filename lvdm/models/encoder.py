import torch
import torch.nn as nn
import open_clip
from transformers import AutoImageProcessor, AutoModel
from typing import Tuple, Optional
import torch.nn.functional as F
import math
from models.vision_transformer import VisionTransformer, vit_base, vit_large, vit_huge


class BaseEncoder(nn.Module):
    """Base encoder class"""
    def __init__(self):
        super().__init__()
        self.downsample_factor = 1  # 默认下采样倍数
        
    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def _apply_mask_in_original_resolution(
        self, 
        features: torch.Tensor,
        mask: torch.Tensor,
        target_size: Tuple[int, int],
        precise_mask: bool = False
    ) -> torch.Tensor:
        """
        在原始分辨率上应用mask，然后再调整到目标尺寸
        
        Args:
            features (torch.Tensor): [B, T, H1, W1, C] 特征图
            mask (torch.Tensor): [B, T, 1, H2, W2] 原始mask
            target_size (Tuple[int, int]): 目标尺寸 (H, W)
            precise_mask (bool): 是否在原始分辨率上应用mask
            
        Returns:
            torch.Tensor: [B, T, target_H, target_W, C] 处理后的特征图
        """
        B, T = features.shape[:2]
        
        if precise_mask:
            # 获取原始mask的尺寸
            mask_h, mask_w = mask.shape[-2:]
            # print(f"precise_mask: {precise_mask}")
            # print(f"特征图尺寸: {features.shape}")
            # print(f"Mask尺寸: {mask.shape}")
            # print(f"目标尺寸: {target_size}")
            
            # 计算合适的中间尺寸，避免尺寸过大
            max_size = 256  # 降低最大尺寸限制
            scale_factor = min(max_size / mask_h, max_size / mask_w)
            if scale_factor < 1:
                intermediate_h = int(mask_h * scale_factor)
                intermediate_w = int(mask_w * scale_factor)
                # print(f"使用中间尺寸: ({intermediate_h}, {intermediate_w})")
            else:
                intermediate_h, intermediate_w = mask_h, mask_w
                print(f"直接使用原始尺寸: ({intermediate_h}, {intermediate_w})")
            
            # 分批处理以避免显存溢出
            batch_size = 1  # 每批处理1个样本
            processed_features = []
            
            for b_start in range(0, B, batch_size):
                b_end = min(b_start + batch_size, B)
                curr_batch_size = b_end - b_start
                
                # 处理当前批次的特征
                curr_features = features[b_start:b_end]  # [curr_B, T, H1, W1, C]
                curr_mask = mask[b_start:b_end]  # [curr_B, T, 1, H2, W2]
                
                # 将特征图上采样到中间尺寸（分通道处理）
                curr_features = curr_features.permute(0, 1, 4, 2, 3)  # [curr_B, T, C, H1, W1]
                C = curr_features.shape[2]
                channel_batch_size = 64  # 每次处理64个通道
                processed_channels = []
                
                for c_start in range(0, C, channel_batch_size):
                    c_end = min(c_start + channel_batch_size, C)
                    curr_channels = curr_features[:, :, c_start:c_end]  # [curr_B, T, curr_C, H1, W1]
                    
                    # 上采样当前通道组
                    curr_channels = curr_channels.reshape(curr_batch_size * T, -1, *curr_channels.shape[-2:])
                    curr_channels = F.interpolate(
                        curr_channels,
                        size=(intermediate_h, intermediate_w),
                        mode='bilinear',
                        align_corners=False
                    )
                    processed_channels.append(curr_channels)
                    
                    # 清理显存
                    torch.cuda.empty_cache()
                
                # 合并所有通道
                curr_features = torch.cat(processed_channels, dim=1)  # [curr_B*T, C, H, W]
                curr_features = curr_features.view(curr_batch_size, T, C, intermediate_h, intermediate_w)
                
                # 将mask下采样到中间尺寸
                curr_mask = F.interpolate(
                    curr_mask.view(curr_batch_size * T, 1, *curr_mask.shape[-2:]),
                    size=(intermediate_h, intermediate_w),
                    mode='nearest'
                )
                curr_mask = curr_mask.view(curr_batch_size, T, 1, intermediate_h, intermediate_w)
                
                # 应用mask
                curr_features = curr_features.permute(0, 1, 3, 4, 2)  # [curr_B, T, H, W, C]
                curr_mask = curr_mask.permute(0, 1, 3, 4, 2)  # [curr_B, T, H, W, 1]
                curr_features = curr_features * curr_mask
                
                # 调整到目标尺寸（分通道处理）
                curr_features = curr_features.permute(0, 1, 4, 2, 3)  # [curr_B, T, C, H, W]
                processed_channels = []
                
                for c_start in range(0, C, channel_batch_size):
                    c_end = min(c_start + channel_batch_size, C)
                    curr_channels = curr_features[:, :, c_start:c_end]  # [curr_B, T, curr_C, H, W]
                    
                    # 下采样当前通道组
                    curr_channels = curr_channels.reshape(curr_batch_size * T, -1, intermediate_h, intermediate_w)
                    curr_channels = F.interpolate(
                        curr_channels,
                        size=target_size,
                        mode='bilinear',
                        align_corners=False
                    )
                    processed_channels.append(curr_channels)
                    
                    # 清理显存
                    torch.cuda.empty_cache()
                
                # 合并所有通道
                curr_features = torch.cat(processed_channels, dim=1)  # [curr_B*T, C, H, W]
                curr_features = curr_features.view(curr_batch_size, T, C, *target_size)
                curr_features = curr_features.permute(0, 1, 3, 4, 2)  # [curr_B, T, H, W, C]
                
                processed_features.append(curr_features)
                
                # 清理显存
                torch.cuda.empty_cache()
            
            # 合并所有处理后的特征
            features = torch.cat(processed_features, dim=0)
            
        else:
            # 原始处理方式：直接将mask调整到目标尺寸
            mask = F.interpolate(
                mask.reshape(B * T, 1, *mask.shape[-2:]),
                size=target_size,
                mode='nearest'
            )
            mask = mask.permute(0, 2, 3, 1).view(B, T, *target_size, 1)
            features = features * mask
            
        return features


class BasicEncoder(nn.Module):
    """
    Basic encoder module for RAFT
    
    Args:
        output_dim (int): Output feature dimension
        norm_fn (str): Normalization method, 'instance' or 'batch'
        dropout (float): Dropout rate
    """
    def __init__(self, output_dim=128, norm_fn='instance', dropout=0.0):
        super().__init__()
        self.norm_fn = norm_fn
        
        if self.norm_fn == 'group':
            self.norm1 = nn.GroupNorm(8, 64)
            
        elif self.norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(64)
            
        elif self.norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(64)
            
        elif self.norm_fn == 'none':
            self.norm1 = nn.Sequential()
            
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.relu1 = nn.ReLU(inplace=True)
        
        self.in_planes = 64
        self.layer1 = self._make_layer(64,  stride=1)
        self.layer2 = self._make_layer(96, stride=2)
        self.layer3 = self._make_layer(128, stride=2)
        
        # output convolution
        self.conv2 = nn.Conv2d(128, output_dim, kernel_size=1)
        
        self.dropout = None
        if dropout > 0:
            self.dropout = nn.Dropout2d(p=dropout)
            
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                    
    def _make_layer(self, dim, stride=1):
        layer1 = ResidualBlock(self.in_planes, dim, self.norm_fn, stride=stride)
        layer2 = ResidualBlock(dim, dim, self.norm_fn, stride=1)
        layers = (layer1, layer2)
        
        self.in_planes = dim
        return nn.Sequential(*layers)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        x = self.conv2(x)
        
        if self.dropout is not None:
            x = self.dropout(x)
            
        return x


class ResidualBlock(nn.Module):
    """RAFT中的残差块"""
    def __init__(self, in_planes, planes, norm_fn='group', stride=1):
        super().__init__()
  
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        
        num_groups = planes // 8
        
        if norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            if not stride == 1:
                self.norm3 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
        
        elif norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(planes)
            self.norm2 = nn.BatchNorm2d(planes)
            if not stride == 1:
                self.norm3 = nn.BatchNorm2d(planes)
        
        elif norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(planes)
            self.norm2 = nn.InstanceNorm2d(planes)
            if not stride == 1:
                self.norm3 = nn.InstanceNorm2d(planes)
        
        elif norm_fn == 'none':
            self.norm1 = nn.Sequential()
            self.norm2 = nn.Sequential()
            if not stride == 1:
                self.norm3 = nn.Sequential()
                
        if stride == 1:
            self.downsample = None
        else:    
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride), self.norm3)
            
    def forward(self, x):
        y = x
        y = self.conv1(y)
        y = self.norm1(y)
        y = self.relu(y)
        y = self.conv2(y)
        y = self.norm2(y)
        y = self.relu(y)
        
        if self.downsample is not None:
            x = self.downsample(x)
            
        return self.relu(x+y)


class RAFTEncoder(BaseEncoder):
    """
    Using RAFT architecture as feature extractor
    
    Args:
        output_dim (int): Output feature dimension
        dropout (float): Dropout rate
        precise_mask (bool): 是否在原始分辨率上应用mask
    """
    def __init__(self, output_dim: int = 128, dropout: float = 0.0, precise_mask: bool = False):
        super().__init__()
        self.encoder = BasicEncoder(output_dim=output_dim, dropout=dropout)
        self.precise_mask = precise_mask
        self.downsample_factor = 8  # RAFT通过3次stride=2的下采样，总下采样倍数为8
        
    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x (torch.Tensor): [B, T, C, H, W]
            mask (torch.Tensor): [B, T, 1, H, W]
        Returns:
            torch.Tensor: [B, T, H//8, W//8, feature_dim]
        """
        B, T = x.shape[:2]
        x = x.view(B * T, *x.shape[2:])  # [B*T, C, H, W]
        
        # Extract features
        features = self.encoder(x)  # [B*T, output_dim, H//8, W//8]
        
        # Reshape to [B, T, H, W, C]
        features = features.permute(0, 2, 3, 1)  # [B*T, H//8, W//8, output_dim]
        features = features.view(B, T, *features.shape[1:])  # [B, T, H//8, W//8, output_dim]
        
        # 获取目标尺寸
        target_h, target_w = features.shape[2:4]
        
        # 应用mask
        features = self._apply_mask_in_original_resolution(
            features, mask, (target_h, target_w), self.precise_mask
        )
        
        return features


class OpenCLIPEncoder(BaseEncoder):
    """
    使用 OpenCLIP 提取视觉特征
    
    Args:
        model_name (str): CLIP模型名称，如 'ViT-B-32'
        pretrained (str): 预训练模型名称，如 'laion2b_s34b_b79k'
        precise_mask (bool): 是否在原始分辨率上应用mask
    """
    def __init__(
        self, 
        model_name: str = 'ViT-B-32', 
        pretrained: str = 'laion2b_s34b_b79k',
        precise_mask: bool = False
    ):
        super().__init__()
        import open_clip
        
        self.model = open_clip.create_model(
            model_name,
            pretrained=pretrained
        )
        
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
            
        self.feature_dim = 768  # ViT-B-32 的特征维度
        self.precise_mask = precise_mask
        self.downsample_factor = 32  # OpenCLIP从224x224下采样到7x7，倍数为32
            
    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): [B, T, C, H, W]
            mask (torch.Tensor): [B, T, 1, H, W]
        Returns:
            torch.Tensor: [B, T, 7, 7, feature_dim]
        """
        B, T = x.shape[:2]
        x = x.view(B * T, *x.shape[2:])
        
        # 调整图像尺寸为模型预期的尺寸 (224x224)
        x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        
        # 获取 patch tokens
        with torch.no_grad():
            x = self.model.visual.conv1(x)
            x = x.reshape(x.shape[0], x.shape[1], -1)
            x = x.permute(0, 2, 1)
            x = self.model.visual.ln_pre(x)
            x = x + self.model.visual.positional_embedding[1:]
            x = self.model.visual.transformer(x)
            features = self.model.visual.ln_post(x)
        
        # 重塑为网格形式
        grid_size = 7  # 224/32 = 7 for ViT-B/32
        features = features.view(B, T, grid_size, grid_size, -1)
        
        # 应用mask
        features = self._apply_mask_in_original_resolution(
            features, mask, (grid_size, grid_size), self.precise_mask
        )
        
        return features


class DINOv2Encoder(BaseEncoder):
    """
    Using DINOv2 as feature extractor
    
    Args:
        model_name (str): DINOv2 model name, default is facebook/dinov2-base
        precise_mask (bool): 是否在原始分辨率上应用mask
    """
    def __init__(self, model_name: str = "facebook/dinov2-base", precise_mask: bool = False):
        super().__init__()
        try:
            from transformers import AutoImageProcessor, Dinov2Model
            
            self.processor = AutoImageProcessor.from_pretrained(
                model_name,
                size=518  # DINOv2的原始训练尺寸
            )
            self.model = Dinov2Model.from_pretrained(model_name)
            self.precise_mask = precise_mask
            self.downsample_factor = 10  # DINOv2从518x518下采样到54x54，倍数约为9.6，取整为10
            
        except Exception as e:
            print(f"Error loading model: {e}")
            print(f"Attempted model name: {model_name}")
            raise
        
    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): [B, T, C, H, W]
            mask (torch.Tensor): [B, T, 1, H, W]
        Returns:
            torch.Tensor: [B, T, 54, 54, feature_dim]
        """
        B, T = x.shape[:2]
        x = x.view(B * T, *x.shape[2:])
        
        # Get features
        outputs = self.model(x)
        features = outputs.last_hidden_state[:, 1:, :]  # Remove CLS token [B*T, N_patches, feature_dim]
        
        # Calculate grid size
        grid_size = 54  # 实际的patch grid size
        features = features[:, :grid_size*grid_size, :]  # 确保patch数量正确
        features = features.view(B, T, grid_size, grid_size, -1)  # [B, T, H, W, feature_dim]
        
        # 应用mask
        features = self._apply_mask_in_original_resolution(
            features, mask, (grid_size, grid_size), self.precise_mask
        )
        
        return features


class JEPAViTEncoder(BaseEncoder):
    """
    使用JEPA预训练的Vision Transformer作为编码器
    
    Args:
        model_size (str): 模型大小，支持'base'、'large'和'huge'
        checkpoint_path (str): 预训练权重文件路径
        precise_mask (bool): 是否在原始分辨率上应用mask
    """
    
    def __init__(
        self, 
        model_size: str = 'large', 
        checkpoint_path: str = '/work/guanqi/jepa-intuitive-physics/checkpoint/vitl16.pth.tar',
        precise_mask: bool = False
    ):
        super().__init__()
        
        # 根据模型大小选择对应的ViT模型，使用视频版本（带有时间维度的3D ViT）
        if model_size == 'base':
            self.model = vit_base(patch_size=16, num_frames=16, tubelet_size=2)
            self.feature_dim = 768
        elif model_size == 'large':
            self.model = vit_large(patch_size=16, num_frames=16, tubelet_size=2, precise_mask=precise_mask)
            self.feature_dim = 1024
        elif model_size == 'huge':
            self.model = vit_huge(patch_size=16, num_frames=16, tubelet_size=2)
            self.feature_dim = 1280
        else:
            raise ValueError(f"不支持的模型大小: {model_size}")
        
        # 加载预训练权重
        if checkpoint_path:
            self._load_checkpoint(checkpoint_path)
        
        # 设置下采样因子（ViT的patch_size决定了下采样倍数）
        self.downsample_factor = 16  # 对应patch_size=16
        self.precise_mask = precise_mask
        
    def _load_checkpoint(self, checkpoint_path: str):
        """
        加载预训练权重，严格按照eval.py中的实现方式
        
        Args:
            checkpoint_path (str): 预训练权重文件路径
        """
        print(f'加载预训练权重: {checkpoint_path}')
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            # 处理不同格式的checkpoint
            if 'model' in checkpoint:
                state_dict = checkpoint['model']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'encoder' in checkpoint:
                state_dict = checkpoint['encoder']
            else:
                state_dict = checkpoint
            
            # 移除可能的'module.'前缀（分布式训练产生的）
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            
            # 处理backbone前缀
            new_state_dict = {}
            for k, v in state_dict.items():
                # 如果键名以backbone.开头，则移除前缀
                if k.startswith('backbone.'):
                    new_key = k[9:]  # 移除'backbone.'前缀
                    new_state_dict[new_key] = v
                else:
                    new_state_dict[k] = v
            
            # 使用处理后的state_dict
            state_dict = new_state_dict
            
            # 检查每个键，处理形状不匹配的情况
            model_state_dict = self.model.state_dict()
            for k, v in model_state_dict.items():
                if k not in state_dict:
                    print(f'键 "{k}" 在加载的权重中未找到')
                elif state_dict[k].shape != v.shape:
                    print(f'键 "{k}" 在模型和加载的权重中形状不同')
                    print(f'  模型形状: {v.shape}, 权重形状: {state_dict[k].shape}')
                    # 使用模型中的初始值
                    state_dict[k] = v
            
            # 使用strict=False允许部分加载
            msg = self.model.load_state_dict(state_dict, strict=False)
            print(f"加载预训练权重结果: {msg}")
            
            # 打印缺失和意外的键
            if msg.missing_keys:
                print(f"缺失的键数量: {len(msg.missing_keys)}")
                if len(msg.missing_keys) < 10:
                    print(f"缺失的键: {msg.missing_keys}")
                else:
                    print(f"部分缺失的键: {msg.missing_keys[:10]}...")
            
            if msg.unexpected_keys:
                print(f"意外的键数量: {len(msg.unexpected_keys)}")
                if len(msg.unexpected_keys) < 10:
                    print(f"意外的键: {msg.unexpected_keys}")
                else:
                    print(f"部分意外的键: {msg.unexpected_keys[:10]}...")
                
        except Exception as e:
            print(f"加载预训练权重失败: {e}")
            import traceback
            traceback.print_exc()
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        前向传播，直接使用vision_transformer.py中的forward方法
        
        Args:
            x (torch.Tensor): [B, T, 3, H, W] 输入视频帧
            mask (torch.Tensor): [B, T, 1, H, W] 掩码
            
        Returns:
            torch.Tensor: [B, T//2, H//16, W//16, C] 特征图
        """
        B, T, C, H, W = x.shape

        if self.precise_mask:
            x_permuted = x.permute(0, 1, 3, 4, 2)  # [B, T, H, W, 3]
            mask_permuted = mask.permute(0, 1, 3, 4, 2)  # [B, T, H, W, 1]
            masked_x = x_permuted * mask_permuted  # 广播机制会自动扩展mask
            # 如果需要，再转回原来的维度顺序
            masked_x = masked_x.permute(0, 1, 4, 2, 3)  # [B, T, 3, H, W]
            x = masked_x
        
        # 确保输入帧数是tubelet_size的倍数
        tubelet_size = self.model.tubelet_size
        num_frames = self.model.num_frames

        feat_t = T // self.model.tubelet_size
        feat_h = H // self.downsample_factor
        feat_w = W // self.downsample_factor

        x = x.permute(0, 2, 1, 3, 4)
        # mask = mask.permute(0, 2, 1, 3, 4)
        # print(f"x shape before jepa vit: {x.shape}")

        x = self.model(x, mask, precise_mask=self.precise_mask)

        # print(f"x shape after jepa vit: {x.shape}") # B,N,D

        x = x.reshape(B, feat_t, feat_h, feat_w, -1) # B,T//2,H//16,W//16,C
        # mask (torch.Tensor): [B, T, 1, H, W] 掩码
        # 调整mask的时间维度以匹配特征图
        # 方法1：取每个tubelet的第一帧对应的mask
        adjusted_mask = mask[:, ::tubelet_size, :, :, :]
        # 确保时间维度匹配
        adjusted_mask = adjusted_mask[:, :feat_t, :, :, :]
        
        # 应用调整后的mask
        if not self.precise_mask:
            x = self._apply_mask_in_original_resolution(
                x, adjusted_mask, (feat_h, feat_w), self.precise_mask
            )
        return x


class VJEPA2Encoder(BaseEncoder):
    """
    使用VJEPA2预训练的Vision Transformer作为编码器
    通过HuggingFace transformers库加载
    
    Args:
        model_size (str): 模型大小，支持'large'、'huge'和'giant'
        resolution (int): 输入分辨率，支持256和384（仅giant模型）
        checkpoint_path (str): 本地检查点路径，如果为None则从HuggingFace下载
    """
    
    def __init__(
        self, 
        model_size: str = 'giant', 
        resolution: int = 256,
        checkpoint_path: str = None
    ):
        super().__init__()
        
        # 构建HuggingFace仓库名称 - 修正映射
        size_map = {
            'large': 'vitl',
            'huge': 'vith',  # 对应vith模型
            'giant': 'vitg'
        }
        
        if model_size not in size_map:
            raise ValueError(f"不支持的VJEPA2模型大小: {model_size}. 支持: {list(size_map.keys())}")
            
        # 检查分辨率支持
        if resolution == 384 and model_size != 'giant':
            raise ValueError(f"384分辨率仅支持giant模型，当前模型: {model_size}")
        if resolution not in [256, 384]:
            raise ValueError(f"不支持的分辨率: {resolution}. 支持: [256, 384]")
            
        # 构建仓库名称
        hf_repo = f"facebook/vjepa2-{size_map[model_size]}-fpc64-{resolution}"
        print(f"加载VJEPA2模型: {hf_repo}")
        
        try:
            # 使用AutoVideoProcessor替代AutoImageProcessor
            from transformers import AutoModel, AutoVideoProcessor
            
            if checkpoint_path:
                # 从本地路径加载
                print(f"从本地路径加载: {checkpoint_path}")
                self.model = AutoModel.from_pretrained(checkpoint_path)
                self.processor = AutoVideoProcessor.from_pretrained(checkpoint_path)
            else:
                # 从HuggingFace下载
                print(f"从HuggingFace下载: {hf_repo}")
                self.model = AutoModel.from_pretrained(hf_repo)
                self.processor = AutoVideoProcessor.from_pretrained(hf_repo)
                
            print(f"成功加载VJEPA2模型和处理器")
                
        except Exception as e:
            print(f"加载VJEPA2模型失败: {e}")
            raise
        
        # 设置特征维度
        feature_dims = {
            'large': 1024,
            'huge': 1280,
            'giant': 1408
        }
        self.feature_dim = feature_dims[model_size]
        
        # 设置模型参数
        self.model_size = model_size
        self.resolution = resolution
        
        # 推断下采样因子和tubelet_size（基于VJEPA2的标准配置）
        self.downsample_factor = 16  # ViT标准patch_size
        self.tubelet_size = 2  # VJEPA2标准tubelet_size
        
        print(f"VJEPA2 {model_size} 特征维度: {self.feature_dim}")
        
    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x (torch.Tensor): [B, T, 3, H, W] 输入视频帧
            mask (torch.Tensor): [B, T, 1, H, W] 掩码（VJEPA2可能不使用，但保持接口一致）
            
        Returns:
            torch.Tensor: [B, T', H', W', C] 特征图
        """
        B, T, C, H, W = x.shape
        original_T = T  # 保存原始时间维度
        
        try:
            # 先尝试使用原始帧数，看看模型是否支持
            
            # 直接使用原始帧数试试
            
            with torch.no_grad():
                # 直接传入模型，使用正确的参数名
                outputs = self.model(pixel_values_videos=x)
                
        except Exception as e:
            # 如果原始帧数失败，再尝试64帧
            target_frames = 64  # VJEPA2的标准帧数
            
            if T != target_frames:
                # 如果帧数不匹配，进行时间维度的插值或采样
                if T > target_frames:
                    # 如果帧数太多，进行下采样
                    indices = torch.linspace(0, T-1, target_frames).long()
                    x = x[:, indices, :, :, :]
                else:
                    # 如果帧数太少，进行重复
                    # 计算需要重复多少次
                    repeat_factor = target_frames // T
                    remainder = target_frames % T
                    
                    # 重复整个序列
                    x_repeated = x.repeat(1, repeat_factor, 1, 1, 1)
                    
                    # 如果还有余数，添加部分帧
                    if remainder > 0:
                        x_remainder = x[:, :remainder, :, :, :]
                        x = torch.cat([x_repeated, x_remainder], dim=1)
                    else:
                        x = x_repeated
                
                T = target_frames
            
            with torch.no_grad():
                # 重新尝试
                outputs = self.model(pixel_values_videos=x)
            
        # 提取特征
        if hasattr(outputs, 'last_hidden_state'):
            features = outputs.last_hidden_state  # [B, N, D]
        elif hasattr(outputs, 'pooler_output'):
            features = outputs.pooler_output
        elif isinstance(outputs, torch.Tensor):
            features = outputs
        else:
            # 如果输出是字典，尝试获取主要输出
            if hasattr(outputs, 'keys'):
                # 尝试常见的输出键
                for key in ['last_hidden_state', 'hidden_states', 'logits', 'prediction_scores']:
                    if hasattr(outputs, key):
                        features = getattr(outputs, key)
                        if isinstance(features, (list, tuple)):
                            features = features[-1]  # 取最后一层
                        break
                else:
                    raise ValueError(f"无法从模型输出中提取特征，输出类型: {type(outputs)}")
            else:
                raise ValueError(f"未知的模型输出格式: {type(outputs)}")
                
        # 重塑特征到期望的输出格式 [B, T', H', W', C]
        # 使用原始的T值来计算特征维度
        feat_t = original_T // self.tubelet_size
        feat_h = H // self.downsample_factor
        feat_w = W // self.downsample_factor
        
        # 如果features是[B, N, D]格式，重塑为[B, T', H', W', D]
        if len(features.shape) == 3:
            expected_tokens = feat_t * feat_h * feat_w
            if features.shape[1] == expected_tokens:
                features = features.reshape(B, feat_t, feat_h, feat_w, -1)
            else:
                # 尝试适配：如果token数量不匹配，可能需要调整
                if features.shape[1] > expected_tokens:
                    # 截取前面的tokens
                    features = features[:, :expected_tokens, :]
                    features = features.reshape(B, feat_t, feat_h, feat_w, -1)
                else:
                    # 如果tokens太少，进行插值或填充
                    # 这里简单地重复最后的特征
                    needed_tokens = expected_tokens - features.shape[1]
                    last_token = features[:, -1:, :].repeat(1, needed_tokens, 1)
                    features = torch.cat([features, last_token], dim=1)
                    features = features.reshape(B, feat_t, feat_h, feat_w, -1)
        elif len(features.shape) == 4:
            # 如果已经是4D，添加时间维度
            features = features.unsqueeze(1).repeat(1, feat_t, 1, 1, 1)
        
        return features


def get_encoder(encoder_type: str, **kwargs) -> BaseEncoder:
    """
    Get specified type of encoder
    
    Args:
        encoder_type (str): Encoder type, supports 'openclip', 'dinov2', 'raft' and 'jepavit'
        **kwargs: Additional parameters for encoder
        
    Returns:
        BaseEncoder: Encoder instance
    """
    encoder_map = {
        'openclip': OpenCLIPEncoder,
        'dinov2': DINOv2Encoder,
        'raft': RAFTEncoder,
        'jepavit': JEPAViTEncoder,  # 添加新的编码器类型
        'vjepa2': VJEPA2Encoder,  # 添加新的编码器类型
    }
    
    if encoder_type not in encoder_map:
        raise ValueError(f"Unsupported encoder type: {encoder_type}")
        
    return encoder_map[encoder_type](**kwargs) 