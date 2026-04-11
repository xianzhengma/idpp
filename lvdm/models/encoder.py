import torch
import torch.nn as nn
import open_clip
from transformers import AutoImageProcessor, AutoModel
from typing import Tuple, Optional
import torch.nn.functional as F
import math
from models.vision_transformer import VisionTransformer, vit_base, vit_large, vit_huge

class BaseEncoder(nn.Module):

    def __init__(self):
        super().__init__()
        self.downsample_factor = 1

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def _apply_mask_in_original_resolution(self, features: torch.Tensor, mask: torch.Tensor, target_size: Tuple[int, int], precise_mask: bool=False) -> torch.Tensor:
        B, T = features.shape[:2]
        if precise_mask:
            mask_h, mask_w = mask.shape[-2:]
            max_size = 256
            scale_factor = min(max_size / mask_h, max_size / mask_w)
            if scale_factor < 1:
                intermediate_h = int(mask_h * scale_factor)
                intermediate_w = int(mask_w * scale_factor)
            else:
                intermediate_h, intermediate_w = (mask_h, mask_w)
                print(f'Using original mask size: ({intermediate_h}, {intermediate_w})')
            batch_size = 1
            processed_features = []
            for b_start in range(0, B, batch_size):
                b_end = min(b_start + batch_size, B)
                curr_batch_size = b_end - b_start
                curr_features = features[b_start:b_end]
                curr_mask = mask[b_start:b_end]
                curr_features = curr_features.permute(0, 1, 4, 2, 3)
                C = curr_features.shape[2]
                channel_batch_size = 64
                processed_channels = []
                for c_start in range(0, C, channel_batch_size):
                    c_end = min(c_start + channel_batch_size, C)
                    curr_channels = curr_features[:, :, c_start:c_end]
                    curr_channels = curr_channels.reshape(curr_batch_size * T, -1, *curr_channels.shape[-2:])
                    curr_channels = F.interpolate(curr_channels, size=(intermediate_h, intermediate_w), mode='bilinear', align_corners=False)
                    processed_channels.append(curr_channels)
                    torch.cuda.empty_cache()
                curr_features = torch.cat(processed_channels, dim=1)
                curr_features = curr_features.view(curr_batch_size, T, C, intermediate_h, intermediate_w)
                curr_mask = F.interpolate(curr_mask.view(curr_batch_size * T, 1, *curr_mask.shape[-2:]), size=(intermediate_h, intermediate_w), mode='nearest')
                curr_mask = curr_mask.view(curr_batch_size, T, 1, intermediate_h, intermediate_w)
                curr_features = curr_features.permute(0, 1, 3, 4, 2)
                curr_mask = curr_mask.permute(0, 1, 3, 4, 2)
                curr_features = curr_features * curr_mask
                curr_features = curr_features.permute(0, 1, 4, 2, 3)
                processed_channels = []
                for c_start in range(0, C, channel_batch_size):
                    c_end = min(c_start + channel_batch_size, C)
                    curr_channels = curr_features[:, :, c_start:c_end]
                    curr_channels = curr_channels.reshape(curr_batch_size * T, -1, intermediate_h, intermediate_w)
                    curr_channels = F.interpolate(curr_channels, size=target_size, mode='bilinear', align_corners=False)
                    processed_channels.append(curr_channels)
                    torch.cuda.empty_cache()
                curr_features = torch.cat(processed_channels, dim=1)
                curr_features = curr_features.view(curr_batch_size, T, C, *target_size)
                curr_features = curr_features.permute(0, 1, 3, 4, 2)
                processed_features.append(curr_features)
                torch.cuda.empty_cache()
            features = torch.cat(processed_features, dim=0)
        else:
            mask = F.interpolate(mask.reshape(B * T, 1, *mask.shape[-2:]), size=target_size, mode='nearest')
            mask = mask.permute(0, 2, 3, 1).view(B, T, *target_size, 1)
            features = features * mask
        return features

class BasicEncoder(nn.Module):

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
        self.layer1 = self._make_layer(64, stride=1)
        self.layer2 = self._make_layer(96, stride=2)
        self.layer3 = self._make_layer(128, stride=2)
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
            self.downsample = nn.Sequential(nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride), self.norm3)

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
        return self.relu(x + y)

class RAFTEncoder(BaseEncoder):

    def __init__(self, output_dim: int=128, dropout: float=0.0, precise_mask: bool=False):
        super().__init__()
        self.encoder = BasicEncoder(output_dim=output_dim, dropout=dropout)
        self.precise_mask = precise_mask
        self.downsample_factor = 8

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        B, T = x.shape[:2]
        x = x.view(B * T, *x.shape[2:])
        features = self.encoder(x)
        features = features.permute(0, 2, 3, 1)
        features = features.view(B, T, *features.shape[1:])
        target_h, target_w = features.shape[2:4]
        features = self._apply_mask_in_original_resolution(features, mask, (target_h, target_w), self.precise_mask)
        return features

class OpenCLIPEncoder(BaseEncoder):

    def __init__(self, model_name: str='ViT-B-32', pretrained: str='laion2b_s34b_b79k', precise_mask: bool=False):
        super().__init__()
        import open_clip
        self.model = open_clip.create_model(model_name, pretrained=pretrained)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        self.feature_dim = 768
        self.precise_mask = precise_mask
        self.downsample_factor = 32

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        B, T = x.shape[:2]
        x = x.view(B * T, *x.shape[2:])
        x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        with torch.no_grad():
            x = self.model.visual.conv1(x)
            x = x.reshape(x.shape[0], x.shape[1], -1)
            x = x.permute(0, 2, 1)
            x = self.model.visual.ln_pre(x)
            x = x + self.model.visual.positional_embedding[1:]
            x = self.model.visual.transformer(x)
            features = self.model.visual.ln_post(x)
        grid_size = 7
        features = features.view(B, T, grid_size, grid_size, -1)
        features = self._apply_mask_in_original_resolution(features, mask, (grid_size, grid_size), self.precise_mask)
        return features

class DINOv2Encoder(BaseEncoder):

    def __init__(self, model_name: str='facebook/dinov2-base', precise_mask: bool=False):
        super().__init__()
        try:
            from transformers import AutoImageProcessor, Dinov2Model
            self.processor = AutoImageProcessor.from_pretrained(model_name, size=518)
            self.model = Dinov2Model.from_pretrained(model_name)
            self.precise_mask = precise_mask
            self.downsample_factor = 10
        except Exception as e:
            print(f'Error loading model: {e}')
            print(f'Attempted model name: {model_name}')
            raise

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        B, T = x.shape[:2]
        x = x.view(B * T, *x.shape[2:])
        outputs = self.model(x)
        features = outputs.last_hidden_state[:, 1:, :]
        grid_size = 54
        features = features[:, :grid_size * grid_size, :]
        features = features.view(B, T, grid_size, grid_size, -1)
        features = self._apply_mask_in_original_resolution(features, mask, (grid_size, grid_size), self.precise_mask)
        return features

class JEPAViTEncoder(BaseEncoder):

    def __init__(self, model_size: str='large', checkpoint_path: str='/path/to/jepa-intuitive-physics/checkpoint/vitl16.pth.tar', precise_mask: bool=False):
        super().__init__()
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
            raise ValueError(f'Unsupported model_size: {model_size}')
        if checkpoint_path:
            self._load_checkpoint(checkpoint_path)
        self.downsample_factor = 16
        self.precise_mask = precise_mask

    def _load_checkpoint(self, checkpoint_path: str):
        print(f'Loading pretrained weights: {checkpoint_path}')
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            if 'model' in checkpoint:
                state_dict = checkpoint['model']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'encoder' in checkpoint:
                state_dict = checkpoint['encoder']
            else:
                state_dict = checkpoint
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('backbone.'):
                    new_key = k[9:]
                    new_state_dict[new_key] = v
                else:
                    new_state_dict[k] = v
            state_dict = new_state_dict
            model_state_dict = self.model.state_dict()
            for k, v in model_state_dict.items():
                if k not in state_dict:
                    print(f'Key "{k}" not found in checkpoint')
                elif state_dict[k].shape != v.shape:
                    print(f'Key "{k}" shape mismatch between model and checkpoint')
                    print(f'  model: {v.shape}, checkpoint: {state_dict[k].shape}')
                    state_dict[k] = v
            msg = self.model.load_state_dict(state_dict, strict=False)
            print(f'load_state_dict result: {msg}')
            if msg.missing_keys:
                print(f'missing_keys count: {len(msg.missing_keys)}')
                if len(msg.missing_keys) < 10:
                    print(f'missing_keys: {msg.missing_keys}')
                else:
                    print(f'missing_keys (first 10): {msg.missing_keys[:10]}...')
            if msg.unexpected_keys:
                print(f'unexpected_keys count: {len(msg.unexpected_keys)}')
                if len(msg.unexpected_keys) < 10:
                    print(f'unexpected_keys: {msg.unexpected_keys}')
                else:
                    print(f'unexpected_keys (first 10): {msg.unexpected_keys[:10]}...')
        except Exception as e:
            print(f'Failed to load pretrained weights: {e}')
            import traceback
            traceback.print_exc()

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = x.shape
        if self.precise_mask:
            x_permuted = x.permute(0, 1, 3, 4, 2)
            mask_permuted = mask.permute(0, 1, 3, 4, 2)
            masked_x = x_permuted * mask_permuted
            masked_x = masked_x.permute(0, 1, 4, 2, 3)
            x = masked_x
        tubelet_size = self.model.tubelet_size
        num_frames = self.model.num_frames
        feat_t = T // self.model.tubelet_size
        feat_h = H // self.downsample_factor
        feat_w = W // self.downsample_factor
        x = x.permute(0, 2, 1, 3, 4)
        x = self.model(x, mask, precise_mask=self.precise_mask)
        x = x.reshape(B, feat_t, feat_h, feat_w, -1)
        adjusted_mask = mask[:, ::tubelet_size, :, :, :]
        adjusted_mask = adjusted_mask[:, :feat_t, :, :, :]
        if not self.precise_mask:
            x = self._apply_mask_in_original_resolution(x, adjusted_mask, (feat_h, feat_w), self.precise_mask)
        return x

class VJEPA2Encoder(BaseEncoder):

    def __init__(self, model_size: str='giant', resolution: int=256, checkpoint_path: str=None):
        super().__init__()
        size_map = {'large': 'vitl', 'huge': 'vith', 'giant': 'vitg'}
        if model_size not in size_map:
            raise ValueError(f'Unsupported VJEPA2 model_size {model_size}. Allowed: {list(size_map.keys())}')
        if resolution == 384 and model_size != 'giant':
            raise ValueError(f'Resolution 384 requires model_size giant, got {model_size}')
        if resolution not in [256, 384]:
            raise ValueError(f'Unsupported resolution {resolution}. Allowed: [256, 384]')
        hf_repo = f'facebook/vjepa2-{size_map[model_size]}-fpc64-{resolution}'
        print(f'Loading VJEPA2: {hf_repo}')
        try:
            from transformers import AutoModel, AutoVideoProcessor
            if checkpoint_path:
                print(f'Loading from local path: {checkpoint_path}')
                self.model = AutoModel.from_pretrained(checkpoint_path)
                self.processor = AutoVideoProcessor.from_pretrained(checkpoint_path)
            else:
                print(f'Loading from Hugging Face Hub: {hf_repo}')
                self.model = AutoModel.from_pretrained(hf_repo)
                self.processor = AutoVideoProcessor.from_pretrained(hf_repo)
            print('VJEPA2 model and processor loaded.')
        except Exception as e:
            print(f'Failed to load VJEPA2: {e}')
            raise
        feature_dims = {'large': 1024, 'huge': 1280, 'giant': 1408}
        self.feature_dim = feature_dims[model_size]
        self.model_size = model_size
        self.resolution = resolution
        self.downsample_factor = 16
        self.tubelet_size = 2
        print(f'VJEPA2 {model_size} feature_dim: {self.feature_dim}')

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = x.shape
        original_T = T
        try:
            with torch.no_grad():
                outputs = self.model(pixel_values_videos=x)
        except Exception as e:
            target_frames = 64
            if T != target_frames:
                if T > target_frames:
                    indices = torch.linspace(0, T - 1, target_frames).long()
                    x = x[:, indices, :, :, :]
                else:
                    repeat_factor = target_frames // T
                    remainder = target_frames % T
                    x_repeated = x.repeat(1, repeat_factor, 1, 1, 1)
                    if remainder > 0:
                        x_remainder = x[:, :remainder, :, :, :]
                        x = torch.cat([x_repeated, x_remainder], dim=1)
                    else:
                        x = x_repeated
                T = target_frames
            with torch.no_grad():
                outputs = self.model(pixel_values_videos=x)
        if hasattr(outputs, 'last_hidden_state'):
            features = outputs.last_hidden_state
        elif hasattr(outputs, 'pooler_output'):
            features = outputs.pooler_output
        elif isinstance(outputs, torch.Tensor):
            features = outputs
        elif hasattr(outputs, 'keys'):
            for key in ['last_hidden_state', 'hidden_states', 'logits', 'prediction_scores']:
                if hasattr(outputs, key):
                    features = getattr(outputs, key)
                    if isinstance(features, (list, tuple)):
                        features = features[-1]
                    break
            else:
                raise ValueError(f'Cannot extract features from model output, type={type(outputs)}')
        else:
            raise ValueError(f'Unknown model output format, type={type(outputs)}')
        feat_t = original_T // self.tubelet_size
        feat_h = H // self.downsample_factor
        feat_w = W // self.downsample_factor
        if len(features.shape) == 3:
            expected_tokens = feat_t * feat_h * feat_w
            if features.shape[1] == expected_tokens:
                features = features.reshape(B, feat_t, feat_h, feat_w, -1)
            elif features.shape[1] > expected_tokens:
                features = features[:, :expected_tokens, :]
                features = features.reshape(B, feat_t, feat_h, feat_w, -1)
            else:
                needed_tokens = expected_tokens - features.shape[1]
                last_token = features[:, -1:, :].repeat(1, needed_tokens, 1)
                features = torch.cat([features, last_token], dim=1)
                features = features.reshape(B, feat_t, feat_h, feat_w, -1)
        elif len(features.shape) == 4:
            features = features.unsqueeze(1).repeat(1, feat_t, 1, 1, 1)
        return features

def get_encoder(encoder_type: str, **kwargs) -> BaseEncoder:
    encoder_map = {'openclip': OpenCLIPEncoder, 'dinov2': DINOv2Encoder, 'raft': RAFTEncoder, 'jepavit': JEPAViTEncoder, 'vjepa2': VJEPA2Encoder}
    if encoder_type not in encoder_map:
        raise ValueError(f'Unsupported encoder type: {encoder_type}')
    return encoder_map[encoder_type](**kwargs)
