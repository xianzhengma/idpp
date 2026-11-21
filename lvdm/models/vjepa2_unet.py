import torch
import torch.nn as nn
from transformers import AutoModel, AutoVideoProcessor
import math
import re
from einops import repeat

def timestep_embedding(timesteps, dim, max_period=10000, repeat_only=False):
    if not repeat_only:
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(device=timesteps.device)
        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    else:
        embedding = repeat(timesteps, 'b -> b d', d=dim)
    return embedding

class VJEPA2_UNet_Wrapper(nn.Module):

    def __init__(self, in_channels: int, model_channels: int, out_channels: int, context_dim: int, vjepa_model_size: str='giant', vjepa_checkpoint_path: str=None, v84_feat_dims: list=None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.context_dim = context_dim
        self.model_channels = model_channels
        hf_repo = self._get_hf_repo(vjepa_model_size)
        print(f'Loading VJEPA2 model from: {hf_repo}')
        try:
            model_path = vjepa_checkpoint_path if vjepa_checkpoint_path else hf_repo
            self.vjepa = AutoModel.from_pretrained(model_path, output_hidden_states=True)
        except Exception as e:
            raise IOError(f'Failed to load VJEPA2 model: {e}')
        self.vjepa_hidden_dim = self.vjepa.config.hidden_size
        for param in self.vjepa.parameters():
            param.requires_grad = False
        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(nn.Linear(model_channels, time_embed_dim), nn.SiLU(), nn.Linear(time_embed_dim, self.vjepa_hidden_dim))
        vjepa_patch_embed = self.vjepa.embeddings.patch_embeddings.projection
        self.input_proj = nn.Conv3d(in_channels=in_channels, out_channels=self.vjepa_hidden_dim, kernel_size=vjepa_patch_embed.kernel_size, stride=vjepa_patch_embed.stride)
        self.pos_embed = self.vjepa.embeddings.position_embeddings
        self.context_proj = nn.Linear(context_dim, self.vjepa_hidden_dim)
        self.v84_feat_dims = v84_feat_dims or [320, 320, 320, 320, 640, 640, 640, 1280, 1280, 1280, 1280, 1280, 1280, 1280, 1280, 1280, 1280, 1280, 1280, 640, 640, 640, 320, 320, 320]
        num_vjepa_layers = len(self.vjepa.encoder.layer)
        self.vjepa_layer_indices = torch.linspace(0, num_vjepa_layers - 1, len(self.v84_feat_dims)).long().tolist()
        self.feat_list_projectors = nn.ModuleList([nn.Linear(self.vjepa_hidden_dim, c_out) for c_out in self.v84_feat_dims])
        self.out_proj = nn.ConvTranspose3d(self.vjepa_hidden_dim, out_channels, kernel_size=vjepa_patch_embed.kernel_size, stride=vjepa_patch_embed.stride)

    def _get_hf_repo(self, model_size: str) -> str:
        size_map = {'large': 'vitl', 'huge': 'vith', 'giant': 'vitg'}
        if model_size not in size_map:
            raise ValueError(f'Unsupported VJEPA2 model size: {model_size}.')
        return f'facebook/vjepa2-{size_map[model_size]}-fpc64-256'

    def forward(self, x, timesteps, context, **kwargs):
        B, C_in, T_in, H_in, W_in = x.shape
        x_embed = self.input_proj(x)
        B, D, T_p, H_p, W_p = x_embed.shape
        x_embed = x_embed.flatten(2).transpose(1, 2)
        if x_embed.shape[1] != self.pos_embed.shape[1]:
            num_patches = x_embed.shape[1]
            if num_patches > self.pos_embed.shape[1]:
                pos_embed = torch.cat([self.pos_embed] * (num_patches // self.pos_embed.shape[1]) + [self.pos_embed[:, :num_patches % self.pos_embed.shape[1]]], dim=1)
            else:
                pos_embed = self.pos_embed[:, :num_patches]
        else:
            pos_embed = self.pos_embed
        x_embed = x_embed + pos_embed
        t_embed = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        t_embed = t_embed.unsqueeze(1)
        x_embed = x_embed + t_embed
        c_crossattn = context['c_crossattn'][0]
        c_embed = self.context_proj(c_crossattn)
        full_seq = torch.cat([c_embed, x_embed], dim=1)
        vjepa_outputs = self.vjepa(inputs_embeds=full_seq, output_hidden_states=True)
        all_hidden_states = vjepa_outputs.hidden_states
        content_hidden_states = [h[:, c_embed.shape[1]:, :] for h in all_hidden_states]
        feat_list = []
        for i, proj_layer in enumerate(self.feat_list_projectors):
            vjepa_layer_idx = self.vjepa_layer_indices[i]
            hidden_state = content_hidden_states[vjepa_layer_idx + 1]
            projected_feat = proj_layer(hidden_state)
            C_feat = projected_feat.shape[-1]
            projected_feat_reshaped = projected_feat.transpose(1, 2)
            projected_feat_reshaped = projected_feat_reshaped.view(B, C_feat, T_p, H_p, W_p)
            final_feat = projected_feat_reshaped.permute(0, 2, 1, 3, 4)
            feat_list.append(final_feat)
        last_hidden_state = content_hidden_states[-1]
        last_hidden_state_reshaped = last_hidden_state.transpose(1, 2).view(B, D, T_p, H_p, W_p)
        model_output = self.out_proj(last_hidden_state_reshaped)
        if model_output.shape != x.shape:
            model_output = nn.functional.interpolate(model_output, size=x.shape[2:], mode='trilinear', align_corners=False)
        return (model_output, feat_list)

class VJEPA2_Direct_Wrapper(nn.Module):

    def __init__(self, context_dim: int, vjepa_model_size: str='giant', vjepa_checkpoint_path: str=None, num_output_features: int=16, temporal_length: int=16, image_cross_attention: bool=True, fix_input_rescale: bool=False, **kwargs):
        super().__init__()
        self.context_dim = context_dim
        self.num_output_features = num_output_features
        self.temporal_length = temporal_length
        self.image_cross_attention = image_cross_attention
        # Frames arrive in [-1, 1] and the HF processor rescales by 1/255 by default, so the default path rescales twice.
        # Released checkpoints were trained that way; fix_input_rescale=True maps frames to [0, 1] and skips the extra rescale.
        self.fix_input_rescale = fix_input_rescale
        hf_repo = self._get_hf_repo(vjepa_model_size)
        print(f'Loading VJEPA2 model for direct video processing from: {hf_repo}')
        try:
            model_path = vjepa_checkpoint_path if vjepa_checkpoint_path else hf_repo
            self.vjepa = AutoModel.from_pretrained(model_path, output_hidden_states=True)
            self.processor = AutoVideoProcessor.from_pretrained(model_path)
        except Exception as e:
            raise IOError(f'Failed to load VJEPA2 model or processor: {e}')
        self.vjepa_hidden_dim = self.vjepa.config.hidden_size
        for param in self.vjepa.parameters():
            param.requires_grad = False
        self.context_proj = nn.Linear(context_dim, self.vjepa_hidden_dim)
        num_vjepa_layers = len(self.vjepa.encoder.layer)
        self.vjepa_layer_indices = torch.linspace(0, num_vjepa_layers - 1, self.num_output_features).long().tolist()
        print(f'VJEPA2 Wrapper will output {self.num_output_features} features.')
        print(f'Selected VJEPA layer indices: {self.vjepa_layer_indices}')
        self.feat_list_projectors = None

    def _get_hf_repo(self, model_size: str) -> str:
        size_map = {'large': 'vitl', 'huge': 'vith', 'giant': 'vitg'}
        if model_size not in size_map:
            raise ValueError(f'Unsupported VJEPA2 model size: {model_size}.')
        return f'facebook/vjepa2-{size_map[model_size]}-fpc64-256'

    def forward(self, x, t=None, timesteps=None, cond=None, context=None, **kwargs):
        final_context = cond if cond is not None else context
        B, C, T_in, H, W = x.shape
        x_permuted = x.permute(0, 2, 1, 3, 4)
        if self.fix_input_rescale:
            video_list = [((frame + 1) / 2).clamp(0, 1) for frame in x_permuted]
            inputs = self.processor(video_list, return_tensors='pt', do_rescale=False).to(x.device)
        else:
            video_list = [frame for frame in x_permuted]
            inputs = self.processor(video_list, return_tensors='pt').to(x.device)
        vjepa_outputs = self.vjepa(**inputs, output_hidden_states=True)
        all_hidden_states = vjepa_outputs.hidden_states[1:]
        feat_list = []
        processed_video = inputs['pixel_values_videos']
        _, _, _, H, W = processed_video.shape
        cfg = self.vjepa.config
        patch_size = cfg.patch_size
        H_p = H // patch_size
        W_p = W // patch_size
        temp_hidden_state = all_hidden_states[0]
        B, N_patches, D = temp_hidden_state.shape
        assert N_patches % (H_p * W_p) == 0, f'Total patches {N_patches} not divisible by spatial patches {H_p}x{W_p}'
        T_p = N_patches // (H_p * W_p)
        for vjepa_layer_idx in self.vjepa_layer_indices:
            hidden_state = all_hidden_states[vjepa_layer_idx]
            hidden_state_final = all_hidden_states[-1]
            C_feat = self.vjepa_hidden_dim
            feat_reshaped = hidden_state.transpose(1, 2)
            feat_reshaped = feat_reshaped.view(B, C_feat, T_p, H_p, W_p)
            final_feat = feat_reshaped.permute(0, 2, 1, 3, 4)
            feat_list.append(final_feat)
        model_output = hidden_state_final
        return (model_output, feat_list)
