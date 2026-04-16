from functools import partial
from contextlib import contextmanager
import numpy as np
from tqdm import tqdm
from einops import rearrange, repeat
import logging
mainlogger = logging.getLogger('mainlogger')
import random
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR
import torch.nn.functional as F
from torchvision.utils import make_grid
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only
from utils.utils import instantiate_from_config
from lvdm.ema import LitEma
from lvdm.models.samplers.ddim import DDIMSampler
from lvdm.models.p_decoder import PDecoder, PDecoderSupElasticity, PDecoderSupElasticityv1, PDecoderSupElasticityv2, PDecoderSupElasticityv2512, PDecoderSupElasticityv3, PDecoderSupElasticityv3512, PDecoderSupElasticityv4, PDecoderFC1, PDecoderFC2, PDecoderSigmoid, PDecoderCondMask, PDecoderAdv, PDecoderTemporalConv, PDecoderTemporalGRU, PDecoderProbev1, PDecoderProbev2, PDecoderProbev3
from lvdm.distributions import DiagonalGaussianDistribution
from lvdm.models.utils_diffusion import make_beta_schedule, rescale_zero_terminal_snr
from lvdm.basics import disabled_train
from lvdm.common import extract_into_tensor, noise_like, exists, default
import ipdb
import os
import json
import re
import uuid
import inspect
from lvdm.models.vjepa2_unet import VJEPA2_Direct_Wrapper
from lvdm.models.attentive_probe import AttentiveClassifier
__conditioning_keys__ = {'concat': 'c_concat', 'crossattn': 'c_crossattn', 'adm': 'y'}


def _dump_elasticity_abs_clip_json(save_dir: str, clip_path: str, pred_val: float, gt_val: float) -> None:
    """
    作用：为单条 clip 写入与 `calculate_e729_newdata_dc_scores.load_results` 兼容的 JSON（顶层 key 为路径，值为含 pred/gt 的字典）。

    输入：save_dir 输出目录；clip_path 原始路径或文件名；pred_val、gt_val 标量。
    输出：在 save_dir 下写入基于完整路径清洗后的唯一 `.json` 文件名，避免不同目录下同 basename 互相覆盖。
    """
    try:
        outp = os.path.join(save_dir, f'clip_{uuid.uuid4().hex}.json')
        payload = {clip_path: {'pred': float(pred_val), 'gt': float(gt_val)}}
        with open(outp, 'w', encoding='utf-8') as handle:
            json.dump(payload, handle, indent=2)
    except (OSError, TypeError, ValueError) as exc:
        mainlogger.warning('elasticity abs json dump skipped: %s', exc)

class DDPM(pl.LightningModule):

    def __init__(self, unet_config, timesteps=1000, beta_schedule='linear', loss_type='l2', ckpt_path=None, ignore_keys=[], load_only_unet=False, monitor=None, use_ema=True, first_stage_key='image', image_size=256, channels=3, log_every_t=100, clip_denoised=True, linear_start=0.0001, linear_end=0.02, cosine_s=0.008, given_betas=None, original_elbo_weight=0.0, v_posterior=0.0, l_simple_weight=1.0, conditioning_key=None, parameterization='eps', scheduler_config=None, use_positional_encodings=False, learn_logvar=False, logvar_init=0.0, rescale_betas_zero_snr=False):
        super().__init__()
        assert parameterization in ['eps', 'x0', 'v'], 'currently only supporting "eps" and "x0" and "v"'
        self.parameterization = parameterization
        mainlogger.info(f'{self.__class__.__name__}: Running in {self.parameterization}-prediction mode')
        self.cond_stage_model = None
        self.clip_denoised = clip_denoised
        self.log_every_t = log_every_t
        self.first_stage_key = first_stage_key
        self.channels = channels
        self.temporal_length = unet_config.params.temporal_length
        self.image_size = image_size
        if isinstance(self.image_size, int):
            self.image_size = [self.image_size, self.image_size]
        self.use_positional_encodings = use_positional_encodings
        self.model = DiffusionWrapper(unet_config, conditioning_key)
        self.use_ema = use_ema
        self.rescale_betas_zero_snr = rescale_betas_zero_snr
        if self.use_ema:
            self.model_ema = LitEma(self.model)
            mainlogger.info(f'Keeping EMAs of {len(list(self.model_ema.buffers()))}.')
        self.use_scheduler = scheduler_config is not None
        if self.use_scheduler:
            self.scheduler_config = scheduler_config
        self.v_posterior = v_posterior
        self.original_elbo_weight = original_elbo_weight
        self.l_simple_weight = l_simple_weight
        if monitor is not None:
            self.monitor = monitor
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys, only_model=load_only_unet)
        self.register_schedule(given_betas=given_betas, beta_schedule=beta_schedule, timesteps=timesteps, linear_start=linear_start, linear_end=linear_end, cosine_s=cosine_s)
        self.given_betas = given_betas
        self.beta_schedule = beta_schedule
        self.timesteps = timesteps
        self.cosine_s = cosine_s
        self.loss_type = loss_type
        self.learn_logvar = learn_logvar
        self.logvar = torch.full(fill_value=logvar_init, size=(self.num_timesteps,))
        if self.learn_logvar:
            self.logvar = nn.Parameter(self.logvar, requires_grad=True)

    def register_schedule(self, given_betas=None, beta_schedule='linear', timesteps=1000, linear_start=0.0001, linear_end=0.02, cosine_s=0.008):
        if exists(given_betas):
            betas = given_betas
        else:
            betas = make_beta_schedule(beta_schedule, timesteps, linear_start=linear_start, linear_end=linear_end, cosine_s=cosine_s)
        if self.rescale_betas_zero_snr:
            betas = rescale_zero_terminal_snr(betas)
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])
        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.linear_start = linear_start
        self.linear_end = linear_end
        assert alphas_cumprod.shape[0] == self.num_timesteps, 'alphas have to be defined for each timestep'
        to_torch = partial(torch.tensor, dtype=torch.float32)
        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1.0 - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1.0 - alphas_cumprod)))
        if self.parameterization != 'v':
            self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1.0 / alphas_cumprod)))
            self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1.0 / alphas_cumprod - 1)))
        else:
            self.register_buffer('sqrt_recip_alphas_cumprod', torch.zeros_like(to_torch(alphas_cumprod)))
            self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.zeros_like(to_torch(alphas_cumprod)))
        posterior_variance = (1 - self.v_posterior) * betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod) + self.v_posterior * betas
        self.register_buffer('posterior_variance', to_torch(posterior_variance))
        self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch((1.0 - alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - alphas_cumprod)))
        if self.parameterization == 'eps':
            lvlb_weights = self.betas ** 2 / (2 * self.posterior_variance * to_torch(alphas) * (1 - self.alphas_cumprod))
        elif self.parameterization == 'x0':
            lvlb_weights = 0.5 * np.sqrt(torch.Tensor(alphas_cumprod)) / (2.0 * 1 - torch.Tensor(alphas_cumprod))
        elif self.parameterization == 'v':
            lvlb_weights = torch.ones_like(self.betas ** 2 / (2 * self.posterior_variance * to_torch(alphas) * (1 - self.alphas_cumprod)))
        else:
            raise NotImplementedError('mu not supported')
        lvlb_weights[0] = lvlb_weights[1]
        self.register_buffer('lvlb_weights', lvlb_weights, persistent=False)
        assert not torch.isnan(self.lvlb_weights).all()

    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store(self.model.parameters())
            self.model_ema.copy_to(self.model)
            if context is not None:
                mainlogger.info(f'{context}: Switched to EMA weights')
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.model.parameters())
                if context is not None:
                    mainlogger.info(f'{context}: Restored training weights')

    def init_from_ckpt(self, path, ignore_keys=list(), only_model=False):
        sd = torch.load(path, map_location='cpu')
        if 'state_dict' in list(sd.keys()):
            sd = sd['state_dict']
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    mainlogger.info('Deleting key {} from state_dict.'.format(k))
                    del sd[k]
        missing, unexpected = self.load_state_dict(sd, strict=False) if not only_model else self.model.load_state_dict(sd, strict=False)
        mainlogger.info(f'Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys')
        if len(missing) > 0:
            mainlogger.info(f'Missing Keys: {missing}')
        if len(unexpected) > 0:
            mainlogger.info(f'Unexpected Keys: {unexpected}')

    def q_mean_variance(self, x_start, t):
        mean = extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = extract_into_tensor(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract_into_tensor(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return (mean, variance, log_variance)

    def predict_start_from_noise(self, x_t, t, noise):
        return extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise

    def predict_start_from_z_and_v(self, x_t, t, v):
        return extract_into_tensor(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t - extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v

    def predict_eps_from_z_and_v(self, x_t, t, v):
        return extract_into_tensor(self.sqrt_alphas_cumprod, t, x_t.shape) * v + extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * x_t

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start + extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        posterior_variance = extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract_into_tensor(self.posterior_log_variance_clipped, t, x_t.shape)
        return (posterior_mean, posterior_variance, posterior_log_variance_clipped)

    def p_mean_variance(self, x, t, clip_denoised: bool):
        model_out = self.model(x, t)
        if self.parameterization == 'eps':
            x_recon = self.predict_start_from_noise(x, t=t, noise=model_out)
        elif self.parameterization == 'x0':
            x_recon = model_out
        if clip_denoised:
            x_recon.clamp_(-1.0, 1.0)
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return (model_mean, posterior_variance, posterior_log_variance)

    @torch.no_grad()
    def p_sample(self, x, t, clip_denoised=True, repeat_noise=False):
        b, *_, device = (*x.shape, x.device)
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, t=t, clip_denoised=clip_denoised)
        noise = noise_like(x.shape, device, repeat_noise)
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *(1,) * (len(x.shape) - 1))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_loop(self, shape, return_intermediates=False):
        device = self.betas.device
        b = shape[0]
        img = torch.randn(shape, device=device)
        intermediates = [img]
        for i in tqdm(reversed(range(0, self.num_timesteps)), desc='Sampling t', total=self.num_timesteps):
            img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long), clip_denoised=self.clip_denoised)
            if i % self.log_every_t == 0 or i == self.num_timesteps - 1:
                intermediates.append(img)
        if return_intermediates:
            return (img, intermediates)
        return img

    @torch.no_grad()
    def sample(self, batch_size=16, return_intermediates=False):
        image_size = self.image_size
        channels = self.channels
        return self.p_sample_loop((batch_size, channels, image_size, image_size), return_intermediates=return_intermediates)

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        return extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start + extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise

    def get_v(self, x, noise, t):
        return extract_into_tensor(self.sqrt_alphas_cumprod, t, x.shape) * noise - extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x.shape) * x

    def get_loss(self, pred, target, mean=True):
        if self.loss_type == 'l1':
            loss = (target - pred).abs()
            if mean:
                loss = loss.mean()
        elif self.loss_type == 'l2':
            if mean:
                loss = torch.nn.functional.mse_loss(target, pred)
            else:
                loss = torch.nn.functional.mse_loss(target, pred, reduction='none')
        else:
            raise NotImplementedError("unknown loss type '{loss_type}'")
        return loss

    def p_losses(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        model_out = self.model(x_noisy, t)
        loss_dict = {}
        if self.parameterization == 'eps':
            target = noise
        elif self.parameterization == 'x0':
            target = x_start
        elif self.parameterization == 'v':
            target = self.get_v(x_start, noise, t)
        else:
            raise NotImplementedError(f'Paramterization {self.parameterization} not yet supported')
        loss = self.get_loss(model_out, target, mean=False).mean(dim=[1, 2, 3])
        log_prefix = 'train' if self.training else 'val'
        loss_dict.update({f'{log_prefix}/loss_simple': loss.mean()})
        loss_simple = loss.mean() * self.l_simple_weight
        loss_vlb = (self.lvlb_weights[t] * loss).mean()
        loss_dict.update({f'{log_prefix}/loss_vlb': loss_vlb})
        loss = loss_simple + self.original_elbo_weight * loss_vlb
        loss_dict.update({f'{log_prefix}/loss': loss})
        return (loss, loss_dict)

    def forward(self, x, *args, **kwargs):
        t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=self.device).long()
        return self.p_losses(x, t, *args, **kwargs)

    def get_input(self, batch, k):
        x = batch[k]
        "\n        if len(x.shape) == 3:\n            x = x[..., None]\n        x = rearrange(x, 'b h w c -> b c h w')\n        "
        x = x.to(memory_format=torch.contiguous_format).float()
        return x

    def shared_step(self, batch):
        x = self.get_input(batch, self.first_stage_key)
        loss, loss_dict = self(x)
        return (loss, loss_dict)

    def training_step(self, batch, batch_idx):
        loss, loss_dict = self.shared_step(batch)
        self.log_dict(loss_dict, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log('global_step', self.global_step, prog_bar=True, logger=True, on_step=True, on_epoch=False)
        if self.use_scheduler:
            lr = self.optimizers().param_groups[0]['lr']
            self.log('lr_abs', lr, prog_bar=True, logger=True, on_step=True, on_epoch=False)
        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        _, loss_dict_no_ema = self.shared_step(batch)
        with self.ema_scope():
            _, loss_dict_ema = self.shared_step(batch)
            loss_dict_ema = {key + '_ema': loss_dict_ema[key] for key in loss_dict_ema}
        self.log_dict(loss_dict_no_ema, prog_bar=False, logger=True, on_step=False, on_epoch=True)
        self.log_dict(loss_dict_ema, prog_bar=False, logger=True, on_step=False, on_epoch=True)

    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            self.model_ema(self.model)

    def _get_rows_from_list(self, samples):
        n_imgs_per_row = len(samples)
        denoise_grid = rearrange(samples, 'n b c h w -> b n c h w')
        denoise_grid = rearrange(denoise_grid, 'b n c h w -> (b n) c h w')
        denoise_grid = make_grid(denoise_grid, nrow=n_imgs_per_row)
        return denoise_grid

    @torch.no_grad()
    def log_images(self, batch, N=8, n_row=2, sample=True, return_keys=None, **kwargs):
        log = dict()
        x = self.get_input(batch, self.first_stage_key)
        N = min(x.shape[0], N)
        n_row = min(x.shape[0], n_row)
        x = x.to(self.device)[:N]
        log['inputs'] = x
        diffusion_row = list()
        x_start = x[:n_row]
        for t in range(self.num_timesteps):
            if t % self.log_every_t == 0 or t == self.num_timesteps - 1:
                t = repeat(torch.tensor([t]), '1 -> b', b=n_row)
                t = t.to(self.device).long()
                noise = torch.randn_like(x_start)
                x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
                diffusion_row.append(x_noisy)
        log['diffusion_row'] = self._get_rows_from_list(diffusion_row)
        if sample:
            with self.ema_scope('Plotting'):
                samples, denoise_row = self.sample(batch_size=N, return_intermediates=True)
            log['samples'] = samples
            log['denoise_row'] = self._get_rows_from_list(denoise_row)
        if return_keys:
            if np.intersect1d(list(log.keys()), return_keys).shape[0] == 0:
                return log
            else:
                return {key: log[key] for key in return_keys}
        return log

    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.model.parameters())
        if self.learn_logvar:
            params = params + [self.logvar]
        opt = torch.optim.AdamW(params, lr=lr)
        return opt

class LatentDiffusion(DDPM):

    def __init__(self, first_stage_config, cond_stage_config, num_timesteps_cond=None, cond_stage_key='caption', cond_stage_trainable=False, cond_stage_forward=None, conditioning_key=None, uncond_prob=0.2, uncond_type='empty_seq', scale_factor=1.0, scale_by_std=False, encoder_type='2d', only_model=False, noise_strength=0, use_dynamic_rescale=False, base_scale=0.7, turning_step=400, interp_mode=False, fps_condition_type='fs', perframe_ae=False, logdir=None, rand_cond_frame=False, en_and_decode_n_samples_a_time=None, *args, **kwargs):
        self.num_timesteps_cond = default(num_timesteps_cond, 1)
        self.scale_by_std = scale_by_std
        assert self.num_timesteps_cond <= kwargs['timesteps']
        ckpt_path = kwargs.pop('ckpt_path', None)
        ignore_keys = kwargs.pop('ignore_keys', [])
        conditioning_key = default(conditioning_key, 'crossattn')
        super().__init__(*args, conditioning_key=conditioning_key, **kwargs)
        self.cond_stage_trainable = cond_stage_trainable
        self.cond_stage_key = cond_stage_key
        self.noise_strength = noise_strength
        self.use_dynamic_rescale = use_dynamic_rescale
        self.interp_mode = interp_mode
        self.fps_condition_type = fps_condition_type
        self.perframe_ae = perframe_ae
        self.logdir = logdir
        self.rand_cond_frame = rand_cond_frame
        self.en_and_decode_n_samples_a_time = en_and_decode_n_samples_a_time
        try:
            self.num_downs = len(first_stage_config.params.ddconfig.ch_mult) - 1
        except:
            self.num_downs = 0
        if not scale_by_std:
            self.scale_factor = scale_factor
        else:
            self.register_buffer('scale_factor', torch.tensor(scale_factor))
        if use_dynamic_rescale:
            scale_arr1 = np.linspace(1.0, base_scale, turning_step)
            scale_arr2 = np.full(self.num_timesteps, base_scale)
            scale_arr = np.concatenate((scale_arr1, scale_arr2))
            to_torch = partial(torch.tensor, dtype=torch.float32)
            self.register_buffer('scale_arr', to_torch(scale_arr))
        self.instantiate_first_stage(first_stage_config)
        self.instantiate_cond_stage(cond_stage_config)
        self.first_stage_config = first_stage_config
        self.cond_stage_config = cond_stage_config
        self.clip_denoised = False
        self.cond_stage_forward = cond_stage_forward
        self.encoder_type = encoder_type
        assert encoder_type in ['2d', '3d']
        self.uncond_prob = uncond_prob
        self.classifier_free_guidance = True if uncond_prob > 0 else False
        assert uncond_type in ['zero_embed', 'empty_seq']
        self.uncond_type = uncond_type
        self.restarted_from_ckpt = False
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys, only_model=only_model)
            self.restarted_from_ckpt = True

    def make_cond_schedule(self):
        self.cond_ids = torch.full(size=(self.num_timesteps,), fill_value=self.num_timesteps - 1, dtype=torch.long)
        ids = torch.round(torch.linspace(0, self.num_timesteps - 1, self.num_timesteps_cond)).long()
        self.cond_ids[:self.num_timesteps_cond] = ids

    @rank_zero_only
    @torch.no_grad()
    def on_train_batch_start(self, batch, batch_idx, dataloader_idx=None):
        if self.scale_by_std and self.current_epoch == 0 and (self.global_step == 0) and (batch_idx == 0) and (not self.restarted_from_ckpt):
            assert self.scale_factor == 1.0, 'rather not use custom rescaling and std-rescaling simultaneously'
            mainlogger.info('### USING STD-RESCALING ###')
            x = super().get_input(batch, self.first_stage_key)
            x = x.to(self.device)
            encoder_posterior = self.encode_first_stage(x)
            z = self.get_first_stage_encoding(encoder_posterior).detach()
            del self.scale_factor
            self.register_buffer('scale_factor', 1.0 / z.flatten().std())
            mainlogger.info(f'setting self.scale_factor to {self.scale_factor}')
            mainlogger.info('### USING STD-RESCALING ###')
            mainlogger.info(f'std={z.flatten().std()}')

    def register_schedule(self, given_betas=None, beta_schedule='linear', timesteps=1000, linear_start=0.0001, linear_end=0.02, cosine_s=0.008):
        super().register_schedule(given_betas, beta_schedule, timesteps, linear_start, linear_end, cosine_s)
        self.shorten_cond_schedule = self.num_timesteps_cond > 1
        if self.shorten_cond_schedule:
            self.make_cond_schedule()

    def instantiate_first_stage(self, config):
        model = instantiate_from_config(config)
        self.first_stage_model = model.eval()
        self.first_stage_model.train = disabled_train
        for param in self.first_stage_model.parameters():
            param.requires_grad = False

    def instantiate_cond_stage(self, config):
        if not self.cond_stage_trainable:
            model = instantiate_from_config(config)
            self.cond_stage_model = model.eval()
            self.cond_stage_model.train = disabled_train
            for param in self.cond_stage_model.parameters():
                param.requires_grad = False
        else:
            model = instantiate_from_config(config)
            self.cond_stage_model = model

    def get_learned_conditioning(self, c):
        if self.cond_stage_forward is None:
            if hasattr(self.cond_stage_model, 'encode') and callable(self.cond_stage_model.encode):
                c = self.cond_stage_model.encode(c)
                if isinstance(c, DiagonalGaussianDistribution):
                    c = c.mode()
            else:
                c = self.cond_stage_model(c)
        else:
            assert hasattr(self.cond_stage_model, self.cond_stage_forward)
            c = getattr(self.cond_stage_model, self.cond_stage_forward)(c)
        return c

    def get_first_stage_encoding(self, encoder_posterior, noise=None):
        if isinstance(encoder_posterior, DiagonalGaussianDistribution):
            z = encoder_posterior.sample(noise=noise)
        elif isinstance(encoder_posterior, torch.Tensor):
            z = encoder_posterior
        else:
            raise NotImplementedError(f"encoder_posterior of type '{type(encoder_posterior)}' not yet implemented")
        return self.scale_factor * z

    @torch.no_grad()
    def encode_first_stage(self, x):
        if self.encoder_type == '2d' and x.dim() == 5:
            b, _, t, _, _ = x.shape
            x = rearrange(x, 'b c t h w -> (b t) c h w')
            reshape_back = True
        else:
            reshape_back = False
        if not self.perframe_ae:
            encoder_posterior = self.first_stage_model.encode(x)
            results = self.get_first_stage_encoding(encoder_posterior).detach()
        else:
            results = []
            for index in range(x.shape[0]):
                frame_batch = self.first_stage_model.encode(x[index:index + 1, :, :, :])
                frame_result = self.get_first_stage_encoding(frame_batch).detach()
                results.append(frame_result)
            results = torch.cat(results, dim=0)
        if reshape_back:
            results = rearrange(results, '(b t) c h w -> b c t h w', b=b, t=t)
        return results

    def decode_core(self, z, **kwargs):
        if self.encoder_type == '2d' and z.dim() == 5:
            b, _, t, _, _ = z.shape
            z = rearrange(z, 'b c t h w -> (b t) c h w')
            reshape_back = True
        else:
            reshape_back = False
        if not self.perframe_ae:
            z = 1.0 / self.scale_factor * z
            results = self.first_stage_model.decode(z, **kwargs)
        else:
            results = []
            for index in range(z.shape[0]):
                frame_z = 1.0 / self.scale_factor * z[index:index + 1, :, :, :]
                frame_result = self.first_stage_model.decode(frame_z, **kwargs)
                results.append(frame_result)
            results = torch.cat(results, dim=0)
        if reshape_back:
            results = rearrange(results, '(b t) c h w -> b c t h w', b=b, t=t)
        return results

    @torch.no_grad()
    def decode_first_stage(self, z, **kwargs):
        return self.decode_core(z, **kwargs)

    def differentiable_decode_first_stage(self, z, **kwargs):
        return self.decode_core(z, **kwargs)

    @torch.no_grad()
    def get_batch_input(self, batch, random_uncond, return_first_stage_outputs=False, return_original_cond=False):
        x = super().get_input(batch, self.first_stage_key)
        z = self.encode_first_stage(x)
        cond = batch[self.cond_stage_key]
        if random_uncond and self.uncond_type == 'empty_seq':
            for i, ci in enumerate(cond):
                if random.random() < self.uncond_prob:
                    cond[i] = ''
        if isinstance(cond, dict) or isinstance(cond, list):
            cond_emb = self.get_learned_conditioning(cond)
        else:
            cond_emb = self.get_learned_conditioning(cond.to(self.device))
        if random_uncond and self.uncond_type == 'zero_embed':
            for i, ci in enumerate(cond):
                if random.random() < self.uncond_prob:
                    cond_emb[i] = torch.zeros_like(cond_emb[i])
        out = [z, cond_emb]
        if return_first_stage_outputs:
            xrec = self.decode_first_stage(z)
            out.extend([xrec])
        if return_original_cond:
            out.append(cond)
        return out

    def forward(self, x, c, gt, mask, **kwargs):
        t = torch.randint(1, 2, (x.shape[0],), device=self.device).long()
        if self.use_dynamic_rescale:
            x = x * extract_into_tensor(self.scale_arr, t, x.shape)
        return self.p_losses(x, c, t, gt, mask, **kwargs)

    def shared_step(self, batch, random_uncond, **kwargs):
        x, c = self.get_batch_input(batch, random_uncond=random_uncond)
        loss, loss_dict = self(x, c, **kwargs)
        return (loss, loss_dict)

    def apply_model(self, x_noisy, t, cond, **kwargs):
        if hasattr(self, 'parameterization') and self.parameterization == 'vjepa_direct':
            return self.model.diffusion_model(x_noisy, context=cond, **kwargs)
        if isinstance(cond, dict):
            pass
        else:
            if not isinstance(cond, list):
                cond = [cond]
            key = 'c_concat' if self.model.conditioning_key == 'concat' else 'c_crossattn'
            cond = {key: cond}
        x_recon, feat_list = self.model(x_noisy, t, **cond, **kwargs)
        if isinstance(x_recon, tuple):
            return (x_recon[0], feat_list)
        else:
            return (x_recon, feat_list)

    def p_losses(self, x_start, cond, t, gt, mask, noise=None, **kwargs):
        if self.noise_strength > 0:
            b, c, f, _, _ = x_start.shape
            offset_noise = torch.randn(b, c, f, 1, 1, device=x_start.device)
            noise = default(noise, lambda: torch.randn_like(x_start) + self.noise_strength * offset_noise)
        else:
            noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        model_output, feat_list = self.apply_model(x_noisy, t, cond, **kwargs)
        feat_dict = {}
        feat_dict[0] = (feat_list[0] + feat_list[1] + feat_list[2] + feat_list[-1] + feat_list[-2] + feat_list[-3]) / 6
        feat_dict[4] = feat_list[-4]
        feat_dict[1] = (feat_list[4] + feat_list[5] + feat_list[-5] + feat_list[-6]) / 4
        feat_dict[5] = feat_list[-7]
        feat_dict[6] = feat_list[3]
        feat_dict[2] = (feat_list[7] + feat_list[8] + feat_list[-8] + feat_list[-9] + feat_list[-10]) / 6
        feat_dict[7] = feat_list[6]
        feat_dict[3] = (feat_list[9] + feat_list[10] + feat_list[11] + feat_list[-11] + feat_list[-12] + feat_list[-13]) / 6
        p_output = self.p_decoder(feat_dict, mask)
        l1_loss = nn.L1Loss()
        loss = l1_loss(p_output[:, 0], gt)
        loss_dict = {}
        loss_dict.update({'p_l1_loss': loss})
        return (loss, loss_dict)

    def training_step(self, batch, batch_idx):
        loss, loss_dict = self.shared_step(batch, random_uncond=self.classifier_free_guidance)
        self.log_dict(loss_dict, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=False)
        "\n        if self.use_scheduler:\n            lr = self.optimizers().param_groups[0]['lr']\n            self.log('lr_abs', lr, prog_bar=True, logger=True, on_step=True, on_epoch=False, rank_zero_only=True)\n        "
        if (batch_idx + 1) % self.log_every_t == 0:
            mainlogger.info(f'batch:{batch_idx}|epoch:{self.current_epoch} [globalstep:{self.global_step}]: loss={loss}')
        return loss

    def _get_denoise_row_from_list(self, samples, desc=''):
        denoise_row = []
        for zd in tqdm(samples, desc=desc):
            denoise_row.append(self.decode_first_stage(zd.to(self.device)))
        n_log_timesteps = len(denoise_row)
        denoise_row = torch.stack(denoise_row)
        if denoise_row.dim() == 5:
            denoise_grid = rearrange(denoise_row, 'n b c h w -> b n c h w')
            denoise_grid = rearrange(denoise_grid, 'b n c h w -> (b n) c h w')
            denoise_grid = make_grid(denoise_grid, nrow=n_log_timesteps)
        elif denoise_row.dim() == 6:
            video_length = denoise_row.shape[3]
            denoise_grid = rearrange(denoise_row, 'n b c t h w -> b n c t h w')
            denoise_grid = rearrange(denoise_grid, 'b n c t h w -> (b n) c t h w')
            denoise_grid = rearrange(denoise_grid, 'n c t h w -> (n t) c h w')
            denoise_grid = make_grid(denoise_grid, nrow=video_length)
        else:
            raise ValueError
        return denoise_grid

    @torch.no_grad()
    def log_images(self, batch, sample=True, ddim_steps=200, ddim_eta=1.0, plot_denoise_rows=False, unconditional_guidance_scale=1.0, **kwargs):
        sampled_img_num = 2
        for key in batch.keys():
            batch[key] = batch[key][:sampled_img_num]
        use_ddim = ddim_steps is not None
        log = dict()
        z, c, xrec, xc = self.get_batch_input(batch, random_uncond=False, return_first_stage_outputs=True, return_original_cond=True)
        N = xrec.shape[0]
        log['reconst'] = xrec
        log['condition'] = xc
        if sample:
            if unconditional_guidance_scale != 1.0:
                if isinstance(c, dict):
                    c_cat, c_emb = (c['c_concat'][0], c['c_crossattn'][0])
                    log['condition_cat'] = c_cat
                else:
                    c_emb = c
                if self.uncond_type == 'empty_seq':
                    prompts = N * ['']
                    uc = self.get_learned_conditioning(prompts)
                elif self.uncond_type == 'zero_embed':
                    uc = torch.zeros_like(c_emb)
                if isinstance(c, dict):
                    uc_hybrid = {'c_concat': [c_cat], 'c_crossattn': [uc]}
                    uc = uc_hybrid
            else:
                uc = None
            with self.ema_scope('Plotting'):
                samples, z_denoise_row = self.sample_log(cond=c, batch_size=N, ddim=use_ddim, ddim_steps=ddim_steps, eta=ddim_eta, unconditional_guidance_scale=unconditional_guidance_scale, unconditional_conditioning=uc, x0=z, **kwargs)
            x_samples = self.decode_first_stage(samples)
            log['samples'] = x_samples
            if plot_denoise_rows:
                denoise_grid = self._get_denoise_row_from_list(z_denoise_row)
                log['denoise_row'] = denoise_grid
        return log

    def p_mean_variance(self, x, c, t, clip_denoised: bool, return_x0=False, score_corrector=None, corrector_kwargs=None, **kwargs):
        t_in = t
        model_out = self.apply_model(x, t_in, c, **kwargs)
        if score_corrector is not None:
            assert self.parameterization == 'eps'
            model_out = score_corrector.modify_score(self, model_out, x, t, c, **corrector_kwargs)
        if self.parameterization == 'eps':
            x_recon = self.predict_start_from_noise(x, t=t, noise=model_out)
        elif self.parameterization == 'x0':
            x_recon = model_out
        else:
            raise NotImplementedError()
        if clip_denoised:
            x_recon.clamp_(-1.0, 1.0)
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        if return_x0:
            return (model_mean, posterior_variance, posterior_log_variance, x_recon)
        else:
            return (model_mean, posterior_variance, posterior_log_variance)

    @torch.no_grad()
    def p_sample(self, x, c, t, clip_denoised=False, repeat_noise=False, return_x0=False, temperature=1.0, noise_dropout=0.0, score_corrector=None, corrector_kwargs=None, **kwargs):
        b, *_, device = (*x.shape, x.device)
        outputs = self.p_mean_variance(x=x, c=c, t=t, clip_denoised=clip_denoised, return_x0=return_x0, score_corrector=score_corrector, corrector_kwargs=corrector_kwargs, **kwargs)
        if return_x0:
            model_mean, _, model_log_variance, x0 = outputs
        else:
            model_mean, _, model_log_variance = outputs
        noise = noise_like(x.shape, device, repeat_noise) * temperature
        if noise_dropout > 0.0:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *(1,) * (len(x.shape) - 1))
        if return_x0:
            return (model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise, x0)
        else:
            return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_loop(self, cond, shape, return_intermediates=False, x_T=None, verbose=True, callback=None, timesteps=None, mask=None, x0=None, img_callback=None, start_T=None, log_every_t=None, **kwargs):
        if not log_every_t:
            log_every_t = self.log_every_t
        device = self.betas.device
        b = shape[0]
        if x_T is None:
            img = torch.randn(shape, device=device)
        else:
            img = x_T
        intermediates = [img]
        if timesteps is None:
            timesteps = self.num_timesteps
        if start_T is not None:
            timesteps = min(timesteps, start_T)
        iterator = tqdm(reversed(range(0, timesteps)), desc='Sampling t', total=timesteps) if verbose else reversed(range(0, timesteps))
        if mask is not None:
            assert x0 is not None
            assert x0.shape[2:3] == mask.shape[2:3]
        for i in iterator:
            ts = torch.full((b,), i, device=device, dtype=torch.long)
            if self.shorten_cond_schedule:
                assert self.model.conditioning_key != 'hybrid'
                tc = self.cond_ids[ts].to(cond.device)
                cond = self.q_sample(x_start=cond, t=tc, noise=torch.randn_like(cond))
            img = self.p_sample(img, cond, ts, clip_denoised=self.clip_denoised, **kwargs)
            if mask is not None:
                img_orig = self.q_sample(x0, ts)
                img = img_orig * mask + (1.0 - mask) * img
            if i % log_every_t == 0 or i == timesteps - 1:
                intermediates.append(img)
            if callback:
                callback(i)
            if img_callback:
                img_callback(img, i)
        if return_intermediates:
            return (img, intermediates)
        return img

    @torch.no_grad()
    def sample(self, cond, batch_size=16, return_intermediates=False, x_T=None, verbose=True, timesteps=None, mask=None, x0=None, shape=None, **kwargs):
        if shape is None:
            shape = (batch_size, self.channels, self.temporal_length, *self.image_size)
        if cond is not None:
            if isinstance(cond, dict):
                cond = {key: cond[key][:batch_size] if not isinstance(cond[key], list) else list(map(lambda x: x[:batch_size], cond[key])) for key in cond}
            else:
                cond = [c[:batch_size] for c in cond] if isinstance(cond, list) else cond[:batch_size]
        return self.p_sample_loop(cond, shape, return_intermediates=return_intermediates, x_T=x_T, verbose=verbose, timesteps=timesteps, mask=mask, x0=x0, **kwargs)

    @torch.no_grad()
    def sample_log(self, cond, batch_size, ddim, ddim_steps, **kwargs):
        if ddim:
            ddim_sampler = DDIMSampler(self)
            shape = (self.channels, self.temporal_length, *self.image_size)
            samples, intermediates = ddim_sampler.sample(ddim_steps, batch_size, shape, cond, verbose=False, **kwargs)
        else:
            samples, intermediates = self.sample(cond=cond, batch_size=batch_size, return_intermediates=True, **kwargs)
        return (samples, intermediates)

    def configure_schedulers(self, optimizer):
        assert 'target' in self.scheduler_config
        scheduler_name = self.scheduler_config.target.split('.')[-1]
        interval = self.scheduler_config.interval
        frequency = self.scheduler_config.frequency
        if scheduler_name == 'LambdaLRScheduler':
            scheduler = instantiate_from_config(self.scheduler_config)
            scheduler.start_step = self.global_step
            lr_scheduler = {'scheduler': LambdaLR(optimizer, lr_lambda=scheduler.schedule), 'interval': interval, 'frequency': frequency}
        elif scheduler_name == 'CosineAnnealingLRScheduler':
            scheduler = instantiate_from_config(self.scheduler_config)
            decay_steps = scheduler.decay_steps
            last_step = -1 if self.global_step == 0 else scheduler.start_step
            lr_scheduler = {'scheduler': CosineAnnealingLR(optimizer, T_max=decay_steps, last_epoch=last_step), 'interval': interval, 'frequency': frequency}
        else:
            raise NotImplementedError
        return lr_scheduler

class LatentVisualDiffusion(LatentDiffusion):

    def __init__(self, img_cond_stage_config, image_proj_stage_config, freeze_embedder=True, image_proj_model_trainable=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.image_proj_model_trainable = image_proj_model_trainable
        self._init_embedder(img_cond_stage_config, freeze_embedder)
        self._init_img_ctx_projector(image_proj_stage_config, image_proj_model_trainable)

    def _init_img_ctx_projector(self, config, trainable):
        self.image_proj_model = instantiate_from_config(config)
        if not trainable:
            self.image_proj_model.eval()
            self.image_proj_model.train = disabled_train
            for param in self.image_proj_model.parameters():
                param.requires_grad = False

    def _init_embedder(self, config, freeze=True):
        self.embedder = instantiate_from_config(config)
        if freeze:
            self.embedder.eval()
            self.embedder.train = disabled_train
            for param in self.embedder.parameters():
                param.requires_grad = False

    def _freeze_all(self):
        for name, param in self.named_parameters():
            if 'p_decoder' not in name:
                param.requires_grad = False

    def shared_step(self, batch, random_uncond, **kwargs):
        x, c, fs, gt, mask = self.get_batch_input(batch, random_uncond=random_uncond, return_fs=True)
        kwargs.update({'fs': fs.long()})
        loss, loss_dict = self(x, c, gt, mask, **kwargs)
        return (loss, loss_dict)

    def get_batch_input(self, batch, random_uncond, return_first_stage_outputs=False, return_original_cond=False, return_fs=False, return_cond_frame=False, return_original_input=False, **kwargs):
        x = super().get_input(batch, self.first_stage_key)
        z = self.encode_first_stage(x)
        cond_input = batch[self.cond_stage_key]
        if isinstance(cond_input, dict) or isinstance(cond_input, list):
            cond_emb = self.get_learned_conditioning(cond_input)
        else:
            cond_emb = self.get_learned_conditioning(cond_input.to(self.device))
        cond = {}
        if random_uncond:
            random_num = torch.rand(x.size(0), device=x.device)
        else:
            random_num = torch.ones(x.size(0), device=x.device)
        prompt_mask = rearrange(random_num < 2 * self.uncond_prob, 'n -> n 1 1')
        input_mask = 1 - rearrange((random_num >= self.uncond_prob).float() * (random_num < 3 * self.uncond_prob).float(), 'n -> n 1 1 1')
        null_prompt = self.get_learned_conditioning([''])
        prompt_imb = torch.where(prompt_mask, null_prompt, cond_emb.detach())
        cond_frame_index = 0
        if self.rand_cond_frame:
            cond_frame_index = random.randint(0, self.model.diffusion_model.temporal_length - 1)
        img = x[:, :, cond_frame_index, ...]
        img = input_mask * img
        img_emb = self.embedder(img)
        img_emb = self.image_proj_model(img_emb)
        if self.model.conditioning_key == 'hybrid':
            if self.interp_mode:
                img_cat_cond = torch.zeros_like(z)
                img_cat_cond[:, :, 0, :, :] = z[:, :, 0, :, :]
                img_cat_cond[:, :, -1, :, :] = z[:, :, -1, :, :]
            else:
                img_cat_cond = z[:, :, cond_frame_index, :, :]
                img_cat_cond = img_cat_cond.unsqueeze(2)
                img_cat_cond = repeat(img_cat_cond, 'b c t h w -> b c (repeat t) h w', repeat=z.shape[2])
            cond['c_concat'] = [img_cat_cond]
        cond['c_crossattn'] = [torch.cat([prompt_imb, img_emb], dim=1)]
        out = [z, cond]
        if return_first_stage_outputs:
            xrec = self.decode_first_stage(z)
            out.extend([xrec])
        if return_original_cond:
            out.append(cond_input)
        if return_fs:
            if self.fps_condition_type == 'fs':
                fs = super().get_input(batch, 'frame_stride')
            elif self.fps_condition_type == 'fps':
                fs = super().get_input(batch, 'fps')
            out.append(fs)
        if return_cond_frame:
            out.append(x[:, :, cond_frame_index, ...].unsqueeze(2))
        if return_original_input:
            out.append(x)
        gt = [p for p in batch['gt_ann']]
        gt = torch.tensor(gt).to(out[0].device)
        out.append(gt)
        out.append(batch['mask'])
        return out

    @torch.no_grad()
    def log_images(self, batch, sample=True, ddim_steps=50, ddim_eta=1.0, plot_denoise_rows=False, unconditional_guidance_scale=1.0, mask=None, **kwargs):
        sampled_img_num = 1
        for key in batch.keys():
            batch[key] = batch[key][:sampled_img_num]
        use_ddim = ddim_steps is not None
        log = dict()
        z, c, xrec, xc, fs, cond_x = self.get_batch_input(batch, random_uncond=False, return_first_stage_outputs=True, return_original_cond=True, return_fs=True, return_cond_frame=True)
        N = xrec.shape[0]
        log['image_condition'] = cond_x
        log['reconst'] = xrec
        xc_with_fs = []
        for idx, content in enumerate(xc):
            xc_with_fs.append(content + '_fs=' + str(fs[idx].item()))
        log['condition'] = xc_with_fs
        kwargs.update({'fs': fs.long()})
        c_cat = None
        if sample:
            if unconditional_guidance_scale != 1.0:
                if isinstance(c, dict):
                    c_emb = c['c_crossattn'][0]
                    if 'c_concat' in c.keys():
                        c_cat = c['c_concat'][0]
                else:
                    c_emb = c
                if self.uncond_type == 'empty_seq':
                    prompts = N * ['']
                    uc_prompt = self.get_learned_conditioning(prompts)
                elif self.uncond_type == 'zero_embed':
                    uc_prompt = torch.zeros_like(c_emb)
                img = torch.zeros_like(xrec[:, :, 0])
                img_emb = self.embedder(img)
                uc_img = self.image_proj_model(img_emb)
                uc = torch.cat([uc_prompt, uc_img], dim=1)
                if isinstance(c, dict):
                    uc_hybrid = {'c_concat': [c_cat], 'c_crossattn': [uc]}
                    uc = uc_hybrid
            else:
                uc = None
            with self.ema_scope('Plotting'):
                samples, z_denoise_row = self.sample_log(cond=c, batch_size=N, ddim=use_ddim, ddim_steps=ddim_steps, eta=ddim_eta, unconditional_guidance_scale=unconditional_guidance_scale, unconditional_conditioning=uc, x0=z, **kwargs)
            x_samples = self.decode_first_stage(samples)
            log['samples'] = x_samples
            if plot_denoise_rows:
                denoise_grid = self._get_denoise_row_from_list(z_denoise_row)
                log['denoise_row'] = denoise_grid
        return log

    def configure_optimizers(self):
        lr = self.learning_rate
        params = [p for p in self.parameters() if p.requires_grad]
        mainlogger.info(f'@Training [{len(params)}] Full Paramters.')
        optimizer = torch.optim.AdamW(params, lr=lr)
        if self.use_scheduler:
            mainlogger.info('Setting up scheduler...')
            lr_scheduler = self.configure_schedulers(optimizer)
            return ([optimizer], [lr_scheduler])
        return optimizer

class ImprovedScaleAwareL1Loss(nn.Module):

    def __init__(self, direct_weight=1.0, scale_weight=0.5, value_weight=0.5, eps=1e-08):
        super(ImprovedScaleAwareL1Loss, self).__init__()
        self.l1_loss = nn.L1Loss()
        self.direct_weight = direct_weight
        self.scale_weight = scale_weight
        self.value_weight = value_weight
        self.eps = eps

    def forward(self, pred, gt):
        gt = gt.float()
        pred = pred.float()
        pred = torch.clamp(pred, min=self.eps)
        gt = torch.clamp(gt, min=self.eps)
        direct_loss = self.l1_loss(pred, gt)
        scale_gt = torch.log10(gt)
        scale_pred = torch.log10(pred)
        scale_loss = self.l1_loss(scale_gt, scale_pred)
        norm_gt = gt / 10 ** torch.floor(scale_gt)
        norm_pred = pred / 10 ** torch.floor(scale_pred)
        value_loss = self.l1_loss(norm_gt, norm_pred)
        total_loss = self.direct_weight * direct_loss + self.scale_weight * scale_loss + self.value_weight * value_loss
        return total_loss

class LatentVisualDiffusionv9(LatentVisualDiffusion):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.p_decoder = PDecoder()
        self._freeze_all()

    def forward(self, x, c, gt, mask, **kwargs):
        t = torch.randint(51, 52, (x.shape[0],), device=self.device).long()
        if self.use_dynamic_rescale:
            x = x * extract_into_tensor(self.scale_arr, t, x.shape)
        return self.p_losses(x, c, t, gt, mask, **kwargs)

    def p_losses(self, x_start, cond, t, gt, mask, noise=None, **kwargs):
        if self.noise_strength > 0:
            b, c, f, _, _ = x_start.shape
            offset_noise = torch.randn(b, c, f, 1, 1, device=x_start.device)
            noise = default(noise, lambda: torch.randn_like(x_start) + self.noise_strength * offset_noise)
        else:
            noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        model_output, feat_list = self.apply_model(x_noisy, t, cond, **kwargs)
        feat_dict = {}
        feat_dict[0] = (feat_list[0] + feat_list[1] + feat_list[2] + feat_list[-1] + feat_list[-2] + feat_list[-3]) / 6
        feat_dict[4] = feat_list[-4]
        feat_dict[1] = (feat_list[4] + feat_list[5] + feat_list[-5] + feat_list[-6]) / 4
        feat_dict[5] = feat_list[-7]
        feat_dict[6] = feat_list[3]
        feat_dict[2] = (feat_list[7] + feat_list[8] + feat_list[-8] + feat_list[-9] + feat_list[-10]) / 6
        feat_dict[7] = feat_list[6]
        feat_dict[3] = (feat_list[9] + feat_list[10] + feat_list[11] + feat_list[-11] + feat_list[-12] + feat_list[-13]) / 6
        p_output = self.p_decoder(feat_dict, mask, full_resolution=0)
        l1_loss = nn.L1Loss()
        loss = l1_loss(p_output[:, 0], gt)
        if torch.isnan(loss):
            loss = torch.tensor(0.1, device=loss.device, requires_grad=True)
            print('NaN loss detected, using fallback value')
        loss_dict = {}
        loss_dict.update({'p_l1_loss': loss})
        return (loss, loss_dict)

class LatentVisualDiffusionv41(LatentVisualDiffusionv9):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.p_decoder = PDecoderSupElasticityv1()
        self._freeze_all()

    def shared_step(self, batch, random_uncond, **kwargs):
        x, c, fs, gt, mask, keytime_mask = self.get_batch_input(batch, random_uncond=random_uncond, return_fs=True)
        kwargs.update({'fs': fs.long()})
        loss, loss_dict = self(x, c, gt, mask, keytime_mask, **kwargs)
        return (loss, loss_dict)

    def forward(self, x, c, gt, mask, keytime_mask, **kwargs):
        t = torch.randint(50, 51, (x.shape[0],), device=self.device).long()
        if self.use_dynamic_rescale:
            x = x * extract_into_tensor(self.scale_arr, t, x.shape)
        return self.p_losses(x, c, t, gt, mask, keytime_mask, **kwargs)

    def p_losses(self, x_start, cond, t, gt, mask, keytime_mask, noise=None, **kwargs):
        if self.noise_strength > 0:
            b, c, f, _, _ = x_start.shape
            offset_noise = torch.randn(b, c, f, 1, 1, device=x_start.device)
            noise = default(noise, lambda: torch.randn_like(x_start) + self.noise_strength * offset_noise)
        else:
            noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        model_output, feat_list = self.apply_model(x_noisy, t, cond, **kwargs)
        feat_dict = {}
        feat_dict[0] = (feat_list[0] + feat_list[1] + feat_list[2] + feat_list[-1] + feat_list[-2] + feat_list[-3]) / 6
        feat_dict[4] = feat_list[-4]
        feat_dict[1] = (feat_list[4] + feat_list[5] + feat_list[-5] + feat_list[-6]) / 4
        feat_dict[5] = feat_list[-7]
        feat_dict[6] = feat_list[3]
        feat_dict[2] = (feat_list[7] + feat_list[8] + feat_list[-8] + feat_list[-9] + feat_list[-10]) / 6
        feat_dict[7] = feat_list[6]
        feat_dict[3] = (feat_list[9] + feat_list[10] + feat_list[11] + feat_list[-11] + feat_list[-12] + feat_list[-13]) / 6
        temporal_key_points, height_values, height_differences, normed_height, normed_height_differences, normed_sqrt_height_differences, gt_retitusion, key_masks = self.p_decoder(feat_dict, mask, full_resolution=0)
        l1_loss = nn.L1Loss()
        device = temporal_key_points.device
        for key in gt:
            if isinstance(gt[key], torch.Tensor):
                gt[key] = gt[key].to(device)
            elif isinstance(gt[key], list):
                gt[key] = [item.to(device) if isinstance(item, torch.Tensor) else item for item in gt[key]]
        norm_abs_loss_coef = 0.01
        loss_temporal_key_points = norm_abs_loss_coef * (l1_loss(temporal_key_points[:, 0], gt['temporal_key_points'][0]) + l1_loss(temporal_key_points[:, 1], gt['temporal_key_points'][1]) + l1_loss(temporal_key_points[:, 2], gt['temporal_key_points'][2]))
        loss_height_values = norm_abs_loss_coef * (l1_loss(height_values[:, 0], gt['height_values_of_them'][0]) + l1_loss(height_values[:, 1], gt['height_values_of_them'][1]) + l1_loss(height_values[:, 2], gt['height_values_of_them'][2]))
        loss_height_differences = norm_abs_loss_coef * (l1_loss(height_differences[:, 0], gt['height_differences'][0]) + l1_loss(height_differences[:, 1], gt['height_differences'][1]))
        loss_normed_height = l1_loss(normed_height[:, 0], gt['normed_height'][0]) + l1_loss(normed_height[:, 1], gt['normed_height'][1]) + l1_loss(normed_height[:, 2], gt['normed_height'][2])
        loss_normed_height_differences = l1_loss(normed_height_differences[:, 0], gt['normed_height_differences'][0]) + l1_loss(normed_height_differences[:, 1], gt['normed_height_differences'][1])
        loss_normed_sqrt_height_differences = l1_loss(normed_sqrt_height_differences[:, 0], gt['normed_sqrt_height_differences'][0]) + l1_loss(normed_sqrt_height_differences[:, 1], gt['normed_sqrt_height_differences'][1])
        loss_gt_retitusion = l1_loss(gt_retitusion[:, 0], gt['gt_restitution'])
        print(loss_gt_retitusion)
        keytime_mask = keytime_mask.squeeze(1).float()
        if key_masks.shape != keytime_mask.shape:
            keytime_mask_interpolated = F.interpolate(keytime_mask, size=key_masks.shape[2:], mode='nearest')
            loss_key_masks = F.binary_cross_entropy_with_logits(key_masks, keytime_mask_interpolated)
        else:
            loss_key_masks = F.binary_cross_entropy_with_logits(key_masks, keytime_mask_interpolated)
        extra_loss_coef = 0.1
        loss = extra_loss_coef * (loss_temporal_key_points + loss_height_values + loss_height_differences + loss_normed_height + loss_normed_height_differences + loss_normed_sqrt_height_differences + loss_key_masks) + loss_gt_retitusion
        loss_dict = {}
        loss_dict.update({'total_loss': loss, 'loss_temporal_key_points': loss_temporal_key_points, 'loss_height_values': loss_height_values, 'loss_height_differences': loss_height_differences, 'loss_normed_height': loss_normed_height, 'loss_normed_height_differences': loss_normed_height_differences, 'loss_normed_sqrt_height_differences': loss_normed_sqrt_height_differences, 'loss_key_masks': loss_key_masks, 'loss_gt_retitusion': loss_gt_retitusion})
        if torch.isnan(loss):
            loss = torch.tensor(0.1, device=loss.device, requires_grad=True)
            print('NaN loss detected, using fallback value')
        for key_i in loss_dict.keys():
            if torch.isnan(loss_dict[key_i]):
                loss_dict[key_i] = torch.tensor(0.001, device=loss.device, requires_grad=True)
                print('NaN loss detected, using fallback value')
        return (loss, loss_dict)

    def get_batch_input(self, batch, random_uncond, return_first_stage_outputs=False, return_original_cond=False, return_fs=False, return_cond_frame=False, return_original_input=False, **kwargs):
        x = super().get_input(batch, self.first_stage_key)
        z = self.encode_first_stage(x)
        cond_input = batch[self.cond_stage_key]
        if isinstance(cond_input, dict) or isinstance(cond_input, list):
            cond_emb = self.get_learned_conditioning(cond_input)
        else:
            cond_emb = self.get_learned_conditioning(cond_input.to(self.device))
        cond = {}
        if random_uncond:
            random_num = torch.rand(x.size(0), device=x.device)
        else:
            random_num = torch.ones(x.size(0), device=x.device)
        prompt_mask = rearrange(random_num < 2 * self.uncond_prob, 'n -> n 1 1')
        input_mask = 1 - rearrange((random_num >= self.uncond_prob).float() * (random_num < 3 * self.uncond_prob).float(), 'n -> n 1 1 1')
        null_prompt = self.get_learned_conditioning([''])
        prompt_imb = torch.where(prompt_mask, null_prompt, cond_emb.detach())
        cond_frame_index = 0
        if self.rand_cond_frame:
            cond_frame_index = random.randint(0, self.model.diffusion_model.temporal_length - 1)
        img = x[:, :, cond_frame_index, ...]
        img = input_mask * img
        img_emb = self.embedder(img)
        img_emb = self.image_proj_model(img_emb)
        if self.model.conditioning_key == 'hybrid':
            if self.interp_mode:
                img_cat_cond = torch.zeros_like(z)
                img_cat_cond[:, :, 0, :, :] = z[:, :, 0, :, :]
                img_cat_cond[:, :, -1, :, :] = z[:, :, -1, :, :]
            else:
                img_cat_cond = z[:, :, cond_frame_index, :, :]
                img_cat_cond = img_cat_cond.unsqueeze(2)
                img_cat_cond = repeat(img_cat_cond, 'b c t h w -> b c (repeat t) h w', repeat=z.shape[2])
            cond['c_concat'] = [img_cat_cond]
        cond['c_crossattn'] = [torch.cat([prompt_imb, img_emb], dim=1)]
        out = [z, cond]
        if return_first_stage_outputs:
            xrec = self.decode_first_stage(z)
            out.extend([xrec])
        if return_original_cond:
            out.append(cond_input)
        if return_fs:
            if self.fps_condition_type == 'fs':
                fs = super().get_input(batch, 'frame_stride')
            elif self.fps_condition_type == 'fps':
                fs = super().get_input(batch, 'fps')
            out.append(fs)
        if return_cond_frame:
            out.append(x[:, :, cond_frame_index, ...].unsqueeze(2))
        if return_original_input:
            out.append(x)
        gt = batch['gt_ann']
        out.append(gt)
        out.append(batch['mask'])
        out.append(batch['keytime_mask'])
        return out

class LatentVisualDiffusionv43(LatentVisualDiffusionv41):

    def __init__(self, *args, ft_weights_list=None, ft_lr=1e-05, **kwargs):
        super().__init__(*args, **kwargs)
        self.p_decoder = PDecoderSupElasticityv1()
        self.ft_weights_list = ft_weights_list
        self.ft_lr = ft_lr
        self._freeze_all_but_some(self.ft_weights_list)

    def _freeze_all_but_some(self, ft_weights_list):
        for name, param in self.named_parameters():
            if 'p_decoder' not in name:
                param.requires_grad = False
        for ft_weight in ft_weights_list:
            if any((c in ft_weight for c in '.*+?^$[](){}|\\')):
                pattern = re.compile(ft_weight.replace('.', '\\.').replace('*', '.*'))
                for name, param in self.named_parameters():
                    if pattern.search(name):
                        param.requires_grad = True
            else:
                for name, param in self.named_parameters():
                    if ft_weight in name:
                        param.requires_grad = True

    def configure_optimizers(self):
        lr = self.learning_rate
        params_1 = [p for n, p in self.named_parameters() if p.requires_grad and 'p_decoder' in n]
        params_2 = [p for n, p in self.named_parameters() if p.requires_grad and 'p_decoder' not in n]
        mainlogger.info(f'@Training [{len(params_1)}] Full Paramters.')
        mainlogger.info(f'@Fine-tuning [{len(params_2)}] Full Paramters.')
        optimizer = torch.optim.AdamW([{'params': params_1, 'lr': lr}, {'params': params_2, 'lr': self.ft_lr}])
        if self.use_scheduler:
            ipdb.set_trace()
            mainlogger.info('Setting up scheduler...')
            lr_scheduler = self.configure_schedulers(optimizer)
            return ([optimizer], [lr_scheduler])
        return optimizer

class LatentVisualDiffusionv47(LatentVisualDiffusionv43):

    def __init__(self, *args, extra_loss_coef=0.1, norm_abs_loss_coef=0.01, pd_config=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.extra_loss_coef = extra_loss_coef
        self.norm_abs_loss_coef = norm_abs_loss_coef
        self.pd_config = pd_config
        self.p_decoder = PDecoderSupElasticityv2(self.pd_config.temporal_key_points, self.pd_config.height_values, self.pd_config.height_differences, self.pd_config.normed_height, self.pd_config.normed_height_differences, self.pd_config.normed_sqrt_height_differences, self.pd_config.keytime_masks, getattr(self.pd_config, 'full_resolution', 0))
        self._freeze_all_but_some(self.ft_weights_list)

    def p_losses(self, x_start, cond, t, gt, mask, keytime_mask, noise=None, **kwargs):
        if self.noise_strength > 0:
            b, c, f, _, _ = x_start.shape
            offset_noise = torch.randn(b, c, f, 1, 1, device=x_start.device)
            noise = default(noise, lambda: torch.randn_like(x_start) + self.noise_strength * offset_noise)
        else:
            noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        model_output, feat_list = self.apply_model(x_noisy, t, cond, **kwargs)
        feat_dict = {}
        feat_dict[0] = (feat_list[0] + feat_list[1] + feat_list[2] + feat_list[-1] + feat_list[-2] + feat_list[-3]) / 6
        feat_dict[4] = feat_list[-4]
        feat_dict[1] = (feat_list[4] + feat_list[5] + feat_list[-5] + feat_list[-6]) / 4
        feat_dict[5] = feat_list[-7]
        feat_dict[6] = feat_list[3]
        feat_dict[2] = (feat_list[7] + feat_list[8] + feat_list[-8] + feat_list[-9] + feat_list[-10]) / 6
        feat_dict[7] = feat_list[6]
        feat_dict[3] = (feat_list[9] + feat_list[10] + feat_list[11] + feat_list[-11] + feat_list[-12] + feat_list[-13]) / 6
        temporal_key_points, height_values, height_differences, normed_height, normed_height_differences, normed_sqrt_height_differences, gt_retitusion, key_masks = self.p_decoder(feat_dict, mask)
        l1_loss = nn.L1Loss()
        device = feat_dict[0].device
        for key in gt:
            if isinstance(gt[key], torch.Tensor):
                gt[key] = gt[key].to(device)
            elif isinstance(gt[key], list):
                gt[key] = [item.to(device) if isinstance(item, torch.Tensor) else item for item in gt[key]]
        loss_temporal_key_points = 0 if temporal_key_points is None else self.norm_abs_loss_coef * (l1_loss(temporal_key_points[:, 0], gt['temporal_key_points'][0]) + l1_loss(temporal_key_points[:, 1], gt['temporal_key_points'][1]) + l1_loss(temporal_key_points[:, 2], gt['temporal_key_points'][2]))
        loss_height_values = 0 if height_values is None else self.norm_abs_loss_coef * (l1_loss(height_values[:, 0], gt['height_values_of_them'][0]) + l1_loss(height_values[:, 1], gt['height_values_of_them'][1]) + l1_loss(height_values[:, 2], gt['height_values_of_them'][2]))
        loss_height_differences = 0 if height_differences is None else self.norm_abs_loss_coef * (l1_loss(height_differences[:, 0], gt['height_differences'][0]) + l1_loss(height_differences[:, 1], gt['height_differences'][1]))
        loss_normed_height = 0 if normed_height is None else l1_loss(normed_height[:, 0], gt['normed_height'][0]) + l1_loss(normed_height[:, 1], gt['normed_height'][1]) + l1_loss(normed_height[:, 2], gt['normed_height'][2])
        loss_normed_height_differences = 0 if normed_height_differences is None else l1_loss(normed_height_differences[:, 0], gt['normed_height_differences'][0]) + l1_loss(normed_height_differences[:, 1], gt['normed_height_differences'][1])
        loss_normed_sqrt_height_differences = 0 if normed_sqrt_height_differences is None else l1_loss(normed_sqrt_height_differences[:, 0], gt['normed_sqrt_height_differences'][0]) + l1_loss(normed_sqrt_height_differences[:, 1], gt['normed_sqrt_height_differences'][1])
        loss_gt_retitusion = 0 if gt_retitusion is None else l1_loss(gt_retitusion[:, 0], gt['gt_restitution'])
        if key_masks is None:
            loss_key_masks = 0
        else:
            keytime_mask = keytime_mask.squeeze(1).float()
            if key_masks.shape != keytime_mask.shape:
                keytime_mask_interpolated = F.interpolate(keytime_mask, size=key_masks.shape[2:], mode='nearest')
                loss_key_masks = F.binary_cross_entropy_with_logits(key_masks, keytime_mask_interpolated)
            else:
                loss_key_masks = F.binary_cross_entropy_with_logits(key_masks, keytime_mask_interpolated)
        if gt_retitusion is not None:
            print(loss_gt_retitusion)
        else:
            if temporal_key_points is not None:
                print('loss_temporal_key_points: ', loss_temporal_key_points)
            if height_values is not None:
                print('loss_height_values: ', loss_height_values)
            if height_differences is not None:
                print('loss_height_differences: ', loss_height_differences)
            if normed_height is not None:
                print('loss_normed_height: ', loss_normed_height)
            if normed_height_differences is not None:
                print('loss_normed_height_differences: ', loss_normed_height_differences)
            if normed_sqrt_height_differences is not None:
                print('loss_normed_sqrt_height_differences: ', loss_normed_sqrt_height_differences)
            if key_masks is not None:
                print('loss_key_masks: ', loss_key_masks)
        loss = self.extra_loss_coef * (loss_temporal_key_points + loss_height_values + loss_height_differences + loss_normed_height + loss_normed_height_differences + loss_normed_sqrt_height_differences + loss_key_masks) + loss_gt_retitusion
        loss_dict = {}
        loss_dict.update({'total_loss': loss, 'loss_temporal_key_points': loss_temporal_key_points, 'loss_height_values': loss_height_values, 'loss_height_differences': loss_height_differences, 'loss_normed_height': loss_normed_height, 'loss_normed_height_differences': loss_normed_height_differences, 'loss_normed_sqrt_height_differences': loss_normed_sqrt_height_differences, 'loss_key_masks': loss_key_masks, 'loss_gt_retitusion': loss_gt_retitusion})
        for key_i in loss_dict.keys():
            if isinstance(loss_dict[key_i], torch.Tensor) and torch.isnan(loss_dict[key_i]):
                loss_dict[key_i] = torch.tensor(0.001, device=device, requires_grad=True)
                print('NaN loss detected, using fallback value')
        return (loss, loss_dict)

class LatentVisualDiffusionv84(LatentVisualDiffusionv47):

    def __init__(self, *args, prob3_dim=0, major_specify=0, cond_specify=0, collect_map_together=0, concat_2_feat=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.concat_2_feat = concat_2_feat
        self.collect_map_together = collect_map_together
        self.major_specify = major_specify
        self.cond_specify = cond_specify
        self.prob3_dim = prob3_dim
        self.prob3_learnable_vector = nn.Parameter(torch.randn((1, self.prob3_dim)) * 0.02)
        self.p_decoder = None
        feat_dims = [320, 320, 320, 320, 640, 640, 640, 1280, 1280, 1280, 1280, 1280, 1280, 1280, 1280, 1280, 1280, 1280, 1280, 640, 640, 640, 320, 320, 320]
        self.prob3_feat_mlps = nn.ModuleList([nn.Linear(self.prob3_dim, feat_dim) for feat_dim in feat_dims])
        self.prob3_feat_output_mlps = nn.ModuleList([nn.Linear(feat_dim, self.prob3_dim) for feat_dim in feat_dims])
        self.prob3_output_projection = nn.Linear(self.prob3_dim, 1)
        self.prob3_learnable_prompt = nn.Parameter(torch.randn((1, 77, 1024)) * 0.02)
        self._freeze_all_but_some(self.ft_weights_list)

    def _freeze_all_but_some(self, ft_weights_list):
        for name, param in self.named_parameters():
            if 'prob3' not in name:
                param.requires_grad = False
            if 'prob3' in name:
                param.requires_grad = True
        for ft_weight in ft_weights_list:
            if any((c in ft_weight for c in '.*+?^$[](){}|\\')):
                pattern = re.compile(ft_weight.replace('.', '\\.').replace('*', '.*'))
                for name, param in self.named_parameters():
                    if pattern.search(name):
                        param.requires_grad = True
            else:
                for name, param in self.named_parameters():
                    if ft_weight in name:
                        param.requires_grad = True

    def configure_optimizers(self):
        lr = self.learning_rate
        params_1 = [p for n, p in self.named_parameters() if p.requires_grad and ('prob3' in n or 'p_decoder' in n)]
        for n, p in self.named_parameters():
            if p.requires_grad:
                print(n)
        mainlogger.info(f'@Training [{len(params_1)}] Full Paramters.')
        optimizer = torch.optim.AdamW([{'params': params_1, 'lr': lr}])
        if self.use_scheduler:
            ipdb.set_trace()
            mainlogger.info('Setting up scheduler...')
            lr_scheduler = self.configure_schedulers(optimizer)
            return ([optimizer], [lr_scheduler])
        return optimizer

    def p_losses(self, x_start, cond, t, gt, mask, keytime_mask, noise=None, **kwargs):
        if self.noise_strength > 0:
            b, c, f, _, _ = x_start.shape
            offset_noise = torch.randn(b, c, f, 1, 1, device=x_start.device)
            noise = default(noise, lambda: torch.randn_like(x_start) + self.noise_strength * offset_noise)
        else:
            noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        cond['c_crossattn'][0][:, :77, :] = self.prob3_learnable_prompt
        model_output, feat_list = self.apply_model(x_noisy, t, cond, **kwargs)
        prob3_outputs = []
        for feat_i, feat in enumerate(feat_list):
            B, T, C, H, W = feat.shape
            query_vector = self.prob3_feat_mlps[feat_i](self.prob3_learnable_vector)
            query_vector = query_vector.expand(B, -1).unsqueeze(1)
            feat_reshaped = feat.permute(0, 1, 3, 4, 2).contiguous().view(B, T * H * W, C)
            scale = C ** (-0.5)
            attn_scores = torch.matmul(query_vector, feat_reshaped.transpose(-2, -1)) * scale
            attn_weights = torch.softmax(attn_scores, dim=-1)
            attended_feat = torch.matmul(attn_weights, feat_reshaped)
            attended_feat = attended_feat.squeeze(1)
            prob3_feat_output = self.prob3_feat_output_mlps[feat_i](attended_feat)
            prob3_outputs.append(prob3_feat_output)
        if prob3_outputs:
            prob3_combined = torch.stack(prob3_outputs, dim=1)
            prob3_averaged = torch.mean(prob3_combined, dim=1)
            gt_retitusion = self.prob3_output_projection(prob3_averaged)
        else:
            gt_retitusion = None
        l1_loss = nn.L1Loss()
        device = feat_list[0].device
        for key in gt:
            if isinstance(gt[key], torch.Tensor):
                gt[key] = gt[key].to(device)
            elif isinstance(gt[key], list):
                gt[key] = [item.to(device) if isinstance(item, torch.Tensor) else item for item in gt[key]]
        loss_gt_retitusion = 0 if gt_retitusion is None else l1_loss(gt_retitusion[:, 0], gt['gt_restitution'])
        if gt_retitusion is not None:
            print(loss_gt_retitusion)
        loss = loss_gt_retitusion
        loss_dict = {}
        loss_dict.update({'total_loss': loss, 'loss_gt_retitusion': loss_gt_retitusion})
        for key_i in loss_dict.keys():
            if isinstance(loss_dict[key_i], torch.Tensor) and torch.isnan(loss_dict[key_i]):
                loss_dict[key_i] = torch.tensor(0.001, device=device, requires_grad=True)
                print('NaN loss detected, using fallback value')
        return (loss, loss_dict)

    def get_batch_input(self, batch, random_uncond, return_first_stage_outputs=False, return_original_cond=False, return_fs=False, return_cond_frame=False, return_original_input=False, **kwargs):
        x = super().get_input(batch, self.first_stage_key)
        if self.major_specify == 0:
            z = self.encode_first_stage(x)
        elif self.major_specify == 1:
            cur_batch_mask = batch['mask']
            z = self.encode_first_stage(x * cur_batch_mask)
        elif self.major_specify == 2:
            x_red = batch['mask']
            z = self.encode_first_stage(x_red)
        elif self.major_specify == 3:
            x_red = batch['mask']
            x_concat_new = torch.cat([x[:, :, :-1, :, :], x_red[:, :, -1:, :, :]], dim=2)
            z = self.encode_first_stage(x_concat_new)
        cond_input = batch[self.cond_stage_key]
        cond_input = cond_input
        if isinstance(cond_input, dict) or isinstance(cond_input, list):
            cond_emb = self.get_learned_conditioning(cond_input)
        else:
            cond_emb = self.get_learned_conditioning(cond_input.to(self.device))
        cond = {}
        if random_uncond:
            random_num = torch.rand(x.size(0), device=x.device)
        else:
            random_num = torch.ones(x.size(0), device=x.device)
        prompt_mask = rearrange(random_num < 2 * self.uncond_prob, 'n -> n 1 1')
        input_mask = 1 - rearrange((random_num >= self.uncond_prob).float() * (random_num < 3 * self.uncond_prob).float(), 'n -> n 1 1 1')
        null_prompt = self.get_learned_conditioning([''])
        prompt_imb = torch.where(prompt_mask, null_prompt, cond_emb.detach())
        cond_frame_index = 0
        if self.rand_cond_frame:
            cond_frame_index = random.randint(0, self.model.diffusion_model.temporal_length - 1)
        if self.cond_specify == 3:
            cond_frame_index = -1
        if self.cond_specify == 0:
            img = x[:, :, cond_frame_index, ...]
        elif self.cond_specify == 1:
            img = x[:, :, cond_frame_index, ...]
            cur_batch_mask = batch['mask']
            img = img * cur_batch_mask[:, :, cond_frame_index, ...]
        elif self.cond_specify == 2:
            cur_batch_mask = batch['mask']
            img = cur_batch_mask[:, :, cond_frame_index, ...]
        elif self.cond_specify == 3:
            cur_batch_mask = batch['mask']
            img = cur_batch_mask[:, :, cond_frame_index, ...]
        img = input_mask * img
        img_emb = self.embedder(img)
        img_emb = self.image_proj_model(img_emb)
        if self.model.conditioning_key == 'hybrid':
            if self.interp_mode:
                img_cat_cond = torch.zeros_like(z)
                img_cat_cond[:, :, 0, :, :] = z[:, :, 0, :, :]
                img_cat_cond[:, :, -1, :, :] = z[:, :, -1, :, :]
            elif self.cond_specify == 0 or self.cond_specify == 3:
                img_cat_cond = z[:, :, cond_frame_index, :, :]
                img_cat_cond = img_cat_cond.unsqueeze(2)
                img_cat_cond = repeat(img_cat_cond, 'b c t h w -> b c (repeat t) h w', repeat=z.shape[2])
            elif self.cond_specify == 1:
                cur_batch_mask = batch['mask']
                img_cat_cond = self.encode_first_stage(x * cur_batch_mask)
            elif self.cond_specify == 2:
                cur_batch_mask = batch['mask']
                img_cat_cond = self.encode_first_stage(cur_batch_mask)
            cond['c_concat'] = [img_cat_cond]
        cond['c_crossattn'] = [torch.cat([prompt_imb, img_emb], dim=1)]
        out = [z, cond]
        if return_first_stage_outputs:
            xrec = self.decode_first_stage(z)
            out.extend([xrec])
        if return_original_cond:
            out.append(cond_input)
        if return_fs:
            if self.fps_condition_type == 'fs':
                fs = super().get_input(batch, 'frame_stride')
            elif self.fps_condition_type == 'fps':
                fs = super().get_input(batch, 'fps')
            out.append(fs)
        if return_cond_frame:
            out.append(x[:, :, cond_frame_index, ...].unsqueeze(2))
        if return_original_input:
            out.append(x)
        gt = batch['gt_ann']
        out.append(gt)
        out.append(batch['mask'])
        out.append(batch['keytime_mask'])
        return out

    def calculate_physics_loss(self, feat_list, gt):
        device = feat_list[0].device
        prob3_outputs = []
        for feat_i, feat in enumerate(feat_list):
            B, T, C, H, W = feat.shape
            query_vector = self.prob3_feat_mlps[feat_i](self.prob3_learnable_vector)
            query_vector = query_vector.expand(B, -1).unsqueeze(1)
            feat_reshaped = feat.permute(0, 1, 3, 4, 2).contiguous().view(B, T * H * W, C)
            scale = C ** (-0.5)
            attn_scores = torch.matmul(query_vector, feat_reshaped.transpose(-2, -1)) * scale
            attn_weights = torch.softmax(attn_scores, dim=-1)
            attended_feat = torch.matmul(attn_weights, feat_reshaped).squeeze(1)
            prob3_feat_output = self.prob3_feat_output_mlps[feat_i](attended_feat)
            prob3_outputs.append(prob3_feat_output)
        if prob3_outputs:
            prob3_combined = torch.stack(prob3_outputs, dim=1)
            prob3_averaged = torch.mean(prob3_combined, dim=1)
            gt_retitusion = self.prob3_output_projection(prob3_averaged)
        else:
            gt_retitusion = None
        l1_loss = nn.L1Loss()
        for key in gt:
            if isinstance(gt[key], torch.Tensor):
                gt[key] = gt[key].to(device)
            elif isinstance(gt[key], list):
                gt[key] = [item.to(device) if isinstance(item, torch.Tensor) else item for item in gt[key]]
        loss_gt_retitusion = 0 if gt_retitusion is None else l1_loss(gt_retitusion[:, 0], gt['gt_restitution'])
        if gt_retitusion is not None:
            print(loss_gt_retitusion)
        loss = loss_gt_retitusion
        loss_dict = {'total_loss': loss, 'loss_gt_retitusion': loss_gt_retitusion}
        for key_i in loss_dict.keys():
            if isinstance(loss_dict[key_i], torch.Tensor) and torch.isnan(loss_dict[key_i]):
                loss_dict[key_i] = torch.tensor(0.001, device=device, requires_grad=True)
                print('NaN loss detected, using fallback value')
        return (loss, loss_dict)

    def prepare_conditions(self, batch, x, random_uncond):
        cond_input = batch[self.cond_stage_key]
        if isinstance(cond_input, dict) or isinstance(cond_input, list):
            cond_emb = self.get_learned_conditioning(cond_input)
        else:
            cond_emb = self.get_learned_conditioning(cond_input.to(self.device))
        cond = {}
        if random_uncond:
            random_num = torch.rand(x.size(0), device=x.device)
        else:
            random_num = torch.ones(x.size(0), device=x.device)
        prompt_mask = rearrange(random_num < 2 * self.uncond_prob, 'n -> n 1 1')
        input_mask = 1 - rearrange((random_num >= self.uncond_prob).float() * (random_num < 3 * self.uncond_prob).float(), 'n -> n 1 1 1')
        null_prompt = self.get_learned_conditioning([''])
        prompt_imb = torch.where(prompt_mask, null_prompt, cond_emb.detach())
        cond_frame_index = 0
        if self.rand_cond_frame:
            cond_frame_index = random.randint(0, self.model.diffusion_model.temporal_length - 1)
        if self.cond_specify == 3:
            cond_frame_index = -1
        if self.cond_specify == 0:
            img = x[:, :, cond_frame_index, ...]
        elif self.cond_specify == 1:
            img = x[:, :, cond_frame_index, ...]
            cur_batch_mask = batch['mask']
            img = img * cur_batch_mask[:, :, cond_frame_index, ...]
        elif self.cond_specify == 2:
            cur_batch_mask = batch['mask']
            img = cur_batch_mask[:, :, cond_frame_index, ...]
        elif self.cond_specify == 3:
            cur_batch_mask = batch['mask']
            img = cur_batch_mask[:, :, cond_frame_index, ...]
        img = input_mask * img
        img_emb = self.embedder(img)
        img_emb = self.image_proj_model(img_emb)
        cond['c_crossattn'] = [torch.cat([prompt_imb, img_emb], dim=1)]
        return cond
class PhysicsPredictionVJEPA(LatentVisualDiffusionv84):

    def __init__(self, *args, vjepa_feat_dim=768, **kwargs):
        super().__init__(*args, **kwargs)
        self.parameterization = 'vjepa_direct'
        self.noise_strength = 0
        self.l1_loss = nn.L1Loss()
        mainlogger.info(f'Re-initializing MLPs for PhysicsPredictionVJEPA to match v84 structure with uniform dimensions.')
        num_features = self.model.diffusion_model.num_output_features
        uniform_feat_dim = self.model.diffusion_model.vjepa_hidden_dim
        self.prob3_feat_mlps = nn.ModuleList([nn.Linear(self.prob3_dim, uniform_feat_dim) for _ in range(num_features)])
        self.prob3_feat_output_mlps = nn.ModuleList([nn.Linear(uniform_feat_dim, self.prob3_dim) for _ in range(num_features)])
        self.prob3_output_projection = nn.Linear(self.prob3_dim, 1)
        self.prob3_feat_mlp = None
        self.prob3_feat_output_mlp = None

    def get_batch_input(self, batch, random_uncond, **kwargs):
        x = batch[self.first_stage_key]
        cond = self.prepare_conditions(batch, x, random_uncond)
        gt = batch['gt_ann']
        mask = batch['mask']
        keytime_mask = batch['keytime_mask']
        if 'c_concat' in cond:
            del cond['c_concat']
        out = [x, cond, gt, mask, keytime_mask]
        if kwargs.get('return_fs', False):
            if self.fps_condition_type == 'fs':
                fs = super(LatentVisualDiffusion, self).get_input(batch, 'frame_stride')
            elif self.fps_condition_type == 'fps':
                fs = super(LatentVisualDiffusion, self).get_input(batch, 'fps')
            out.append(fs)
        if kwargs.get('return_original_input', False):
            out.append(x)
        if 'fns' in inspect.signature(self.forward).parameters or kwargs.get('return_fns', False):
            fn_list = [p.split('/')[-1] for p in batch['path']]
            out.append(fn_list)
        return out

    def p_losses(self, video_input, cond, t, gt, mask, keytime_mask, noise=None, **kwargs):
        _, feat_list = self.apply_model(video_input, t, cond, **kwargs)
        loss, loss_dict = self.calculate_physics_loss(feat_list, gt)
        return (loss, loss_dict)

    def forward(self, x, c, gt, mask, keytime_mask, **kwargs):
        t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=self.device).long()
        return self.p_losses(x, c, t, gt, mask, keytime_mask, **kwargs)

    def shared_step(self, batch, random_uncond, **kwargs):
        if 'fns' in inspect.signature(self.forward).parameters:
            x, c, gt, mask, keytime_mask, fns = self.get_batch_input(batch, random_uncond=random_uncond, return_fns=True)
            loss, loss_dict = self(x, c, gt, mask, keytime_mask, fns=fns, **kwargs)
        else:
            x, c, gt, mask, keytime_mask = self.get_batch_input(batch, random_uncond=random_uncond)
            loss, loss_dict = self(x, c, gt, mask, keytime_mask, **kwargs)
        return (loss, loss_dict)

class PhysicsPredictionVJEPA_Base(LatentVisualDiffusionv84):

    def __init__(self, *args, **kwargs):
        kwargs['use_ema'] = False
        self.use_red_circle = kwargs.pop('use_red_circle', False)
        super().__init__(*args, **kwargs)
        self.linear_start = kwargs.get('linear_start', 0.00085)
        self.linear_end = kwargs.get('linear_end', 0.012)
        self.cosine_s = kwargs.get('cosine_s', 0.008)
        self.num_timesteps = kwargs.get('num_timesteps', 1000)
        self.parameterization = 'vjepa_direct'
        self.noise_strength = 0
        self.l1_loss = nn.L1Loss()
        mainlogger.info(f'Initializing PhysicsPredictionVJEPA_Base: Setting up VJEPA-specific components.')
        if not hasattr(self.model.diffusion_model, 'num_output_features') or not hasattr(self.model.diffusion_model, 'vjepa_hidden_dim'):
            raise TypeError("The underlying model for PhysicsPredictionVJEPA_Base must be VJEPA2_Direct_Wrapper with 'num_output_features' and 'vjepa_hidden_dim' attributes.")
        num_features = self.model.diffusion_model.num_output_features
        uniform_feat_dim = self.model.diffusion_model.vjepa_hidden_dim
        self.prob3_feat_mlps = nn.ModuleList([nn.Linear(self.prob3_dim, uniform_feat_dim) for _ in range(num_features)])
        self.prob3_feat_output_mlps = nn.ModuleList([nn.Linear(uniform_feat_dim, self.prob3_dim) for _ in range(num_features)])
        self.prob3_output_projection = nn.Linear(self.prob3_dim, 1)
        self.prob3_feat_mlp = None
        self.prob3_feat_output_mlp = None

    def register_schedule(self, *args, **kwargs):
        mainlogger.info('PhysicsPredictionVJEPA_Base: Skipping diffusion schedule registration.')
        self.num_timesteps = kwargs.get('num_timesteps', 1000)
        pass

    def get_batch_input(self, batch, random_uncond, **kwargs):
        x = batch[self.first_stage_key]
        cond = self.prepare_conditions(batch, x, random_uncond)
        gt = batch.get('gt_ann', {})
        mask = batch.get('mask')
        keytime_mask = batch.get('keytime_mask')
        if 'c_concat' in cond:
            del cond['c_concat']
        out = [x, cond, gt, mask, keytime_mask]
        if self.use_red_circle:
            out = [mask, cond, gt, mask, keytime_mask]
            print('use red circle video')
        if kwargs.get('return_fs', False):
            fs_key = 'frame_stride' if self.fps_condition_type == 'fs' else 'fps'
            fs = super(LatentVisualDiffusion, self).get_input(batch, fs_key)
            out.append(fs)
        if 'fns' in inspect.signature(self.forward).parameters or kwargs.get('return_fns', False):
            fn_list = [p.split('/')[-1] for p in batch['path']]
            out.append(fn_list)
        return out

    def forward(self, x, c, gt, mask, keytime_mask, **kwargs):
        t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=self.device).long()
        return self.p_losses(x, c, t, gt, mask, keytime_mask, **kwargs)

    def shared_step(self, batch, random_uncond, **kwargs):
        if 'fns' in inspect.signature(self.forward).parameters:
            x, c, gt, mask, keytime_mask, fns = self.get_batch_input(batch, random_uncond=random_uncond, return_fns=True)
            loss, loss_dict = self(x, c, gt, mask, keytime_mask, fns=fns, **kwargs)
        else:
            x, c, gt, mask, keytime_mask = self.get_batch_input(batch, random_uncond=random_uncond)
            loss, loss_dict = self(x, c, gt, mask, keytime_mask, **kwargs)
        return (loss, loss_dict)

class PhysicsPredictionVJEPA_Base_probe(PhysicsPredictionVJEPA_Base):

    def __init__(self, attentive_config: dict=None, direct_weight: float=1.0, scale_weight: float=0.5, value_weight: float=0.0, vjepa_hidden_dim: int=1024, num_output_features: int=25, use_all_feat_in_attentive_probe: bool=False, *args, **kwargs):
        self.direct_weight = direct_weight
        self.scale_weight = scale_weight
        self.value_weight = value_weight
        self.vjepa_hidden_dim = vjepa_hidden_dim
        self.num_output_features = num_output_features
        self.use_all_feat_in_attentive_probe = use_all_feat_in_attentive_probe
        kwargs['use_ema'] = False
        super().__init__(*args, **kwargs)
        self.linear_start = kwargs.get('linear_start', 0.00085)
        self.linear_end = kwargs.get('linear_end', 0.012)
        self.cosine_s = kwargs.get('cosine_s', 0.008)
        self.num_timesteps = kwargs.get('num_timesteps', 1000)
        self.parameterization = 'vjepa_direct'
        self.noise_strength = 0
        self.l1_loss = nn.L1Loss()
        if self.use_all_feat_in_attentive_probe:
            self.prob3_feat_mlps = nn.ModuleList([nn.Linear(self.vjepa_hidden_dim, self.vjepa_hidden_dim) for _ in range(self.num_output_features)])
        else:
            self.prob3_feat_mlps = nn.ModuleList([nn.Linear(self.prob3_dim, self.vjepa_hidden_dim) for _ in range(self.num_output_features)])
        mainlogger.info(f'Initializing PhysicsPredictionVJEPA_Base: Setting up VJEPA-specific components.')
        if not hasattr(self.model.diffusion_model, 'num_output_features') or not hasattr(self.model.diffusion_model, 'vjepa_hidden_dim'):
            raise TypeError("The underlying model for PhysicsPredictionVJEPA_Base must be VJEPA2_Direct_Wrapper with 'num_output_features' and 'vjepa_hidden_dim' attributes.")
        num_features = self.num_output_features
        uniform_feat_dim = self.vjepa_hidden_dim
        self.l1_loss = nn.L1Loss()
        self.bce_loss = nn.BCEWithLogitsLoss()
        default_attentive_config = {'verb_classes': {}, 'noun_classes': {}, 'action_classes': {f'restitution_{i}': i for i in range(2)}, 'embed_dim': uniform_feat_dim, 'num_heads': 8, 'depth': 2, 'use_activation_checkpointing': False}
        if attentive_config:
            default_attentive_config.update(attentive_config)
        self.attentive_classifier = AttentiveClassifier(**default_attentive_config)
        mainlogger.info('PhysicsPredictionVJEPA_Base_probe: using AttentiveClassifier instead of MLP probing.')

    def calculate_vjepa_physics_loss(self, feat_list: list, gt: dict, return_prediction: bool=False) -> tuple:
        if feat_list is None:
            device = self.device
        else:
            device = feat_list.device
        for key in gt:
            if isinstance(gt[key], torch.Tensor):
                gt[key] = gt[key].to(device)
            elif isinstance(gt[key], list):
                gt[key] = [item.to(device) if isinstance(item, torch.Tensor) else item for item in gt[key]]
        l1_loss = nn.L1Loss()
        if feat_list is None:
            gt_retitusion = None
        else:
            last_layer_feat = feat_list
            classifier_output = self.attentive_classifier(last_layer_feat)
            action_logits = classifier_output['action']
            gt_retitusion = action_logits
        if gt_retitusion is None:
            loss_gt_retitusion = torch.tensor(0.0, device=device, requires_grad=True)
        elif gt_retitusion.shape[1] > 1:
            loss_gt_retitusion = l1_loss(gt_retitusion[:, 0], gt['gt_restitution'].float())
        else:
            loss_gt_retitusion = l1_loss(gt_retitusion.squeeze(), gt['gt_restitution'].float())
        loss = loss_gt_retitusion
        loss_dict = {'total_loss': loss, 'loss_gt_retitusion': loss_gt_retitusion}
        for key_i in loss_dict.keys():
            if isinstance(loss_dict[key_i], torch.Tensor) and (torch.isnan(loss_dict[key_i]) or torch.isinf(loss_dict[key_i])):
                mainlogger.info(f'NaN or Inf loss detected in {key_i}, using fallback value')
                loss_dict[key_i] = torch.tensor(0.001, device=device, requires_grad=True)
        if return_prediction:
            return (loss, loss_dict, gt_retitusion)
        return (loss, loss_dict)

    def shared_step(self, batch: dict, **kwargs) -> tuple[torch.Tensor, dict]:
        x, cond, gt, mask, keytime_mask = self.get_batch_input(batch, random_uncond=False)
        t = torch.zeros(x.shape[0], device=self.device, dtype=torch.long)
        model_output, feat_list = self.model.diffusion_model(x, t=t, context=None)
        if self.use_all_feat_in_attentive_probe:
            if feat_list and len(feat_list) > 0:
                target_shape = model_output.shape
                B, N_patches, D = target_shape
                transformed_feats = []
                for i, feat in enumerate(feat_list):
                    B_feat, T, C, H, W = feat.shape
                    feat_reshaped = feat.permute(0, 2, 1, 3, 4).contiguous().view(B_feat, C, T * H * W).transpose(1, 2)
                    feat_idx = min(i, len(self.prob3_feat_mlps) - 1)
                    feat_transformed = self.prob3_feat_mlps[feat_idx](feat_reshaped)
                    transformed_feats.append(feat_transformed)
                model_output = torch.stack(transformed_feats, dim=0).mean(dim=0)
        else:
            model_output = model_output
        if feat_list is None or not feat_list:
            loss = torch.tensor(0.0, device=self.device, requires_grad=True)
            loss_dict = {'total_loss': loss.detach()}
        else:
            loss, loss_dict = self.calculate_vjepa_physics_loss(model_output, gt)
        return (loss, loss_dict)

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        loss, loss_dict = self.shared_step(batch)
        if not torch.isfinite(loss):
            mainlogger.warning(f'Non-finite loss detected at batch {batch_idx}: {loss}')
            loss = torch.tensor(0.001, device=loss.device, requires_grad=True)
        prefix = 'train'
        log_dict_prefixed = {f'{prefix}/{k}': v for k, v in loss_dict.items()}
        if (batch_idx + 1) % self.log_every_t == 0:
            mainlogger.info(f'batch:{batch_idx}|epoch:{self.current_epoch} [globalstep:{self.global_step}]: loss={loss}')
        self.log_dict(log_dict_prefixed, prog_bar=True, logger=True, on_step=True, on_epoch=False)
        return loss

    def register_schedule(self, *args, **kwargs):
        mainlogger.info('PhysicsPredictionVJEPA_Base: Skipping diffusion schedule registration.')
        self.num_timesteps = kwargs.get('num_timesteps', 1000)
        pass

class PhysicsPredictionVJEPA_Base_Test_probe(PhysicsPredictionVJEPA_Base_probe):

    def __init__(self, *args, save_result_dir=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_result_dir = save_result_dir
        if self.save_result_dir is not None:
            os.makedirs(self.save_result_dir, exist_ok=True)
            self.result_file_path = os.path.join(self.save_result_dir, 'results.txt')
            if os.path.exists(self.result_file_path):
                os.remove(self.result_file_path)
            mainlogger.info(f'Test results for PhysicsPredictionVJEPA_Base_Test_probe will be saved to {self.result_file_path}')

    def get_batch_input_for_test(self, batch: dict) -> tuple:
        x, cond, gt, mask, keytime_mask = self.get_batch_input(batch, random_uncond=False)
        fns = batch['path']
        return (x, cond, gt, mask, keytime_mask, fns)

    def run_prediction_step(self, batch: dict) -> tuple:
        x, cond, gt, mask, keytime_mask, fns = self.get_batch_input_for_test(batch)
        t = torch.zeros(x.shape[0], device=self.device, dtype=torch.long)
        model_output, feat_list = self.model.diffusion_model(x, t=t, context=None)
        if self.use_all_feat_in_attentive_probe:
            if feat_list and isinstance(feat_list, list):
                transformed_features = []
                for i, feat in enumerate(feat_list):
                    B, T, C, H, W = feat.shape
                    feat_reshaped = feat.permute(0, 1, 3, 4, 2).contiguous().view(B, T * H * W, C)
                    feat_idx = min(i, len(self.prob3_feat_mlps) - 1)
                    feat_transformed = self.prob3_feat_mlps[feat_idx](feat_reshaped)
                    transformed_features.append(feat_transformed)
                model_output = torch.stack(transformed_features, dim=0).mean(dim=0)
            else:
                model_output = model_output
        else:
            model_output = model_output
        if feat_list is None or not feat_list:
            loss = torch.tensor(0.0, device=self.device, requires_grad=True)
            loss_dict = {'total_loss': loss.detach()}
            predictions = None
        else:
            loss, loss_dict, predictions = self.calculate_vjepa_physics_loss(model_output, gt, return_prediction=True)
        return (loss, loss_dict, predictions, gt, fns)

    @torch.no_grad()
    def validation_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        loss, loss_dict, predictions, ground_truth, fns = self.run_prediction_step(batch)
        prefix = 'val'
        num_pairs = len(fns) // 2 if fns else 0
        self.log_dict(loss_dict, prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True, batch_size=num_pairs)
        if self.save_result_dir and predictions is not None and fns:
            if predictions.dim() > 1 and predictions.shape[1] > 1:
                preds = predictions[:, 0].squeeze().cpu().tolist()
            else:
                preds = predictions.squeeze().cpu().tolist()
            gts = ground_truth['gt_restitution'].cpu().tolist()
            if not isinstance(preds, list):
                preds = [preds]
            if not isinstance(gts, list):
                gts = [gts]
            with open(self.result_file_path, 'a') as f:
                for i in range(len(gts)):
                    f.write(f'File: {fns[i]}, Prediction: {preds[i]:.6f}, GroundTruth: {gts[i]}\n')
        print('save_result_dir: ', self.save_result_dir)
        return loss

class PhysicsPredictionVJEPA_Base_Test(PhysicsPredictionVJEPA_Base):

    def __init__(self, *args, save_result_dir=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_result_dir = save_result_dir
        if self.save_result_dir:
            os.makedirs(self.save_result_dir, exist_ok=True)
            self.result_file_path = os.path.join(self.save_result_dir, 'results.txt')
            if os.path.exists(self.result_file_path):
                os.remove(self.result_file_path)
            mainlogger.info(f'Test results will be saved to {self.result_file_path}')

    def get_batch_input(self, batch, random_uncond, **kwargs):
        kwargs['return_fns'] = True
        return super().get_batch_input(batch, random_uncond, **kwargs)

    def forward(self, x, c, gt, mask, keytime_mask, fns, **kwargs):
        t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=self.device).long()
        return self.p_losses(x, c, t, gt, mask, keytime_mask, fns=fns, **kwargs)

    def p_losses(self, x_start, cond, t, gt, mask, keytime_mask, fns, **kwargs):
        model_output, feat_list = self.model.diffusion_model(x_start, t, **cond)
        prob3_outputs = []
        if feat_list:
            for i, feat in enumerate(feat_list):
                B, T, C, H, W = feat.shape
                query_vector = self.prob3_feat_mlps[i](self.prob3_learnable_vector)
                q = query_vector.expand(B, -1).unsqueeze(1)
                feat_reshaped = feat.permute(0, 1, 3, 4, 2).contiguous().view(B, T * H * W, C)
                scale = C ** (-0.5)
                attn_scores = torch.matmul(q, feat_reshaped.transpose(-2, -1)) * scale
                attn_weights = torch.softmax(attn_scores, dim=-1)
                attended_feat = torch.matmul(attn_weights, feat_reshaped).squeeze(1)
                prob3_feat_output = self.prob3_feat_output_mlps[i](attended_feat)
                prob3_outputs.append(prob3_feat_output)
        loss = torch.tensor(0.0, device=x_start.device)
        loss_dict = {}
        if prob3_outputs:
            prob3_combined = torch.stack(prob3_outputs, dim=1)
            prob3_averaged = torch.mean(prob3_combined, dim=1)
            gt_prediction = self.prob3_output_projection(prob3_averaged)
            loss_gt_restitution = self.l1_loss(gt_prediction.squeeze(), gt['gt_restitution'].to(x_start.device))
            loss = loss_gt_restitution
            loss_dict['loss_gt_restitution'] = loss_gt_restitution.item()
            if self.save_result_dir and fns:
                preds = gt_prediction.squeeze().cpu().tolist()
                gts = gt['gt_restitution'].cpu().tolist()
                if not isinstance(preds, list):
                    preds = [preds]
                if not isinstance(gts, list):
                    gts = [gts]
                with open(self.result_file_path, 'a') as f:
                    for i in range(len(gts)):
                        f.write(f'File: {fns[i]}, Prediction: {preds[i]:.6f}, GroundTruth: {gts[i]}\n')
                        try:
                            gv = float(gts[i])
                        except (TypeError, ValueError):
                            gv = float(gts[i].item()) if hasattr(gts[i], 'item') else float(gts[i])
                        _dump_elasticity_abs_clip_json(self.save_result_dir, fns[i], float(preds[i]), gv)
        loss_dict['total_loss'] = loss.item()
        return (loss, loss_dict)

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        loss, loss_dict = self.shared_step(batch, random_uncond=False)
        self.log_dict(loss_dict, prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        if (batch_idx + 1) % 10 == 0:
            mainlogger.info(f'validation batch:{batch_idx}|epoch:{self.current_epoch} [globalstep:{self.global_step}]: loss={loss.item():.6f}')
        print('save_result_dir: ', self.save_result_dir)
        return loss

class PhysicsPredictionVJEPAElasticityLoss(PhysicsPredictionVJEPA_Base):

    def p_losses(self, video_input, cond, t, gt, mask, keytime_mask, noise=None, **kwargs):
        _, feat_list = self.apply_model(video_input, t, cond, **kwargs)
        return self.calculate_physics_loss(feat_list, gt)

class LatentVisualDiffusionv84LogLoss(LatentVisualDiffusionv84):

    def __init__(self, direct_weight=1.0, scale_weight=0.5, value_weight=0.5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.direct_weight = direct_weight
        self.scale_weight = scale_weight
        self.value_weight = value_weight

    def p_losses(self, x_start, cond, t, gt, mask, keytime_mask, noise=None, **kwargs):
        if self.noise_strength > 0:
            b, c, f, _, _ = x_start.shape
            offset_noise = torch.randn(b, c, f, 1, 1, device=x_start.device)
            noise = default(noise, lambda: torch.randn_like(x_start) + self.noise_strength * offset_noise)
        else:
            noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        cond['c_crossattn'][0][:, :77, :] = self.prob3_learnable_prompt
        model_output, feat_list = self.apply_model(x_noisy, t, cond, **kwargs)
        prob3_outputs = []
        for feat_i, feat in enumerate(feat_list):
            B, T, C, H, W = feat.shape
            query_vector = self.prob3_feat_mlps[feat_i](self.prob3_learnable_vector)
            query_vector = query_vector.expand(B, -1).unsqueeze(1)
            feat_reshaped = feat.permute(0, 1, 3, 4, 2).contiguous().view(B, T * H * W, C)
            scale = C ** (-0.5)
            attn_scores = torch.matmul(query_vector, feat_reshaped.transpose(-2, -1)) * scale
            attn_weights = torch.softmax(attn_scores, dim=-1)
            attended_feat = torch.matmul(attn_weights, feat_reshaped)
            attended_feat = attended_feat.squeeze(1)
            prob3_feat_output = self.prob3_feat_output_mlps[feat_i](attended_feat)
            prob3_outputs.append(prob3_feat_output)
        if prob3_outputs:
            prob3_combined = torch.stack(prob3_outputs, dim=1)
            prob3_averaged = torch.mean(prob3_combined, dim=1)
            gt_retitusion = self.prob3_output_projection(prob3_averaged)
        else:
            gt_retitusion = None
        l1_loss = ImprovedScaleAwareL1Loss(direct_weight=self.direct_weight, scale_weight=self.scale_weight, value_weight=self.value_weight)
        device = feat_list[0].device
        for key in gt:
            if isinstance(gt[key], torch.Tensor):
                gt[key] = gt[key].to(device)
            elif isinstance(gt[key], list):
                gt[key] = [item.to(device) if isinstance(item, torch.Tensor) else item for item in gt[key]]
        loss_gt_retitusion = 0 if gt_retitusion is None else l1_loss(torch.sigmoid(gt_retitusion[:, 0]) * 0.01, gt['gt_restitution'])
        print('pred: ', torch.sigmoid(gt_retitusion[:, 0]))
        print('gt: ', gt['gt_restitution'] / 0.01)
        if self.scale_weight == 0 and self.value_weight == 0:
            loss_gt_retitusion = loss_gt_retitusion / 0.01
        if gt_retitusion is not None:
            print(loss_gt_retitusion)
        loss = loss_gt_retitusion
        loss_dict = {}
        loss_dict.update({'total_loss': loss, 'loss_gt_retitusion': loss_gt_retitusion})
        for key_i in loss_dict.keys():
            if isinstance(loss_dict[key_i], torch.Tensor) and torch.isnan(loss_dict[key_i]):
                loss_dict[key_i] = torch.tensor(0.001, device=device, requires_grad=True)
                print('NaN loss detected, using fallback value')
        return (loss, loss_dict)

    def calculate_physics_loss(self, feat_list, gt):
        prob3_outputs = []
        for feat_i, feat in enumerate(feat_list):
            B, T, C, H, W = feat.shape
            query_vector = self.prob3_feat_mlps[feat_i](self.prob3_learnable_vector)
            query_vector = query_vector.expand(B, -1).unsqueeze(1)
            feat_reshaped = feat.permute(0, 1, 3, 4, 2).contiguous().view(B, T * H * W, C)
            scale = C ** (-0.5)
            attn_scores = torch.matmul(query_vector, feat_reshaped.transpose(-2, -1)) * scale
            attn_weights = torch.softmax(attn_scores, dim=-1)
            attended_feat = torch.matmul(attn_weights, feat_reshaped).squeeze(1)
            prob3_feat_output = self.prob3_feat_output_mlps[feat_i](attended_feat)
            prob3_outputs.append(prob3_feat_output)
        if prob3_outputs:
            prob3_combined = torch.stack(prob3_outputs, dim=1)
            prob3_averaged = torch.mean(prob3_combined, dim=1)
            gt_retitusion = self.prob3_output_projection(prob3_averaged)
        else:
            gt_retitusion = None
        l1_loss = ImprovedScaleAwareL1Loss(direct_weight=self.direct_weight, scale_weight=self.scale_weight, value_weight=self.value_weight)
        device = feat_list[0].device
        for key in gt:
            if isinstance(gt[key], torch.Tensor):
                gt[key] = gt[key].to(device)
            elif isinstance(gt[key], list):
                gt[key] = [item.to(device) if isinstance(item, torch.Tensor) else item for item in gt[key]]
        loss_gt_retitusion = 0 if gt_retitusion is None else l1_loss(torch.sigmoid(gt_retitusion[:, 0]) * 0.01, gt['gt_restitution'])
        if self.training:
            print('pred: ', torch.sigmoid(gt_retitusion[:, 0]))
            print('gt: ', gt['gt_restitution'] / 0.01)
        if self.scale_weight == 0 and self.value_weight == 0:
            loss_gt_retitusion = loss_gt_retitusion / 0.01
        if gt_retitusion is not None and self.training:
            print(loss_gt_retitusion)
        loss = loss_gt_retitusion
        loss_dict = {'total_loss': loss, 'loss_gt_retitusion': loss_gt_retitusion}
        for key_i in loss_dict.keys():
            if isinstance(loss_dict[key_i], torch.Tensor) and torch.isnan(loss_dict[key_i]):
                loss_dict[key_i] = torch.tensor(0.001, device=device, requires_grad=True)
                print('NaN loss detected, using fallback value')
        return (loss, loss_dict)

class PhysicsPredictionVJEPALogLoss(PhysicsPredictionVJEPA_Base, LatentVisualDiffusionv84LogLoss):

    def p_losses(self, video_input, cond, t, gt, mask, keytime_mask, noise=None, **kwargs):
        _, feat_list = self.apply_model(video_input, t, cond, **kwargs)
        return self.calculate_physics_loss(feat_list, gt)

class PhysicsPredictionVJEPALogLoss_probe(PhysicsPredictionVJEPA_Base_probe, LatentVisualDiffusionv84LogLoss):

    def calculate_vjepa_physics_loss(self, feat_list: list, gt: dict, return_prediction: bool=False) -> tuple:
        if feat_list is None:
            device = self.device
        else:
            device = feat_list.device
        for key in gt:
            if isinstance(gt[key], torch.Tensor):
                gt[key] = gt[key].to(device)
            elif isinstance(gt[key], list):
                gt[key] = [item.to(device) if isinstance(item, torch.Tensor) else item for item in gt[key]]
        l1_loss = ImprovedScaleAwareL1Loss(direct_weight=self.direct_weight, scale_weight=self.scale_weight, value_weight=self.value_weight)
        if feat_list is None:
            gt_retitusion = None
        else:
            last_layer_feat = feat_list
            classifier_output = self.attentive_classifier(last_layer_feat)
            action_logits = classifier_output['action']
            gt_retitusion = action_logits
        loss_gt_retitusion = 0 if gt_retitusion is None else l1_loss(torch.sigmoid(gt_retitusion[:, 0]) * 0.01, gt['gt_restitution'])
        if self.training:
            print('pred: ', torch.sigmoid(gt_retitusion[:, 0]))
            print('gt: ', gt['gt_restitution'] / 0.01)
        loss = loss_gt_retitusion
        loss_dict = {'total_loss': loss, 'loss_gt_retitusion': loss_gt_retitusion}
        for key_i in loss_dict.keys():
            if isinstance(loss_dict[key_i], torch.Tensor) and (torch.isnan(loss_dict[key_i]) or torch.isinf(loss_dict[key_i])):
                mainlogger.info(f'NaN or Inf loss detected in {key_i}, using fallback value')
                loss_dict[key_i] = torch.tensor(0.001, device=device, requires_grad=True)
        if return_prediction:
            return (loss, loss_dict, gt_retitusion)
        return (loss, loss_dict)

class LatentVisualDiffusionv84FrictionLoss(LatentVisualDiffusionv84):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def p_losses(self, x_start, cond, t, gt, mask, keytime_mask, noise=None, **kwargs):
        if self.noise_strength > 0:
            b, c, f, _, _ = x_start.shape
            offset_noise = torch.randn(b, c, f, 1, 1, device=x_start.device)
            noise = default(noise, lambda: torch.randn_like(x_start) + self.noise_strength * offset_noise)
        else:
            noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        cond['c_crossattn'][0][:, :77, :] = self.prob3_learnable_prompt
        model_output, feat_list = self.apply_model(x_noisy, t, cond, **kwargs)
        prob3_outputs = []
        for feat_i, feat in enumerate(feat_list):
            B, T, C, H, W = feat.shape
            query_vector = self.prob3_feat_mlps[feat_i](self.prob3_learnable_vector)
            query_vector = query_vector.expand(B, -1).unsqueeze(1)
            feat_reshaped = feat.permute(0, 1, 3, 4, 2).contiguous().view(B, T * H * W, C)
            scale = C ** (-0.5)
            attn_scores = torch.matmul(query_vector, feat_reshaped.transpose(-2, -1)) * scale
            attn_weights = torch.softmax(attn_scores, dim=-1)
            attended_feat = torch.matmul(attn_weights, feat_reshaped)
            attended_feat = attended_feat.squeeze(1)
            prob3_feat_output = self.prob3_feat_output_mlps[feat_i](attended_feat)
            prob3_outputs.append(prob3_feat_output)
        if prob3_outputs:
            prob3_combined = torch.stack(prob3_outputs, dim=1)
            prob3_averaged = torch.mean(prob3_combined, dim=1)
            gt_retitusion = self.prob3_output_projection(prob3_averaged)
        else:
            gt_retitusion = None
        l1_loss = nn.L1Loss()
        device = feat_list[0].device
        for key in gt:
            if isinstance(gt[key], torch.Tensor):
                gt[key] = gt[key].to(device)
            elif isinstance(gt[key], list):
                gt[key] = [item.to(device) if isinstance(item, torch.Tensor) else item for item in gt[key]]
        loss_gt_retitusion = 0 if gt_retitusion is None else l1_loss(torch.sigmoid(gt_retitusion[:, 0]) * 0.2, gt['gt_restitution']) / 0.2
        print('pred: ', torch.sigmoid(gt_retitusion[:, 0]))
        print('gt: ', gt['gt_restitution'] / 0.2)
        if gt_retitusion is not None:
            print(loss_gt_retitusion)
        loss = loss_gt_retitusion
        loss_dict = {}
        loss_dict.update({'total_loss': loss, 'loss_gt_retitusion': loss_gt_retitusion})
        for key_i in loss_dict.keys():
            if isinstance(loss_dict[key_i], torch.Tensor) and torch.isnan(loss_dict[key_i]):
                loss_dict[key_i] = torch.tensor(0.001, device=device, requires_grad=True)
                print('NaN loss detected, using fallback value')
        return (loss, loss_dict)

    def calculate_physics_loss(self, feat_list, gt):
        prob3_outputs = []
        for feat_i, feat in enumerate(feat_list):
            B, T, C, H, W = feat.shape
            query_vector = self.prob3_feat_mlps[feat_i](self.prob3_learnable_vector)
            query_vector = query_vector.expand(B, -1).unsqueeze(1)
            feat_reshaped = feat.permute(0, 1, 3, 4, 2).contiguous().view(B, T * H * W, C)
            scale = C ** (-0.5)
            attn_scores = torch.matmul(query_vector, feat_reshaped.transpose(-2, -1)) * scale
            attn_weights = torch.softmax(attn_scores, dim=-1)
            attended_feat = torch.matmul(attn_weights, feat_reshaped).squeeze(1)
            prob3_feat_output = self.prob3_feat_output_mlps[feat_i](attended_feat)
            prob3_outputs.append(prob3_feat_output)
        if prob3_outputs:
            prob3_combined = torch.stack(prob3_outputs, dim=1)
            prob3_averaged = torch.mean(prob3_combined, dim=1)
            gt_retitusion = self.prob3_output_projection(prob3_averaged)
        else:
            gt_retitusion = None
        l1_loss = nn.L1Loss()
        device = feat_list[0].device
        for key in gt:
            if isinstance(gt[key], torch.Tensor):
                gt[key] = gt[key].to(device)
            elif isinstance(gt[key], list):
                gt[key] = [item.to(device) if isinstance(item, torch.Tensor) else item for item in gt[key]]
        loss_gt_retitusion = 0 if gt_retitusion is None else l1_loss(torch.sigmoid(gt_retitusion[:, 0]) * 0.2, gt['gt_restitution']) / 0.2
        if self.training:
            print('pred: ', torch.sigmoid(gt_retitusion[:, 0]))
            print('gt: ', gt['gt_restitution'] / 0.2)
        if gt_retitusion is not None and self.training:
            print(loss_gt_retitusion)
        loss = loss_gt_retitusion
        loss_dict = {'total_loss': loss, 'loss_gt_retitusion': loss_gt_retitusion}
        for key_i in loss_dict.keys():
            if isinstance(loss_dict[key_i], torch.Tensor) and torch.isnan(loss_dict[key_i]):
                loss_dict[key_i] = torch.tensor(0.001, device=device, requires_grad=True)
                print('NaN loss detected, using fallback value')
        return (loss, loss_dict)

class PhysicsPredictionVJEPAFrictionLoss(PhysicsPredictionVJEPA_Base, LatentVisualDiffusionv84FrictionLoss):

    def p_losses(self, video_input, cond, t, gt, mask, keytime_mask, noise=None, **kwargs):
        _, feat_list = self.apply_model(video_input, t, cond, **kwargs)
        return self.calculate_physics_loss(feat_list, gt)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_result_dir = '/path/to/DynamiCrafter/2.11_result_test4_1800itr'
        if not os.path.exists(self.save_result_dir):
            os.makedirs(self.save_result_dir, exist_ok=True)

class PhysicsPredictionVJEPAFrictionLoss_probe(PhysicsPredictionVJEPA_Base_probe, LatentVisualDiffusionv84FrictionLoss):

    def calculate_vjepa_physics_loss(self, feat_list: list, gt: dict, return_prediction: bool=False) -> tuple:
        if feat_list is None:
            device = self.device
        else:
            device = feat_list.device
        for key in gt:
            if isinstance(gt[key], torch.Tensor):
                gt[key] = gt[key].to(device)
            elif isinstance(gt[key], list):
                gt[key] = [item.to(device) if isinstance(item, torch.Tensor) else item for item in gt[key]]
        l1_loss = nn.L1Loss()
        if feat_list is None:
            gt_retitusion = None
        else:
            last_layer_feat = feat_list
            classifier_output = self.attentive_classifier(last_layer_feat)
            action_logits = classifier_output['action']
            gt_retitusion = action_logits
        loss_gt_retitusion = 0 if gt_retitusion is None else l1_loss(torch.sigmoid(gt_retitusion[:, 0]) * 0.2, gt['gt_restitution']) / 0.2
        if self.training:
            print('pred: ', torch.sigmoid(gt_retitusion[:, 0]))
            print('gt: ', gt['gt_restitution'] / 0.2)
        if gt_retitusion is not None and self.training:
            print(loss_gt_retitusion)
        loss = loss_gt_retitusion
        loss_dict = {'total_loss': loss, 'loss_gt_retitusion': loss_gt_retitusion}
        for key_i in loss_dict.keys():
            if isinstance(loss_dict[key_i], torch.Tensor) and (torch.isnan(loss_dict[key_i]) or torch.isinf(loss_dict[key_i])):
                mainlogger.info(f'NaN or Inf loss detected in {key_i}, using fallback value')
                loss_dict[key_i] = torch.tensor(0.001, device=device, requires_grad=True)
        if return_prediction:
            return (loss, loss_dict, gt_retitusion)
        return (loss, loss_dict)

class LatentVisualDiffusionv87(LatentVisualDiffusionv47):

    def __init__(self, *args, prob3_dim=0, major_specify=0, cond_specify=0, collect_map_together=0, concat_2_feat=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.concat_2_feat = concat_2_feat
        self.collect_map_together = collect_map_together
        self.major_specify = major_specify
        self.cond_specify = cond_specify
        self.prob3_dim = prob3_dim
        self.prob3_learnable_vector = nn.Parameter(torch.randn((1, self.prob3_dim)) * 0.02)
        self.p_decoder = None
        feat_dims = [320, 320, 320, 320, 640, 640, 640, 1280, 1280, 1280, 1280, 1280, 1280, 1280, 1280, 1280, 1280, 1280, 1280, 640, 640, 640, 320, 320, 320]
        self.prob3_feat_mlps = nn.ModuleList([nn.Linear(self.prob3_dim, feat_dim) for feat_dim in feat_dims])
        if self.collect_map_together == 0:
            self.prob3_feat_output_mlps = nn.ModuleList([nn.Linear(feat_dim, self.prob3_dim) for feat_dim in feat_dims])
        elif self.collect_map_together == 1:
            self.prob3_feat_output_mlps = nn.Linear(sum(feat_dims), self.prob3_dim)
        else:
            assert self.collect_map_together == 2
            self.prob3_feat_output_mlps = nn.Linear(sum(feat_dims) * 2, 1)
        if self.collect_map_together != 2:
            if concat_2_feat == False:
                self.prob3_output_projection = nn.Sequential(nn.Linear(self.prob3_dim, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 1))
            else:
                self.prob3_output_projection = nn.Sequential(nn.Linear(self.prob3_dim * 2, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 1))
        self.prob3_learnable_prompt = nn.Parameter(torch.randn((1, 77, 1024)) * 0.02)
        self._freeze_all_but_some(self.ft_weights_list)

    def _freeze_all_but_some(self, ft_weights_list):
        for name, param in self.named_parameters():
            if 'prob3' not in name:
                param.requires_grad = False
            if 'prob3' in name:
                param.requires_grad = True
        for ft_weight in ft_weights_list:
            if any((c in ft_weight for c in '.*+?^$[](){}|\\')):
                pattern = re.compile(ft_weight.replace('.', '\\.').replace('*', '.*'))
                for name, param in self.named_parameters():
                    if pattern.search(name):
                        param.requires_grad = True
            else:
                for name, param in self.named_parameters():
                    if ft_weight in name:
                        param.requires_grad = True

    def configure_optimizers(self):
        lr = self.learning_rate
        params_1 = [p for n, p in self.named_parameters() if p.requires_grad and ('prob3' in n or 'p_decoder' in n)]
        for n, p in self.named_parameters():
            if p.requires_grad:
                print(n)
        mainlogger.info(f'@Training [{len(params_1)}] Full Paramters.')
        optimizer = torch.optim.AdamW([{'params': params_1, 'lr': lr}])
        if self.use_scheduler:
            ipdb.set_trace()
            mainlogger.info('Setting up scheduler...')
            lr_scheduler = self.configure_schedulers(optimizer)
            return ([optimizer], [lr_scheduler])
        return optimizer

    def p_losses(self, x_start, cond, t, gt, mask, keytime_mask, noise=None, **kwargs):
        if self.noise_strength > 0:
            b, c, f, _, _ = x_start.shape
            offset_noise = torch.randn(b, c, f, 1, 1, device=x_start.device)
            noise = default(noise, lambda: torch.randn_like(x_start) + self.noise_strength * offset_noise)
        else:
            noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        cond['c_crossattn'][0][:, :77, :] = self.prob3_learnable_prompt
        model_output, feat_list = self.apply_model(x_noisy, t, cond, **kwargs)
        prob3_outputs = []
        for feat_i, feat in enumerate(feat_list):
            B, T, C, H, W = feat.shape
            query_vector = self.prob3_feat_mlps[feat_i](self.prob3_learnable_vector)
            query_vector = query_vector.expand(B, -1).unsqueeze(1)
            feat_reshaped = feat.permute(0, 1, 3, 4, 2).contiguous().view(B, T * H * W, C)
            scale = C ** (-0.5)
            attn_scores = torch.matmul(query_vector, feat_reshaped.transpose(-2, -1)) * scale
            attn_weights = torch.softmax(attn_scores, dim=-1)
            attended_feat = torch.matmul(attn_weights, feat_reshaped)
            attended_feat = attended_feat.squeeze(1)
            if self.collect_map_together == 0:
                prob3_feat_output = self.prob3_feat_output_mlps[feat_i](attended_feat)
                prob3_outputs.append(prob3_feat_output)
            else:
                prob3_outputs.append(attended_feat)
        if prob3_outputs:
            if self.collect_map_together == 0:
                prob3_combined = torch.stack(prob3_outputs, dim=1)
                prob3_averaged = torch.mean(prob3_combined, dim=1)
                prob3_averaged = prob3_averaged.reshape(int(prob3_averaged.shape[0] / 2), 2, prob3_averaged.shape[1])
                if self.concat_2_feat == False:
                    prob3_diff = prob3_averaged[:, 0, :] - prob3_averaged[:, 1, :]
                    gt_retitusion = self.prob3_output_projection(prob3_diff)
                else:
                    prob3_diff = torch.cat([prob3_averaged[:, 0, :], prob3_averaged[:, 1, :]], dim=1)
                    gt_retitusion = self.prob3_output_projection(prob3_diff)
            elif self.collect_map_together == 1:
                prob3_combined = torch.cat(prob3_outputs, dim=1)
                prob3_averaged = self.prob3_feat_output_mlps(prob3_combined)
                prob3_averaged = prob3_averaged.reshape(int(prob3_averaged.shape[0] / 2), 2, prob3_averaged.shape[1])
                if self.concat_2_feat == False:
                    prob3_diff = prob3_averaged[:, 0, :] - prob3_averaged[:, 1, :]
                    gt_retitusion = self.prob3_output_projection(prob3_diff)
                else:
                    prob3_diff = torch.cat([prob3_averaged[:, 0, :], prob3_averaged[:, 1, :]], dim=1)
                    gt_retitusion = self.prob3_output_projection(prob3_diff)
            else:
                assert self.collect_map_together == 2
                prob3_combined = torch.cat(prob3_outputs, dim=1)
                prob3_combined = prob3_combined.reshape(int(prob3_combined.shape[0] / 2), 2 * prob3_combined.shape[1])
                gt_retitusion = self.prob3_feat_output_mlps(prob3_combined)
        else:
            gt_retitusion = None
        bce_loss = nn.BCEWithLogitsLoss()
        device = feat_list[0].device
        for key in gt:
            if isinstance(gt[key], torch.Tensor):
                gt[key] = gt[key].to(device)
            elif isinstance(gt[key], list):
                gt[key] = [item.to(device) if isinstance(item, torch.Tensor) else item for item in gt[key]]
        loss_gt_retitusion = 0 if gt_retitusion is None else bce_loss(gt_retitusion[:, 0], gt['gt_restitution'].float())
        if gt_retitusion is not None:
            print(loss_gt_retitusion)
        loss = loss_gt_retitusion
        loss_dict = {}
        loss_dict.update({'total_loss': loss, 'loss_gt_retitusion': loss_gt_retitusion})
        for key_i in loss_dict.keys():
            if isinstance(loss_dict[key_i], torch.Tensor) and torch.isnan(loss_dict[key_i]):
                loss_dict[key_i] = torch.tensor(0.001, device=device, requires_grad=True)
                print('NaN loss detected, using fallback value')
        return (loss, loss_dict)

    def get_batch_input(self, batch, random_uncond, return_first_stage_outputs=False, return_original_cond=False, return_fs=False, return_cond_frame=False, return_original_input=False, **kwargs):
        x = super().get_input(batch, self.first_stage_key)
        x = x.reshape(x.shape[0] * x.shape[1], x.shape[2], x.shape[3], x.shape[4], x.shape[5])
        if self.major_specify == 0:
            z = self.encode_first_stage(x)
        elif self.major_specify == 1:
            cur_batch_mask = batch['mask'].reshape(batch['mask'].shape[0] * batch['mask'].shape[1], batch['mask'].shape[2], batch['mask'].shape[3], batch['mask'].shape[4], batch['mask'].shape[5])
            z = self.encode_first_stage(x * cur_batch_mask)
        elif self.major_specify == 2:
            x_red = batch['mask']
            x_red = x_red.reshape(x_red.shape[0] * x_red.shape[1], x_red.shape[2], x_red.shape[3], x_red.shape[4], x_red.shape[5])
            z = self.encode_first_stage(x_red)
        elif self.major_specify == 3:
            x_red = batch['mask']
            x_red = x_red.reshape(x_red.shape[0] * x_red.shape[1], x_red.shape[2], x_red.shape[3], x_red.shape[4], x_red.shape[5])
            x_concat_new = torch.cat([x[:, :, :-1, :, :], x_red[:, :, -1:, :, :]], dim=2)
            z = self.encode_first_stage(x_concat_new)
        cond_input = batch[self.cond_stage_key]
        cond_input = cond_input + cond_input
        if isinstance(cond_input, dict) or isinstance(cond_input, list):
            cond_emb = self.get_learned_conditioning(cond_input)
        else:
            cond_emb = self.get_learned_conditioning(cond_input.to(self.device))
        cond = {}
        if random_uncond:
            random_num = torch.rand(x.size(0), device=x.device)
        else:
            random_num = torch.ones(x.size(0), device=x.device)
        prompt_mask = rearrange(random_num < 2 * self.uncond_prob, 'n -> n 1 1')
        input_mask = 1 - rearrange((random_num >= self.uncond_prob).float() * (random_num < 3 * self.uncond_prob).float(), 'n -> n 1 1 1')
        null_prompt = self.get_learned_conditioning([''])
        prompt_imb = torch.where(prompt_mask, null_prompt, cond_emb.detach())
        cond_frame_index = 0
        if self.rand_cond_frame:
            cond_frame_index = random.randint(0, self.model.diffusion_model.temporal_length - 1)
        if self.cond_specify == 3:
            cond_frame_index = -1
        if self.cond_specify == 0:
            img = x[:, :, cond_frame_index, ...]
        elif self.cond_specify == 1:
            img = x[:, :, cond_frame_index, ...]
            cur_batch_mask = batch['mask'].reshape(batch['mask'].shape[0] * batch['mask'].shape[1], batch['mask'].shape[2], batch['mask'].shape[3], batch['mask'].shape[4], batch['mask'].shape[5])
            img = img * cur_batch_mask[:, :, cond_frame_index, ...]
        elif self.cond_specify == 2:
            cur_batch_mask = batch['mask'].reshape(batch['mask'].shape[0] * batch['mask'].shape[1], batch['mask'].shape[2], batch['mask'].shape[3], batch['mask'].shape[4], batch['mask'].shape[5])
            img = cur_batch_mask[:, :, cond_frame_index, ...]
        elif self.cond_specify == 3:
            cur_batch_mask = batch['mask'].reshape(batch['mask'].shape[0] * batch['mask'].shape[1], batch['mask'].shape[2], batch['mask'].shape[3], batch['mask'].shape[4], batch['mask'].shape[5])
            img = cur_batch_mask[:, :, cond_frame_index, ...]
        img = input_mask * img
        img_emb = self.embedder(img)
        img_emb = self.image_proj_model(img_emb)
        if self.model.conditioning_key == 'hybrid':
            if self.interp_mode:
                img_cat_cond = torch.zeros_like(z)
                img_cat_cond[:, :, 0, :, :] = z[:, :, 0, :, :]
                img_cat_cond[:, :, -1, :, :] = z[:, :, -1, :, :]
            elif self.cond_specify == 0 or self.cond_specify == 3:
                img_cat_cond = z[:, :, cond_frame_index, :, :]
                img_cat_cond = img_cat_cond.unsqueeze(2)
                img_cat_cond = repeat(img_cat_cond, 'b c t h w -> b c (repeat t) h w', repeat=z.shape[2])
            elif self.cond_specify == 1:
                cur_batch_mask = batch['mask'].reshape(batch['mask'].shape[0] * batch['mask'].shape[1], batch['mask'].shape[2], batch['mask'].shape[3], batch['mask'].shape[4], batch['mask'].shape[5])
                img_cat_cond = self.encode_first_stage(x * cur_batch_mask)
            elif self.cond_specify == 2:
                cur_batch_mask = batch['mask'].reshape(batch['mask'].shape[0] * batch['mask'].shape[1], batch['mask'].shape[2], batch['mask'].shape[3], batch['mask'].shape[4], batch['mask'].shape[5])
                img_cat_cond = self.encode_first_stage(cur_batch_mask)
            cond['c_concat'] = [img_cat_cond]
        cond['c_crossattn'] = [torch.cat([prompt_imb, img_emb], dim=1)]
        out = [z, cond]
        if return_first_stage_outputs:
            xrec = self.decode_first_stage(z)
            out.extend([xrec])
        if return_original_cond:
            out.append(cond_input)
        if return_fs:
            if self.fps_condition_type == 'fs':
                fs = super().get_input(batch, 'frame_stride')
            elif self.fps_condition_type == 'fps':
                fs = super().get_input(batch, 'fps')
            out.append(torch.cat([fs, fs], dim=0))
        if return_cond_frame:
            out.append(x[:, :, cond_frame_index, ...].unsqueeze(2))
        if return_original_input:
            out.append(x)
        gt = batch['gt_ann']
        out.append(gt)
        out.append(batch['mask'].reshape(batch['mask'].shape[0] * batch['mask'].shape[1], batch['mask'].shape[2], batch['mask'].shape[3], batch['mask'].shape[4], batch['mask'].shape[5]))
        out.append(batch['keytime_mask'].reshape(batch['keytime_mask'].shape[0] * batch['keytime_mask'].shape[1], batch['keytime_mask'].shape[2], batch['keytime_mask'].shape[3], batch['keytime_mask'].shape[4], batch['keytime_mask'].shape[5]))
        return out

class LatentVisualDiffusionv87Test(LatentVisualDiffusionv87):

    def __init__(self, *args, save_result_dir=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_result_dir = save_result_dir
        if self.save_result_dir is not None and (not os.path.exists(self.save_result_dir)):
            os.makedirs(self.save_result_dir, exist_ok=True)

    def forward(self, x, c, gt, mask, keytime_mask, fns, **kwargs):
        t = torch.randint(50, 51, (x.shape[0],), device=self.device).long()
        if self.use_dynamic_rescale:
            x = x * extract_into_tensor(self.scale_arr, t, x.shape)
        return self.p_losses(x, c, t, gt, mask, keytime_mask, fns, **kwargs)

    def shared_step(self, batch, random_uncond, **kwargs):
        x, c, fs, gt, mask, keytime_mask, fns = self.get_batch_input(batch, random_uncond=random_uncond, return_fs=True)
        kwargs.update({'fs': fs.long()})
        loss, loss_dict = self(x, c, gt, mask, keytime_mask, fns, **kwargs)
        return (loss, loss_dict)

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        loss, loss_dict = self.shared_step(batch, random_uncond=self.classifier_free_guidance)
        self.log_dict(loss_dict, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=False)
        "\n        if self.use_scheduler:\n            lr = self.optimizers().param_groups[0]['lr']\n            self.log('lr_abs', lr, prog_bar=True, logger=True, on_step=True, on_epoch=False, rank_zero_only=True)\n        "
        if (batch_idx + 1) % self.log_every_t == 0:
            mainlogger.info(f'batch:{batch_idx}|epoch:{self.current_epoch} [globalstep:{self.global_step}]: loss={loss}')
        return loss

    def get_batch_input(self, batch, random_uncond, return_first_stage_outputs=False, return_original_cond=False, return_fs=False, return_cond_frame=False, return_original_input=False, **kwargs):
        x = super().get_input(batch, self.first_stage_key)
        x = x.reshape(x.shape[0] * x.shape[1], x.shape[2], x.shape[3], x.shape[4], x.shape[5])
        if self.major_specify == 0:
            z = self.encode_first_stage(x)
        elif self.major_specify == 1:
            cur_batch_mask = batch['mask'].reshape(batch['mask'].shape[0] * batch['mask'].shape[1], batch['mask'].shape[2], batch['mask'].shape[3], batch['mask'].shape[4], batch['mask'].shape[5])
            z = self.encode_first_stage(x * cur_batch_mask)
        elif self.major_specify == 2:
            x_red = batch['mask']
            x_red = x_red.reshape(x_red.shape[0] * x_red.shape[1], x_red.shape[2], x_red.shape[3], x_red.shape[4], x_red.shape[5])
            z = self.encode_first_stage(x_red)
        elif self.major_specify == 3:
            x_red = batch['mask']
            x_red = x_red.reshape(x_red.shape[0] * x_red.shape[1], x_red.shape[2], x_red.shape[3], x_red.shape[4], x_red.shape[5])
            x_concat_new = torch.cat([x[:, :, :-1, :, :], x_red[:, :, -1:, :, :]], dim=2)
            z = self.encode_first_stage(x_concat_new)
        cond_input = batch[self.cond_stage_key]
        cond_input = cond_input + cond_input
        if isinstance(cond_input, dict) or isinstance(cond_input, list):
            cond_emb = self.get_learned_conditioning(cond_input)
        else:
            cond_emb = self.get_learned_conditioning(cond_input.to(self.device))
        cond = {}
        if random_uncond:
            random_num = torch.rand(x.size(0), device=x.device)
        else:
            random_num = torch.ones(x.size(0), device=x.device)
        prompt_mask = rearrange(random_num < 2 * self.uncond_prob, 'n -> n 1 1')
        input_mask = 1 - rearrange((random_num >= self.uncond_prob).float() * (random_num < 3 * self.uncond_prob).float(), 'n -> n 1 1 1')
        null_prompt = self.get_learned_conditioning([''])
        prompt_imb = torch.where(prompt_mask, null_prompt, cond_emb.detach())
        cond_frame_index = 0
        if self.rand_cond_frame:
            cond_frame_index = random.randint(0, self.model.diffusion_model.temporal_length - 1)
        if self.cond_specify == 3:
            cond_frame_index = -1
        if self.cond_specify == 0:
            img = x[:, :, cond_frame_index, ...]
        elif self.cond_specify == 1:
            img = x[:, :, cond_frame_index, ...]
            cur_batch_mask = batch['mask'].reshape(batch['mask'].shape[0] * batch['mask'].shape[1], batch['mask'].shape[2], batch['mask'].shape[3], batch['mask'].shape[4], batch['mask'].shape[5])
            img = img * cur_batch_mask[:, :, cond_frame_index, ...]
        elif self.cond_specify == 2:
            cur_batch_mask = batch['mask'].reshape(batch['mask'].shape[0] * batch['mask'].shape[1], batch['mask'].shape[2], batch['mask'].shape[3], batch['mask'].shape[4], batch['mask'].shape[5])
            img = cur_batch_mask[:, :, cond_frame_index, ...]
        elif self.cond_specify == 3:
            cur_batch_mask = batch['mask'].reshape(batch['mask'].shape[0] * batch['mask'].shape[1], batch['mask'].shape[2], batch['mask'].shape[3], batch['mask'].shape[4], batch['mask'].shape[5])
            img = cur_batch_mask[:, :, cond_frame_index, ...]
        img = input_mask * img
        img_emb = self.embedder(img)
        img_emb = self.image_proj_model(img_emb)
        if self.model.conditioning_key == 'hybrid':
            if self.interp_mode:
                img_cat_cond = torch.zeros_like(z)
                img_cat_cond[:, :, 0, :, :] = z[:, :, 0, :, :]
                img_cat_cond[:, :, -1, :, :] = z[:, :, -1, :, :]
            elif self.cond_specify == 0 or self.cond_specify == 3:
                img_cat_cond = z[:, :, cond_frame_index, :, :]
                img_cat_cond = img_cat_cond.unsqueeze(2)
                img_cat_cond = repeat(img_cat_cond, 'b c t h w -> b c (repeat t) h w', repeat=z.shape[2])
            elif self.cond_specify == 1:
                cur_batch_mask = batch['mask'].reshape(batch['mask'].shape[0] * batch['mask'].shape[1], batch['mask'].shape[2], batch['mask'].shape[3], batch['mask'].shape[4], batch['mask'].shape[5])
                img_cat_cond = self.encode_first_stage(x * cur_batch_mask)
            elif self.cond_specify == 2:
                cur_batch_mask = batch['mask'].reshape(batch['mask'].shape[0] * batch['mask'].shape[1], batch['mask'].shape[2], batch['mask'].shape[3], batch['mask'].shape[4], batch['mask'].shape[5])
                img_cat_cond = self.encode_first_stage(cur_batch_mask)
            cond['c_concat'] = [img_cat_cond]
        cond['c_crossattn'] = [torch.cat([prompt_imb, img_emb], dim=1)]
        out = [z, cond]
        if return_first_stage_outputs:
            xrec = self.decode_first_stage(z)
            out.extend([xrec])
        if return_original_cond:
            out.append(cond_input)
        if return_fs:
            if self.fps_condition_type == 'fs':
                fs = super().get_input(batch, 'frame_stride')
            elif self.fps_condition_type == 'fps':
                fs = super().get_input(batch, 'fps')
            out.append(torch.cat([fs, fs], dim=0))
        if return_cond_frame:
            out.append(x[:, :, cond_frame_index, ...].unsqueeze(2))
        if return_original_input:
            out.append(x)
        gt = batch['gt_ann']
        out.append(gt)
        out.append(batch['mask'].reshape(batch['mask'].shape[0] * batch['mask'].shape[1], batch['mask'].shape[2], batch['mask'].shape[3], batch['mask'].shape[4], batch['mask'].shape[5]))
        out.append(batch['keytime_mask'].reshape(batch['keytime_mask'].shape[0] * batch['keytime_mask'].shape[1], batch['keytime_mask'].shape[2], batch['keytime_mask'].shape[3], batch['keytime_mask'].shape[4], batch['keytime_mask'].shape[5]))
        fn_list = [p for p in batch['path']]
        out.append(fn_list)
        return out

    def p_losses(self, x_start, cond, t, gt, mask, keytime_mask, fns, noise=None, **kwargs):
        if self.noise_strength > 0:
            b, c, f, _, _ = x_start.shape
            offset_noise = torch.randn(b, c, f, 1, 1, device=x_start.device)
            noise = default(noise, lambda: torch.randn_like(x_start) + self.noise_strength * offset_noise)
        else:
            noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        cond['c_crossattn'][0][:, :77, :] = self.prob3_learnable_prompt
        model_output, feat_list = self.apply_model(x_noisy, t, cond, **kwargs)
        prob3_outputs = []
        for feat_i, feat in enumerate(feat_list):
            B, T, C, H, W = feat.shape
            query_vector = self.prob3_feat_mlps[feat_i](self.prob3_learnable_vector)
            query_vector = query_vector.expand(B, -1).unsqueeze(1)
            feat_reshaped = feat.permute(0, 1, 3, 4, 2).contiguous().view(B, T * H * W, C)
            scale = C ** (-0.5)
            attn_scores = torch.matmul(query_vector, feat_reshaped.transpose(-2, -1)) * scale
            attn_weights = torch.softmax(attn_scores, dim=-1)
            attended_feat = torch.matmul(attn_weights, feat_reshaped)
            attended_feat = attended_feat.squeeze(1)
            if self.collect_map_together == 0:
                prob3_feat_output = self.prob3_feat_output_mlps[feat_i](attended_feat)
                prob3_outputs.append(prob3_feat_output)
            else:
                prob3_outputs.append(attended_feat)
        if prob3_outputs:
            if self.collect_map_together == 0:
                prob3_combined = torch.stack(prob3_outputs, dim=1)
                prob3_averaged = torch.mean(prob3_combined, dim=1)
                prob3_averaged = prob3_averaged.reshape(int(prob3_averaged.shape[0] / 2), 2, prob3_averaged.shape[1])
                if self.concat_2_feat == False:
                    prob3_diff = prob3_averaged[:, 0, :] - prob3_averaged[:, 1, :]
                    gt_retitusion = self.prob3_output_projection(prob3_diff)
                else:
                    prob3_diff = torch.cat([prob3_averaged[:, 0, :], prob3_averaged[:, 1, :]], dim=1)
                    gt_retitusion = self.prob3_output_projection(prob3_diff)
            elif self.collect_map_together == 1:
                prob3_combined = torch.cat(prob3_outputs, dim=1)
                prob3_averaged = self.prob3_feat_output_mlps(prob3_combined)
                prob3_averaged = prob3_averaged.reshape(int(prob3_averaged.shape[0] / 2), 2, prob3_averaged.shape[1])
                if self.concat_2_feat == False:
                    prob3_diff = prob3_averaged[:, 0, :] - prob3_averaged[:, 1, :]
                    gt_retitusion = self.prob3_output_projection(prob3_diff)
                else:
                    prob3_diff = torch.cat([prob3_averaged[:, 0, :], prob3_averaged[:, 1, :]], dim=1)
                    gt_retitusion = self.prob3_output_projection(prob3_diff)
            else:
                assert self.collect_map_together == 2
                prob3_combined = torch.cat(prob3_outputs, dim=1)
                prob3_combined = prob3_combined.reshape(int(prob3_combined.shape[0] / 2), 2 * prob3_combined.shape[1])
                gt_retitusion = self.prob3_feat_output_mlps(prob3_combined)
        else:
            gt_retitusion = None
        bce_loss = nn.BCEWithLogitsLoss()
        device = feat_list[0].device
        for key in gt:
            if isinstance(gt[key], torch.Tensor):
                gt[key] = gt[key].to(device)
            elif isinstance(gt[key], list):
                gt[key] = [item.to(device) if isinstance(item, torch.Tensor) else item for item in gt[key]]
        loss_gt_retitusion = 0 if gt_retitusion is None else bce_loss(gt_retitusion[:, 0], gt['gt_restitution'].float())
        if gt_retitusion is not None:
            print(loss_gt_retitusion)
        fn_2_pred = {}
        for fn_i in range(gt_retitusion.shape[0]):
            fn_2_pred[fns[fn_i]] = {}
            fn_2_pred[fns[fn_i]]['pred'] = gt_retitusion[fn_i][0].detach().cpu().numpy().tolist()
            fn_2_pred[fns[fn_i]]['gt'] = gt['gt_restitution'][fn_i].detach().cpu().numpy().tolist()
        with open(os.path.join(self.save_result_dir, fns[0] + '.json'), 'w') as dump_f:
            json.dump(fn_2_pred, dump_f)
        loss = loss_gt_retitusion
        loss_dict = {}
        loss_dict.update({'total_loss': loss, 'loss_gt_retitusion': loss_gt_retitusion})
        for key_i in loss_dict.keys():
            if isinstance(loss_dict[key_i], torch.Tensor) and torch.isnan(loss_dict[key_i]):
                loss_dict[key_i] = torch.tensor(0.001, device=device, requires_grad=True)
                print('NaN loss detected, using fallback value')
        return (loss, loss_dict)

class PhysicsPredictionVJEPAv87_original(LatentVisualDiffusionv87):

    def __init__(self, vjepa_hidden_dim: int, num_output_features: int, *args, **kwargs):
        self.use_red_circle = kwargs.pop('use_red_circle', False)
        self.linear_start = kwargs.pop('linear_start', 0.00085)
        self.linear_end = kwargs.pop('linear_end', 0.012)
        self.cosine_s = kwargs.pop('cosine_s', 0.008)
        self.num_timesteps = kwargs.pop('num_timesteps', 1000)
        super().__init__(*args, **kwargs)
        self.vjepa_hidden_dim = vjepa_hidden_dim
        self.num_output_features = num_output_features
        self.parameterization = 'vjepa_direct'
        self.noise_strength = 0
        self.l1_loss = nn.L1Loss()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.prob3_feat_mlps = nn.ModuleList([nn.Linear(self.prob3_dim, self.vjepa_hidden_dim) for _ in range(self.num_output_features)])
        vjepa_total_feat_dim = self.num_output_features * self.vjepa_hidden_dim
        if self.collect_map_together == 0:
            self.prob3_feat_output_mlps = nn.ModuleList([nn.Linear(self.vjepa_hidden_dim, self.prob3_dim) for _ in range(self.num_output_features)])
        elif self.collect_map_together == 1:
            self.prob3_feat_output_mlps = nn.Linear(vjepa_total_feat_dim, self.prob3_dim)
        else:
            assert self.collect_map_together == 2
            self.prob3_feat_output_mlps = nn.Linear(vjepa_total_feat_dim * 2, 1)

    def get_batch_input_for_vjepa(self, batch: dict) -> tuple:
        x = super().get_input(batch, self.first_stage_key)
        red_video = super().get_input(batch, 'mask')
        if self.use_red_circle:
            x = red_video
            print('use red circle video')
        if len(x.shape) == 6:
            x = x.reshape(x.shape[0] * x.shape[1], x.shape[2], x.shape[3], x.shape[4], x.shape[5])
        gt = batch['gt_ann']
        return (x, gt)

    def register_schedule(self, *args, **kwargs):
        print('PhysicsPredictionVJEPAv87: Bypassing DDPM schedule registration.')
        self.num_timesteps = kwargs.get('num_timesteps', 1000)
        pass

    def calculate_vjepa_physics_loss(self, feat_list: list, gt: dict, return_prediction: bool=False) -> tuple:
        device = feat_list[0].device
        for key in gt:
            if isinstance(gt[key], torch.Tensor):
                gt[key] = gt[key].to(device)
            elif isinstance(gt[key], list):
                gt[key] = [item.to(device) if isinstance(item, torch.Tensor) else item for item in gt[key]]
        bce_loss = nn.BCEWithLogitsLoss()
        prob3_outputs = []
        if self.collect_map_together in [0, 1]:
            for feat_i, feat in enumerate(feat_list):
                B, T, C, H, W = feat.shape
                query_vector = self.prob3_feat_mlps[feat_i](self.prob3_learnable_vector)
                query_vector = query_vector.expand(B, -1).unsqueeze(1)
                feat_reshaped = feat.permute(0, 1, 3, 4, 2).contiguous().view(B, T * H * W, C)
                scale = C ** (-0.5)
                attn_scores = torch.matmul(query_vector, feat_reshaped.transpose(-2, -1)) * scale
                attn_weights = torch.softmax(attn_scores, dim=-1)
                attended_feat = torch.matmul(attn_weights, feat_reshaped).squeeze(1)
                if self.collect_map_together == 0:
                    prob3_feat_output = self.prob3_feat_output_mlps[feat_i](attended_feat)
                    prob3_outputs.append(prob3_feat_output)
                else:
                    prob3_outputs.append(attended_feat)
        elif self.collect_map_together == 2:
            for feat_i, feat in enumerate(feat_list):
                B, T, C, H, W = feat.shape
                query_vector = self.prob3_feat_mlps[feat_i](self.prob3_learnable_vector)
                query_vector = query_vector.expand(B, -1).unsqueeze(1)
                feat_reshaped = feat.permute(0, 1, 3, 4, 2).contiguous().view(B, T * H * W, C)
                scale = C ** (-0.5)
                attn_scores = torch.matmul(query_vector, feat_reshaped.transpose(-2, -1)) * scale
                attn_weights = torch.softmax(attn_scores, dim=-1)
                attended_feat = torch.matmul(attn_weights, feat_reshaped).squeeze(1)
                prob3_outputs.append(attended_feat)
        gt_retitusion = None
        if prob3_outputs:
            if self.collect_map_together == 0:
                prob3_combined = torch.stack(prob3_outputs, dim=1)
                prob3_averaged = torch.mean(prob3_combined, dim=1)
                prob3_averaged = prob3_averaged.reshape(int(prob3_averaged.shape[0] / 2), 2, -1)
                if not self.concat_2_feat:
                    prob3_diff = prob3_averaged[:, 0, :] - prob3_averaged[:, 1, :]
                else:
                    prob3_diff = torch.cat([prob3_averaged[:, 0, :], prob3_averaged[:, 1, :]], dim=1)
                gt_retitusion = self.prob3_output_projection(prob3_diff)
            elif self.collect_map_together == 1:
                prob3_combined = torch.cat(prob3_outputs, dim=1)
                prob3_averaged = self.prob3_feat_output_mlps(prob3_combined)
                prob3_averaged = prob3_averaged.reshape(int(prob3_averaged.shape[0] / 2), 2, -1)
                if not self.concat_2_feat:
                    prob3_diff = prob3_averaged[:, 0, :] - prob3_averaged[:, 1, :]
                else:
                    prob3_diff = torch.cat([prob3_averaged[:, 0, :], prob3_averaged[:, 1, :]], dim=1)
                gt_retitusion = self.prob3_output_projection(prob3_diff)
            elif self.collect_map_together == 2:
                prob3_combined = torch.cat(prob3_outputs, dim=1)
                prob3_reshaped = prob3_combined.reshape(int(prob3_combined.shape[0] / 2), 2 * prob3_combined.shape[1])
                gt_retitusion = self.prob3_feat_output_mlps(prob3_reshaped)
        loss_gt_retitusion = 0 if gt_retitusion is None else bce_loss(gt_retitusion[:, 0], gt['gt_restitution'].float())
        loss = loss_gt_retitusion
        loss_dict = {'total_loss': loss, 'loss_gt_retitusion': loss_gt_retitusion}
        for key_i in loss_dict.keys():
            if isinstance(loss_dict[key_i], torch.Tensor) and (torch.isnan(loss_dict[key_i]) or torch.isinf(loss_dict[key_i])):
                mainlogger.info(f'NaN or Inf loss detected in {key_i}, using fallback value')
                loss_dict[key_i] = torch.tensor(0.001, device=device, requires_grad=True)
        if return_prediction:
            return (loss, loss_dict, gt_retitusion)
        return (loss, loss_dict)

    def calculate_vjepa_physics_loss_train(self, feat_list: list, gt: dict) -> tuple[torch.Tensor, dict]:
        attended_features = []
        for i, feat in enumerate(feat_list):
            B, T, C, H, W = feat.shape
            query_vector = self.prob3_feat_mlps[i](self.prob3_learnable_vector)
            q = query_vector.expand(B, -1).unsqueeze(1)
            feat_reshaped = feat.permute(0, 1, 3, 4, 2).contiguous().view(B, T * H * W, C)
            scale = C ** (-0.5)
            attn_scores = torch.matmul(q, feat_reshaped.transpose(-2, -1)) * scale
            attn_weights = torch.softmax(attn_scores, dim=-1)
            attended_feat = torch.matmul(attn_weights, feat_reshaped).squeeze(1)
            attended_features.append(attended_feat)
        gt_prediction = None
        gt_retitusion = None
        if attended_features:
            if self.collect_map_together == 0:
                prob3_outputs = [self.prob3_feat_output_mlps[i](att_feat) for i, att_feat in enumerate(attended_features)]
                prob3_averaged = torch.mean(torch.stack(prob3_outputs, dim=1), dim=1)
                real_B = prob3_averaged.shape[0] // 2
                prob3_averaged = prob3_averaged.reshape(real_B, 2, prob3_averaged.shape[1])
                prob3_diff = prob3_averaged[:, 0, :] - prob3_averaged[:, 1, :] if not self.concat_2_feat else torch.cat([prob3_averaged[:, 0, :], prob3_averaged[:, 1, :]], dim=1)
                gt_prediction = self.prob3_output_projection(prob3_diff)
            elif self.collect_map_together == 1:
                prob3_combined = torch.cat(attended_features, dim=1)
                prob3_averaged = self.prob3_feat_output_mlps(prob3_combined)
                real_B = prob3_averaged.shape[0] // 2
                prob3_averaged = prob3_averaged.reshape(real_B, 2, prob3_averaged.shape[1])
                prob3_diff = prob3_averaged[:, 0, :] - prob3_averaged[:, 1, :] if not self.concat_2_feat else torch.cat([prob3_averaged[:, 0, :], prob3_averaged[:, 1, :]], dim=1)
                gt_prediction = self.prob3_output_projection(prob3_diff)
            else:
                assert self.collect_map_together == 2
                prob3_combined = torch.cat(attended_features, dim=1)
                real_B = prob3_combined.shape[0] // 2
                prob3_combined = prob3_combined.reshape(real_B, 2 * prob3_combined.shape[1])
                gt_prediction = self.prob3_feat_output_mlps(prob3_combined)
        gt_retitusion = gt_prediction
        bce_loss = nn.BCEWithLogitsLoss()
        device = feat_list[0].device
        for key in gt:
            if isinstance(gt[key], torch.Tensor):
                gt[key] = gt[key].to(device)
            elif isinstance(gt[key], list):
                gt[key] = [item.to(device) if isinstance(item, torch.Tensor) else item for item in gt[key]]
        loss_gt_retitusion = 0 if gt_retitusion is None else bce_loss(gt_retitusion[:, 0], gt['gt_restitution'].float())
        if gt_retitusion is not None:
            print(loss_gt_retitusion)
        loss = loss_gt_retitusion
        loss_dict = {}
        loss_dict.update({'total_loss': loss, 'loss_gt_retitusion': loss_gt_retitusion})
        for key_i in loss_dict.keys():
            if isinstance(loss_dict[key_i], torch.Tensor) and torch.isnan(loss_dict[key_i]):
                loss_dict[key_i] = torch.tensor(0.001, device=device, requires_grad=True)
                print('NaN loss detected, using fallback value')
        return (loss, loss_dict)

    def shared_step(self, batch: dict, **kwargs) -> tuple[torch.Tensor, dict]:
        x, gt = self.get_batch_input_for_vjepa(batch)
        t = torch.zeros(x.shape[0], device=self.device, dtype=torch.long)
        model_output, feat_list = self.model.diffusion_model(x, t=t, context=None)
        if feat_list is None or not feat_list:
            loss = torch.tensor(0.0, device=self.device, requires_grad=True)
            loss_dict = {'total_loss': loss.detach()}
        else:
            loss, loss_dict = self.calculate_vjepa_physics_loss(feat_list, gt)
        return (loss, loss_dict)

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        loss, loss_dict = self.shared_step(batch)
        prefix = 'train'
        log_dict_prefixed = {f'{prefix}/{k}': v for k, v in loss_dict.items()}
        self.log_dict(log_dict_prefixed, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log('global_step', self.global_step, prog_bar=True, logger=True, on_step=True, on_epoch=False)
        if (batch_idx + 1) % self.log_every_t == 0:
            mainlogger.info(f'batch:{batch_idx}|epoch:{self.current_epoch} [globalstep:{self.global_step}]: loss={loss}')
        return loss

    def validation_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        return self.shared_step(batch)

    def p_losses(self, *args, **kwargs):
        raise NotImplementedError('p_losses is obsolete for this model.')

    def get_batch_input(self, *args, **kwargs):
        raise NotImplementedError('Use get_batch_input_for_vjepa for this model.')

class PhysicsPredictionVJEPAv87_attentive_probe(PhysicsPredictionVJEPAv87_original):

    def __init__(self, vjepa_hidden_dim: int, num_output_features: int=1, attentive_config: dict=None, use_all_feat_in_attentive_probe: bool=False, *args, **kwargs):
        super().__init__(vjepa_hidden_dim, num_output_features, *args, **kwargs)
        self.vjepa_hidden_dim = vjepa_hidden_dim
        self.num_output_features = num_output_features
        self.use_all_feat_in_attentive_probe = use_all_feat_in_attentive_probe
        self.linear_start = kwargs.get('linear_start', 0.00085)
        self.linear_end = kwargs.get('linear_end', 0.012)
        self.cosine_s = kwargs.get('cosine_s', 0.008)
        self.num_timesteps = kwargs.get('num_timesteps', 1000)
        self.parameterization = 'vjepa_direct'
        self.noise_strength = 0
        self.l1_loss = nn.L1Loss()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.prob3_feat_mlps = nn.ModuleList([nn.Linear(self.vjepa_hidden_dim, self.vjepa_hidden_dim) for _ in range(self.num_output_features)])
        default_attentive_config = {'verb_classes': {}, 'noun_classes': {}, 'action_classes': {f'restitution_{i}': i for i in range(1)}, 'embed_dim': vjepa_hidden_dim, 'num_heads': 8, 'depth': 4, 'use_activation_checkpointing': False}
        if attentive_config:
            default_attentive_config.update(attentive_config)
        self.attentive_classifier = AttentiveClassifier(**default_attentive_config)
        mlp_input_dim = vjepa_hidden_dim * 2 if kwargs.get('concat_2_feat', False) else vjepa_hidden_dim
        self.final_projection_feature_mlp = nn.Linear(mlp_input_dim, 1)
        mainlogger.info('PhysicsPredictionVJEPAv87_AttentiveClassifier: using AttentiveClassifier instead of MLP probing.')

    def calculate_vjepa_physics_loss(self, feat_list: list, gt: dict, return_prediction: bool=False) -> tuple:
        if feat_list is None:
            device = self.device
        else:
            device = feat_list.device
        for key in gt:
            if isinstance(gt[key], torch.Tensor):
                gt[key] = gt[key].to(device)
            elif isinstance(gt[key], list):
                gt[key] = [item.to(device) if isinstance(item, torch.Tensor) else item for item in gt[key]]
        bce_loss = nn.BCEWithLogitsLoss()
        if feat_list is None:
            gt_retitusion = None
        else:
            last_layer_feat = feat_list
            classifier_output = self.attentive_classifier(last_layer_feat)
            action_logits = classifier_output['skip_projection_feature']
            B = action_logits.shape[0]
            if B % 2 == 0:
                real_batch_size = B // 2
                action_reshaped = action_logits.reshape(real_batch_size, 2, -1)
                if hasattr(self, 'concat_2_feat') and self.concat_2_feat:
                    gt_retitusion = torch.cat([action_reshaped[:, 0, :], action_reshaped[:, 1, :]], dim=1)
                else:
                    gt_retitusion = action_reshaped[:, 0, :] - action_reshaped[:, 1, :]
            else:
                gt_retitusion = action_logits
        gt_retitusion = self.final_projection_feature_mlp(gt_retitusion)
        loss_gt_retitusion = 0 if gt_retitusion is None else bce_loss(gt_retitusion[:, 0], gt['gt_restitution'].float())
        if self.training:
            print('gt_retitusion: ', gt_retitusion)
            print('gt: ', gt['gt_restitution'])
        loss = loss_gt_retitusion
        loss_dict = {'total_loss': loss, 'loss_gt_retitusion': loss_gt_retitusion}
        for key_i in loss_dict.keys():
            if isinstance(loss_dict[key_i], torch.Tensor) and (torch.isnan(loss_dict[key_i]) or torch.isinf(loss_dict[key_i])):
                mainlogger.info(f'NaN or Inf loss detected in {key_i}, using fallback value')
                loss_dict[key_i] = torch.tensor(0.001, device=device, requires_grad=True)
        if return_prediction:
            return (loss, loss_dict, gt_retitusion)
        return (loss, loss_dict)

    def shared_step(self, batch: dict, **kwargs) -> tuple[torch.Tensor, dict]:
        x, gt = self.get_batch_input_for_vjepa(batch)
        t = torch.zeros(x.shape[0], device=self.device, dtype=torch.long)
        model_output, feat_list = self.model.diffusion_model(x, t=t, context=None)
        if self.use_all_feat_in_attentive_probe:
            if feat_list and len(feat_list) > 0:
                target_shape = model_output.shape
                B, N_patches, D = target_shape
                transformed_feats = []
                for i, feat in enumerate(feat_list):
                    B_feat, T, C, H, W = feat.shape
                    feat_reshaped = feat.permute(0, 2, 1, 3, 4).contiguous().view(B_feat, C, T * H * W).transpose(1, 2)
                    feat_idx = min(i, len(self.prob3_feat_mlps) - 1)
                    feat_transformed = self.prob3_feat_mlps[feat_idx](feat_reshaped)
                    transformed_feats.append(feat_transformed)
                model_output = torch.stack(transformed_feats, dim=0).mean(dim=0)
        else:
            model_output = model_output
        if feat_list is None or not feat_list:
            loss = torch.tensor(0.0, device=self.device, requires_grad=True)
            loss_dict = {'total_loss': loss.detach()}
        else:
            loss, loss_dict = self.calculate_vjepa_physics_loss(model_output, gt)
        return (loss, loss_dict)

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        loss, loss_dict = self.shared_step(batch)
        if not torch.isfinite(loss):
            mainlogger.warning(f'Non-finite loss detected at batch {batch_idx}: {loss}')
            loss = torch.tensor(0.001, device=loss.device, requires_grad=True)
        prefix = 'train'
        log_dict_prefixed = {f'{prefix}/{k}': v for k, v in loss_dict.items()}
        if (batch_idx + 1) % self.log_every_t == 0:
            mainlogger.info(f'batch:{batch_idx}|epoch:{self.current_epoch} [globalstep:{self.global_step}]: loss={loss}')
        self.log_dict(log_dict_prefixed, prog_bar=True, logger=True, on_step=True, on_epoch=False)
        return loss

class PhysicsPredictionVJEPAv87Test_original(PhysicsPredictionVJEPAv87_original):

    def __init__(self, *args, save_result_dir=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_result_dir = save_result_dir
        if self.save_result_dir is not None:
            os.makedirs(self.save_result_dir, exist_ok=True)
            mainlogger.info(f'Test results for PhysicsPredictionVJEPAv87Test_original will be saved to {self.save_result_dir}')

    def get_batch_input_for_vjepa(self, batch: dict) -> tuple:
        x, gt = super().get_batch_input_for_vjepa(batch)
        fns = batch['path']
        return (x, gt, fns)

    def run_prediction_step(self, batch: dict) -> tuple:
        x, gt, fns = self.get_batch_input_for_vjepa(batch)
        t = torch.zeros(x.shape[0], device=self.device, dtype=torch.long)
        _, feat_list = self.model.diffusion_model(x, timesteps=t, context=None)
        if feat_list is None or not feat_list:
            loss = torch.tensor(0.0, device=self.device, requires_grad=True)
            loss_dict = {'total_loss': loss.detach()}
            predictions = None
        else:
            loss, loss_dict, predictions = self.calculate_vjepa_physics_loss(feat_list, gt, return_prediction=True)
        return (loss, loss_dict, predictions, gt, fns)

    @torch.no_grad()
    def validation_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        loss, loss_dict, predictions, ground_truth, fns = self.run_prediction_step(batch)
        prefix = 'val'
        num_pairs = len(fns) // 2 if fns else 0
        self.log_dict(loss_dict, prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True, batch_size=num_pairs)
        if self.save_result_dir and predictions is not None and fns:
            fn_2_pred = {}
            for i in range(predictions.shape[0]):
                fn_2_pred[fns[i]] = {}
                fn_2_pred[fns[i]]['pred'] = predictions[i][0].detach().cpu().numpy().tolist()
                fn_2_pred[fns[i]]['gt'] = ground_truth['gt_restitution'][i].detach().cpu().numpy().tolist()
            save_path = os.path.join(self.save_result_dir, fns[0] + '.json')
            with open(save_path, 'w') as dump_f:
                json.dump(fn_2_pred, dump_f)
        print('save_result_dir: ', self.save_result_dir)
        return loss

class PhysicsPredictionVJEPAv87Test_attentive_probe(PhysicsPredictionVJEPAv87_attentive_probe):

    def __init__(self, *args, save_result_dir=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_result_dir = save_result_dir
        if self.save_result_dir is not None:
            os.makedirs(self.save_result_dir, exist_ok=True)
            mainlogger.info(f'Test results for PhysicsPredictionVJEPAv87Test_original will be saved to {self.save_result_dir}')

    def get_batch_input_for_vjepa(self, batch: dict) -> tuple:
        x, gt = super().get_batch_input_for_vjepa(batch)
        fns = batch['path']
        return (x, gt, fns)

    def run_prediction_step(self, batch: dict) -> tuple:
        x, gt, fns = self.get_batch_input_for_vjepa(batch)
        t = torch.zeros(x.shape[0], device=self.device, dtype=torch.long)
        model_output, feat_list = self.model.diffusion_model(x, timesteps=t, context=None)
        if self.use_all_feat_in_attentive_probe:
            if feat_list and isinstance(feat_list, list):
                transformed_features = []
                for i, feat in enumerate(feat_list):
                    B, T, C, H, W = feat.shape
                    feat_reshaped = feat.permute(0, 1, 3, 4, 2).contiguous().view(B, T * H * W, C)
                    feat_idx = min(i, len(self.prob3_feat_mlps) - 1)
                    feat_transformed = self.prob3_feat_mlps[feat_idx](feat_reshaped)
                    transformed_features.append(feat_transformed)
                model_output = torch.stack(transformed_features, dim=0).mean(dim=0)
            else:
                model_output = model_output
        else:
            model_output = model_output
        if feat_list is None or not feat_list:
            loss = torch.tensor(0.0, device=self.device, requires_grad=True)
            loss_dict = {'total_loss': loss.detach()}
            predictions = None
        else:
            loss, loss_dict, predictions = self.calculate_vjepa_physics_loss(model_output, gt, return_prediction=True)
        return (loss, loss_dict, predictions, gt, fns)

    @torch.no_grad()
    def validation_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        loss, loss_dict, predictions, ground_truth, fns = self.run_prediction_step(batch)
        prefix = 'val'
        num_pairs = len(fns) // 2 if fns else 0
        self.log_dict(loss_dict, prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True, batch_size=num_pairs)
        if self.save_result_dir and predictions is not None and fns:
            fn_2_pred = {}
            for i in range(predictions.shape[0]):
                fn_2_pred[fns[i]] = {}
                fn_2_pred[fns[i]]['pred'] = predictions[i][0].detach().cpu().numpy().tolist()
                fn_2_pred[fns[i]]['gt'] = ground_truth['gt_restitution'][i].detach().cpu().numpy().tolist()
            save_path = os.path.join(self.save_result_dir, fns[0] + '.json')
            with open(save_path, 'w') as dump_f:
                json.dump(fn_2_pred, dump_f)
        print('save_result_dir: ', self.save_result_dir)
        return loss

class PhysicsPredictionVJEPAv87_v1(LatentVisualDiffusionv87):

    def __init__(self, vjepa_hidden_dim: int, num_output_features: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.linear_start = kwargs.get('linear_start', 0.00085)
        self.linear_end = kwargs.get('linear_end', 0.012)
        self.cosine_s = kwargs.get('cosine_s', 0.008)
        mainlogger.info(f'Initializing PhysicsPredictionVJEPAv87: Setting up VJEPA-specific components.')
        self.vjepa_predictor = self.model.diffusion_model
        uniform_feat_dim = vjepa_hidden_dim
        self.prob3_feat_mlps = nn.ModuleList([nn.Linear(self.prob3_dim, uniform_feat_dim) for _ in range(num_output_features)])
        if self.collect_map_together in [0, 1]:
            if self.collect_map_together == 0:
                self.prob3_feat_output_mlps = nn.ModuleList([nn.Linear(uniform_feat_dim, self.prob3_dim) for _ in range(num_output_features)])
            else:
                self.prob3_feat_output_mlps = nn.Linear(uniform_feat_dim * num_output_features, self.prob3_dim)
            output_proj_dim = self.prob3_dim * 2 if self.concat_2_feat else self.prob3_dim
            self.prob3_output_projection = nn.Linear(output_proj_dim, 1)
        elif self.collect_map_together == 2:
            self.prob3_feat_output_mlps = nn.Linear(uniform_feat_dim * num_output_features * 2, 1)
        self.prob3_learnable_vector = nn.Parameter(torch.randn(1, self.prob3_dim))

    def get_batch_input_for_vjepa(self, batch: dict, return_full_batch: bool=False) -> tuple:
        _, _, _, cond, gt, mask, _, keytime_mask, fns = super().get_batch_input(batch, random_uncond=False)
        video_input = batch['video_input']
        if return_full_batch:
            return (video_input, cond, gt, mask, keytime_mask, fns)
        else:
            return (video_input, gt)

    def register_schedule(self, *args, **kwargs):
        mainlogger.info('PhysicsPredictionVJEPAv87: Bypassing DDPM schedule registration.')
        self.num_timesteps = kwargs.get('num_timesteps', 1000)
        pass

    def calculate_vjepa_physics_loss(self, feat_list: list, gt: dict, return_prediction: bool=False) -> tuple:
        device = feat_list[0].device
        for key in gt:
            if isinstance(gt[key], torch.Tensor):
                gt[key] = gt[key].to(device)
        bce_loss = nn.BCEWithLogitsLoss()
        attended_features = []
        for i, feat in enumerate(feat_list):
            B, T, C, H, W = feat.shape
            query_vector = self.prob3_feat_mlps[i](self.prob3_learnable_vector)
            q = query_vector.expand(B, -1).unsqueeze(1)
            feat_reshaped = feat.permute(0, 1, 3, 4, 2).contiguous().view(B, T * H * W, C)
            scale = C ** (-0.5)
            attn_scores = torch.matmul(q, feat_reshaped.transpose(-2, -1)) * scale
            attn_weights = torch.softmax(attn_scores, dim=-1)
            attended_feat = torch.matmul(attn_weights, feat_reshaped).squeeze(1)
            attended_features.append(attended_feat)
        gt_prediction = None
        if attended_features:
            if self.collect_map_together == 0:
                prob3_outputs = [self.prob3_feat_output_mlps[i](att_feat) for i, att_feat in enumerate(attended_features)]
                prob3_averaged = torch.mean(torch.stack(prob3_outputs, dim=1), dim=1)
            elif self.collect_map_together == 1:
                prob3_combined = torch.cat(attended_features, dim=1)
                prob3_averaged = self.prob3_feat_output_mlps(prob3_combined)
            if self.collect_map_together in [0, 1]:
                real_B = prob3_averaged.shape[0] // 2
                prob3_averaged = prob3_averaged.reshape(real_B, 2, prob3_averaged.shape[1])
                prob3_diff = prob3_averaged[:, 0, :] - prob3_averaged[:, 1, :] if not self.concat_2_feat else torch.cat([prob3_averaged[:, 0, :], prob3_averaged[:, 1, :]], dim=1)
                gt_prediction = self.prob3_output_projection(prob3_diff)
            else:
                prob3_combined = torch.cat(attended_features, dim=1)
                real_B = prob3_combined.shape[0] // 2
                prob3_combined = prob3_combined.reshape(real_B, 2 * prob3_combined.shape[1])
                gt_prediction = self.prob3_feat_output_mlps(prob3_combined)
        loss_val = torch.tensor(0.0, device=device)
        if gt_prediction is not None:
            loss_val = bce_loss(gt_prediction.squeeze(), gt['gt_restitution'].float())
        loss_dict = {'total_loss': loss_val.item(), 'loss_restitution': loss_val.item()}
        if torch.isnan(loss_val) or torch.isinf(loss_val):
            mainlogger.warning(f'NaN or Inf loss detected, using fallback value 0.001')
            loss_val = torch.tensor(0.001, device=device, requires_grad=True)
        if return_prediction:
            return (loss_val, loss_dict, gt_prediction)
        return (loss_val, loss_dict)

    def calculate_vjepa_physics_loss_train(self, feat_list: list, gt: dict) -> tuple[torch.Tensor, dict]:
        loss, loss_dict, _ = self.calculate_vjepa_physics_loss(feat_list, gt, return_prediction=True)
        return (loss, loss_dict)

    def shared_step(self, batch: dict, **kwargs) -> tuple[torch.Tensor, dict]:
        video_input, gt = self.get_batch_input_for_vjepa(batch, return_full_batch=False)
        feat_list = self.vjepa_predictor(video_input)
        loss, loss_dict = self.calculate_vjepa_physics_loss_train(feat_list, gt)
        return (loss, loss_dict)

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        loss, loss_dict = self.shared_step(batch)
        self.log('train_loss', loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log_dict(loss_dict, prog_bar=False, logger=True, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        with torch.no_grad():
            loss, loss_dict = self.shared_step(batch)
        self.log('val_loss', loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log_dict(loss_dict, prog_bar=False, logger=True, on_step=True, on_epoch=True)
        return loss

    def p_losses(self, *args, **kwargs):
        raise NotImplementedError('p_losses is obsolete for this model.')

class PhysicsPredictionVJEPAv87(LatentVisualDiffusionv87):

    def __init__(self, vjepa_cfg, vjepa_predictor_cfg, vjepa_hidden_dim, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.vjepa = instantiate_from_config(vjepa_cfg)
        self.vjepa_predictor = instantiate_from_config(vjepa_predictor_cfg)
        self.physics_head = torch.nn.Linear(vjepa_hidden_dim, 1)

    def register_schedule(self, beta_schedule='linear', timesteps=1000, linear_start=0.0001, linear_end=0.02, cosine_s=0.008):
        self.num_timesteps = timesteps
        self.linear_start = linear_start
        self.linear_end = linear_end

    @torch.no_grad()
    def get_batch_input_for_vjepa(self, batch: dict, return_full_batch: bool=False) -> tuple:
        video_input = self.get_input(batch, self.first_stage_key)
        cond = None
        if return_full_batch:
            gt = batch['gt_ann']
            mask = batch['mask']
            keytime_mask = batch['keytime_mask']
            if 'rel_path' in batch:
                fns = [os.path.basename(p) for p in batch['rel_path']]
            else:
                fns = [f'sample_{i}' for i in range(video_input.shape[0])]
            return (video_input, cond, gt, mask, keytime_mask, fns)
        return (video_input, cond)

    def predict_physics_from_vjepa(self, feat_list: list, mask: torch.Tensor) -> dict:
        vjepa_feat = feat_list[0]
        restitution = self.physics_head(vjepa_feat).squeeze(-1)
        return {'restitution': restitution}

    def shared_step(self, batch: dict) -> tuple:
        video_input, _ = self.get_batch_input_for_vjepa(batch)
        gt = batch['gt_ann']
        mask = batch['mask']
        batch_size = video_input.shape[0]
        timesteps = torch.zeros(batch_size, device=self.device).long()
        feat_list = self.vjepa_predictor(video_input, timesteps)
        predictions = self.predict_physics_from_vjepa(feat_list, mask)
        loss_func = torch.nn.L1Loss()
        loss = loss_func(predictions['restitution'], gt)
        loss_dict = {'train/l1_loss': loss.item()}
        return (loss, loss_dict)

    def run_prediction_step(self, batch: dict) -> tuple:
        video_input, cond, gt, mask, keytime_mask, fns = self.get_batch_input_for_vjepa(batch, return_full_batch=True)
        batch_size = video_input.shape[0]
        timesteps = torch.zeros(batch_size, device=self.device).long()
        feat_list = self.vjepa_predictor(video_input, timesteps)
        predictions = self.predict_physics_from_vjepa(feat_list, mask)
        ground_truth = {'restitution': gt}
        loss_func = torch.nn.L1Loss()
        loss = loss_func(predictions['restitution'], ground_truth['restitution'])
        loss_dict = {'val/l1_loss': loss.item()}
        return (loss, loss_dict, predictions, ground_truth, fns)

    def training_step(self, batch, batch_idx):
        loss, loss_dict = self.shared_step(batch)
        self.log_dict(loss_dict, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        loss, loss_dict, _, _, _ = self.run_prediction_step(batch)
        self.log_dict(loss_dict, prog_bar=False, logger=True, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.vjepa_predictor.parameters()) + list(self.physics_head.parameters())
        opt = torch.optim.AdamW(params, lr=lr)
        return opt

class DiffusionWrapper(pl.LightningModule):

    def __init__(self, diff_model_config, conditioning_key):
        super().__init__()
        self.diffusion_model = instantiate_from_config(diff_model_config)
        self.conditioning_key = conditioning_key

    def forward(self, x, t, c_concat: list=None, c_crossattn: list=None, c_adm=None, s=None, mask=None, **kwargs):
        if self.conditioning_key is None:
            out = self.diffusion_model(x, t)
        elif self.conditioning_key == 'concat':
            xc = torch.cat([x] + c_concat, dim=1)
            out = self.diffusion_model(xc, t, **kwargs)
        elif self.conditioning_key == 'crossattn':
            cc = torch.cat(c_crossattn, 1)
            out = self.diffusion_model(x, t, context=cc, **kwargs)
        elif self.conditioning_key == 'hybrid':
            xc = torch.cat([x] + c_concat, dim=1)
            cc = torch.cat(c_crossattn, 1)
            out = self.diffusion_model(xc, t, context=cc, **kwargs)
        elif self.conditioning_key == 'resblockcond':
            cc = c_crossattn[0]
            out = self.diffusion_model(x, t, context=cc)
        elif self.conditioning_key == 'adm':
            cc = c_crossattn[0]
            out = self.diffusion_model(x, t, y=cc)
        elif self.conditioning_key == 'hybrid-adm':
            assert c_adm is not None
            xc = torch.cat([x] + c_concat, dim=1)
            cc = torch.cat(c_crossattn, 1)
            out = self.diffusion_model(xc, t, context=cc, y=c_adm, **kwargs)
        elif self.conditioning_key == 'hybrid-time':
            assert s is not None
            xc = torch.cat([x] + c_concat, dim=1)
            cc = torch.cat(c_crossattn, 1)
            out = self.diffusion_model(xc, t, context=cc, s=s)
        elif self.conditioning_key == 'concat-time-mask':
            xc = torch.cat([x] + c_concat, dim=1)
            out = self.diffusion_model(xc, t, context=None, s=s, mask=mask)
        elif self.conditioning_key == 'concat-adm-mask':
            if c_concat is not None:
                xc = torch.cat([x] + c_concat, dim=1)
            else:
                xc = x
            out = self.diffusion_model(xc, t, context=None, y=s, mask=mask)
        elif self.conditioning_key == 'hybrid-adm-mask':
            cc = torch.cat(c_crossattn, 1)
            if c_concat is not None:
                xc = torch.cat([x] + c_concat, dim=1)
            else:
                xc = x
            out = self.diffusion_model(xc, t, context=cc, y=s, mask=mask)
        elif self.conditioning_key == 'hybrid-time-adm':
            assert c_adm is not None
            xc = torch.cat([x] + c_concat, dim=1)
            cc = torch.cat(c_crossattn, 1)
            out = self.diffusion_model(xc, t, context=cc, s=s, y=c_adm)
        elif self.conditioning_key == 'crossattn-adm':
            assert c_adm is not None
            cc = torch.cat(c_crossattn, 1)
            out = self.diffusion_model(x, t, context=cc, y=c_adm)
        else:
            raise NotImplementedError()
        return out
