"""
wild mixture of
https://github.com/openai/improved-diffusion/blob/e94489283bb876ac1477d5dd7709bbbd2d9902ce/improved_diffusion/gaussian_diffusion.py
https://github.com/lucidrains/denoising-diffusion-pytorch/blob/7706bdfc6f527f58d33f84b7b522e61e6e3164b3/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py
https://github.com/CompVis/taming-transformers
-- merci
"""

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
from lvdm.models.p_decoder import PDecoder, PDecoderSupElasticity, PDecoderSupElasticityv1, PDecoderSupElasticityv2, PDecoderSupElasticityv2512, PDecoderSupElasticityv3, PDecoderSupElasticityv3512, PDecoderSupElasticityv4, PDecoderFC1, PDecoderFC2,  PDecoderSigmoid, PDecoderCondMask, PDecoderAdv, PDecoderTemporalConv, PDecoderTemporalGRU, PDecoderProbev1, PDecoderProbev2, PDecoderProbev3
from lvdm.distributions import DiagonalGaussianDistribution
from lvdm.models.utils_diffusion import make_beta_schedule, rescale_zero_terminal_snr
from lvdm.basics import disabled_train
from lvdm.common import (
    extract_into_tensor,
    noise_like,
    exists,
    default
)

import ipdb
import os
import json
import re
import inspect
from lvdm.models.vjepa2_unet import VJEPA2_Direct_Wrapper
from lvdm.models.attentive_probe import AttentiveClassifier



__conditioning_keys__ = {'concat': 'c_concat',
                         'crossattn': 'c_crossattn',
                         'adm': 'y'}

class DDPM(pl.LightningModule):
    # classic DDPM with Gaussian diffusion, in image space
    def __init__(self,
                 unet_config,
                 timesteps=1000,
                 beta_schedule="linear",
                 loss_type="l2",
                 ckpt_path=None,
                 ignore_keys=[],
                 load_only_unet=False,
                 monitor=None,
                 use_ema=True,
                 first_stage_key="image",
                 image_size=256,
                 channels=3,
                 log_every_t=100,
                 clip_denoised=True,
                 linear_start=1e-4,
                 linear_end=2e-2,
                 cosine_s=8e-3,
                 given_betas=None,
                 original_elbo_weight=0.,
                 v_posterior=0.,  # weight for choosing posterior variance as sigma = (1-v) * beta_tilde + v * beta
                 l_simple_weight=1.,
                 conditioning_key=None,
                 parameterization="eps",  # all assuming fixed variance schedules
                 scheduler_config=None,
                 use_positional_encodings=False,
                 learn_logvar=False,
                 logvar_init=0.,
                 rescale_betas_zero_snr=False,
                 ):
        super().__init__()
        assert parameterization in ["eps", "x0", "v"], 'currently only supporting "eps" and "x0" and "v"'
        self.parameterization = parameterization
        mainlogger.info(f"{self.__class__.__name__}: Running in {self.parameterization}-prediction mode")
        self.cond_stage_model = None
        self.clip_denoised = clip_denoised
        self.log_every_t = log_every_t
        self.first_stage_key = first_stage_key
        self.channels = channels
        self.temporal_length = unet_config.params.temporal_length
        self.image_size = image_size  # try conv?
        if isinstance(self.image_size, int):
            self.image_size = [self.image_size, self.image_size]
        self.use_positional_encodings = use_positional_encodings
        self.model = DiffusionWrapper(unet_config, conditioning_key)
        # self.p_decoder = PDecoder()
        #count_params(self.model, verbose=True)
        self.use_ema = use_ema
        self.rescale_betas_zero_snr = rescale_betas_zero_snr
        # ipdb.set_trace()
        if self.use_ema:
            self.model_ema = LitEma(self.model)
            mainlogger.info(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")

        self.use_scheduler = scheduler_config is not None
        if self.use_scheduler:
            self.scheduler_config = scheduler_config

        self.v_posterior = v_posterior
        self.original_elbo_weight = original_elbo_weight
        self.l_simple_weight = l_simple_weight

        if monitor is not None:
            self.monitor = monitor
        if ckpt_path is not None:
            # ipdb.set_trace()
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys, only_model=load_only_unet)

        self.register_schedule(given_betas=given_betas, beta_schedule=beta_schedule, timesteps=timesteps,
                               linear_start=linear_start, linear_end=linear_end, cosine_s=cosine_s)

        ## for reschedule
        self.given_betas = given_betas
        self.beta_schedule = beta_schedule
        self.timesteps = timesteps
        self.cosine_s = cosine_s

        self.loss_type = loss_type

        self.learn_logvar = learn_logvar
        self.logvar = torch.full(fill_value=logvar_init, size=(self.num_timesteps,))
        if self.learn_logvar:
            self.logvar = nn.Parameter(self.logvar, requires_grad=True)

    def register_schedule(self, given_betas=None, beta_schedule="linear", timesteps=1000,
                          linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
        if exists(given_betas):
            betas = given_betas
        else:
            betas = make_beta_schedule(beta_schedule, timesteps, linear_start=linear_start, linear_end=linear_end,
                                       cosine_s=cosine_s)
        if self.rescale_betas_zero_snr:
            betas = rescale_zero_terminal_snr(betas)
        
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.linear_start = linear_start
        self.linear_end = linear_end
        assert alphas_cumprod.shape[0] == self.num_timesteps, 'alphas have to be defined for each timestep'

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod)))

        if self.parameterization != 'v':
            self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod)))
            self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod - 1)))
        else:
            self.register_buffer('sqrt_recip_alphas_cumprod', torch.zeros_like(to_torch(alphas_cumprod)))
            self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.zeros_like(to_torch(alphas_cumprod)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (1 - self.v_posterior) * betas * (1. - alphas_cumprod_prev) / (
                    1. - alphas_cumprod) + self.v_posterior * betas
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance', to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))

        if self.parameterization == "eps":
            lvlb_weights = self.betas ** 2 / (
                        2 * self.posterior_variance * to_torch(alphas) * (1 - self.alphas_cumprod))
        elif self.parameterization == "x0":
            lvlb_weights = 0.5 * np.sqrt(torch.Tensor(alphas_cumprod)) / (2. * 1 - torch.Tensor(alphas_cumprod))
        elif self.parameterization == "v":
            lvlb_weights = torch.ones_like(self.betas ** 2 / (
                    2 * self.posterior_variance * to_torch(alphas) * (1 - self.alphas_cumprod)))
        else:
            raise NotImplementedError("mu not supported")
        # TODO how to choose this term
        lvlb_weights[0] = lvlb_weights[1]
        self.register_buffer('lvlb_weights', lvlb_weights, persistent=False)
        assert not torch.isnan(self.lvlb_weights).all()

    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store(self.model.parameters())
            self.model_ema.copy_to(self.model)
            if context is not None:
                mainlogger.info(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.model.parameters())
                if context is not None:
                    mainlogger.info(f"{context}: Restored training weights")

    def init_from_ckpt(self, path, ignore_keys=list(), only_model=False):
        sd = torch.load(path, map_location="cpu")
        if "state_dict" in list(sd.keys()):
            sd = sd["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    mainlogger.info("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        missing, unexpected = self.load_state_dict(sd, strict=False) if not only_model else self.model.load_state_dict(
            sd, strict=False)
        mainlogger.info(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            mainlogger.info(f"Missing Keys: {missing}")
        if len(unexpected) > 0:
            mainlogger.info(f"Unexpected Keys: {unexpected}")

    def q_mean_variance(self, x_start, t):
        """
        Get the distribution q(x_t | x_0).
        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """
        mean = (extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start)
        variance = extract_into_tensor(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract_into_tensor(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def predict_start_from_noise(self, x_t, t, noise):
        return (
                extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_start_from_z_and_v(self, x_t, t, v):
        # self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        # self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))
        return (
                extract_into_tensor(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t -
                extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    def predict_eps_from_z_and_v(self, x_t, t, v):
        return (
                extract_into_tensor(self.sqrt_alphas_cumprod, t, x_t.shape) * v +
                extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * x_t
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
                extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract_into_tensor(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, clip_denoised: bool):
        model_out = self.model(x, t)
        if self.parameterization == "eps":
            x_recon = self.predict_start_from_noise(x, t=t, noise=model_out)
        elif self.parameterization == "x0":
            x_recon = model_out
        if clip_denoised:
            x_recon.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, t, clip_denoised=True, repeat_noise=False):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, t=t, clip_denoised=clip_denoised)
        noise = noise_like(x.shape, device, repeat_noise)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_loop(self, shape, return_intermediates=False):
        device = self.betas.device
        b = shape[0]
        img = torch.randn(shape, device=device)
        intermediates = [img]
        for i in tqdm(reversed(range(0, self.num_timesteps)), desc='Sampling t', total=self.num_timesteps):
            img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long),
                                clip_denoised=self.clip_denoised)
            if i % self.log_every_t == 0 or i == self.num_timesteps - 1:
                intermediates.append(img)
        if return_intermediates:
            return img, intermediates
        return img

    @torch.no_grad()
    def sample(self, batch_size=16, return_intermediates=False):
        image_size = self.image_size
        channels = self.channels
        return self.p_sample_loop((batch_size, channels, image_size, image_size),
                                  return_intermediates=return_intermediates)

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        return (extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise)

    def get_v(self, x, noise, t):
        return (
                extract_into_tensor(self.sqrt_alphas_cumprod, t, x.shape) * noise -
                extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x.shape) * x
        )

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
        if self.parameterization == "eps":
            target = noise
        elif self.parameterization == "x0":
            target = x_start
        elif self.parameterization == "v":
            target = self.get_v(x_start, noise, t)
        else:
            raise NotImplementedError(f"Paramterization {self.parameterization} not yet supported")

        loss = self.get_loss(model_out, target, mean=False).mean(dim=[1, 2, 3])

        log_prefix = 'train' if self.training else 'val'

        loss_dict.update({f'{log_prefix}/loss_simple': loss.mean()})
        loss_simple = loss.mean() * self.l_simple_weight

        loss_vlb = (self.lvlb_weights[t] * loss).mean()
        loss_dict.update({f'{log_prefix}/loss_vlb': loss_vlb})

        loss = loss_simple + self.original_elbo_weight * loss_vlb

        loss_dict.update({f'{log_prefix}/loss': loss})

        return loss, loss_dict

    def forward(self, x, *args, **kwargs):
        # b, c, h, w, device, img_size, = *x.shape, x.device, self.image_size
        # assert h == img_size and w == img_size, f'height and width of image must be {img_size}'
        t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=self.device).long()
        return self.p_losses(x, t, *args, **kwargs)

    def get_input(self, batch, k):
        x = batch[k]
        '''
        if len(x.shape) == 3:
            x = x[..., None]
        x = rearrange(x, 'b h w c -> b c h w')
        '''
        x = x.to(memory_format=torch.contiguous_format).float()
        return x

    def shared_step(self, batch):
        x = self.get_input(batch, self.first_stage_key)
        loss, loss_dict = self(x)
        return loss, loss_dict

    def training_step(self, batch, batch_idx):
        loss, loss_dict = self.shared_step(batch)

        self.log_dict(loss_dict, prog_bar=True,
                      logger=True, on_step=True, on_epoch=True)

        self.log("global_step", self.global_step,
                 prog_bar=True, logger=True, on_step=True, on_epoch=False)

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
        log["inputs"] = x

        # get diffusion row
        diffusion_row = list()
        x_start = x[:n_row]

        for t in range(self.num_timesteps):
            if t % self.log_every_t == 0 or t == self.num_timesteps - 1:
                t = repeat(torch.tensor([t]), '1 -> b', b=n_row)
                t = t.to(self.device).long()
                noise = torch.randn_like(x_start)
                x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
                diffusion_row.append(x_noisy)

        log["diffusion_row"] = self._get_rows_from_list(diffusion_row)

        if sample:
            # get denoise row
            with self.ema_scope("Plotting"):
                samples, denoise_row = self.sample(batch_size=N, return_intermediates=True)

            log["samples"] = samples
            log["denoise_row"] = self._get_rows_from_list(denoise_row)

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
    """main class"""
    def __init__(self,
                 first_stage_config,
                 cond_stage_config,
                 num_timesteps_cond=None,
                 cond_stage_key="caption",
                 cond_stage_trainable=False,
                 cond_stage_forward=None,
                 conditioning_key=None,
                 uncond_prob=0.2,
                 uncond_type="empty_seq",
                 scale_factor=1.0,
                 scale_by_std=False,
                 encoder_type="2d",
                 only_model=False,
                 noise_strength=0,
                 use_dynamic_rescale=False,
                 base_scale=0.7,
                 turning_step=400,
                 interp_mode=False,
                 fps_condition_type='fs',
                 perframe_ae=False,
                 # added
                 logdir=None,
                 rand_cond_frame=False,
                 en_and_decode_n_samples_a_time=None,
                 *args, **kwargs):
        self.num_timesteps_cond = default(num_timesteps_cond, 1)
        self.scale_by_std = scale_by_std
        assert self.num_timesteps_cond <= kwargs['timesteps']
        # for backwards compatibility after implementation of DiffusionWrapper
        ckpt_path = kwargs.pop("ckpt_path", None)
        ignore_keys = kwargs.pop("ignore_keys", [])
        conditioning_key = default(conditioning_key, 'crossattn')
        super().__init__(conditioning_key=conditioning_key, *args, **kwargs)
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
        assert(encoder_type in ["2d", "3d"])
        self.uncond_prob = uncond_prob
        self.classifier_free_guidance = True if uncond_prob > 0 else False
        assert(uncond_type in ["zero_embed", "empty_seq"])
        self.uncond_type = uncond_type

        self.restarted_from_ckpt = False
        # ipdb.set_trace()
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys, only_model=only_model)
            self.restarted_from_ckpt = True

        # self.p_decoder = PDecoder()
                
    def make_cond_schedule(self, ):
        self.cond_ids = torch.full(size=(self.num_timesteps,), fill_value=self.num_timesteps - 1, dtype=torch.long)
        ids = torch.round(torch.linspace(0, self.num_timesteps - 1, self.num_timesteps_cond)).long()
        self.cond_ids[:self.num_timesteps_cond] = ids

    @rank_zero_only
    @torch.no_grad()
    def on_train_batch_start(self, batch, batch_idx, dataloader_idx=None):
        # only for very first batch, reset the self.scale_factor
        if self.scale_by_std and self.current_epoch == 0 and self.global_step == 0 and batch_idx == 0 and \
                not self.restarted_from_ckpt:
            assert self.scale_factor == 1., 'rather not use custom rescaling and std-rescaling simultaneously'
            # set rescale weight to 1./std of encodings
            mainlogger.info("### USING STD-RESCALING ###")
            x = super().get_input(batch, self.first_stage_key)
            x = x.to(self.device)
            encoder_posterior = self.encode_first_stage(x)
            z = self.get_first_stage_encoding(encoder_posterior).detach()
            del self.scale_factor
            self.register_buffer('scale_factor', 1. / z.flatten().std())
            mainlogger.info(f"setting self.scale_factor to {self.scale_factor}")
            mainlogger.info("### USING STD-RESCALING ###")
            mainlogger.info(f"std={z.flatten().std()}")

    def register_schedule(self, given_betas=None, beta_schedule="linear", timesteps=1000,
                          linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
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
        if self.encoder_type == "2d" and x.dim() == 5:
            b, _, t, _, _ = x.shape
            x = rearrange(x, 'b c t h w -> (b t) c h w')
            reshape_back = True
        else:
            reshape_back = False
        
        ## consume more GPU memory but faster
        if not self.perframe_ae:
            encoder_posterior = self.first_stage_model.encode(x)
            results = self.get_first_stage_encoding(encoder_posterior).detach()
        else:  ## consume less GPU memory but slower
            results = []
            for index in range(x.shape[0]):
                frame_batch = self.first_stage_model.encode(x[index:index+1,:,:,:])
                frame_result = self.get_first_stage_encoding(frame_batch).detach()
                results.append(frame_result)
            results = torch.cat(results, dim=0)

        if reshape_back:
            results = rearrange(results, '(b t) c h w -> b c t h w', b=b,t=t)
        
        return results
    
    def decode_core(self, z, **kwargs):
        if self.encoder_type == "2d" and z.dim() == 5:
            b, _, t, _, _ = z.shape
            z = rearrange(z, 'b c t h w -> (b t) c h w')
            reshape_back = True
        else:
            reshape_back = False
            
        if not self.perframe_ae:    
            z = 1. / self.scale_factor * z
            results = self.first_stage_model.decode(z, **kwargs)
        else:
            results = []
            for index in range(z.shape[0]):
                frame_z = 1. / self.scale_factor * z[index:index+1,:,:,:]
                frame_result = self.first_stage_model.decode(frame_z, **kwargs)
                results.append(frame_result)
            results = torch.cat(results, dim=0)

        if reshape_back:
            results = rearrange(results, '(b t) c h w -> b c t h w', b=b,t=t)
        return results

    @torch.no_grad()
    def decode_first_stage(self, z, **kwargs):
        return self.decode_core(z, **kwargs)

    # same as above but without decorator
    def differentiable_decode_first_stage(self, z, **kwargs):
        return self.decode_core(z, **kwargs)
    
    @torch.no_grad()
    def get_batch_input(self, batch, random_uncond, return_first_stage_outputs=False, return_original_cond=False):
        ## video shape: b, c, t, h, w
        x = super().get_input(batch, self.first_stage_key)

        ## encode video frames x to z via a 2D encoder
        z = self.encode_first_stage(x)
                
        ## get caption condition
        cond = batch[self.cond_stage_key]
        if random_uncond and self.uncond_type == 'empty_seq':
            for i, ci in enumerate(cond):
                if random.random() < self.uncond_prob:
                    cond[i] = ""
        if isinstance(cond, dict) or isinstance(cond, list):
            cond_emb = self.get_learned_conditioning(cond)
        else:
            cond_emb = self.get_learned_conditioning(cond.to(self.device))
        if random_uncond and self.uncond_type == 'zero_embed':
            for i, ci in enumerate(cond):
                if random.random() < self.uncond_prob:
                    cond_emb[i] = torch.zeros_like(cond_emb[i])
        
        out = [z, cond_emb]
        ## optional output: self-reconst or caption
        if return_first_stage_outputs:
            xrec = self.decode_first_stage(z)
            out.extend([xrec])

        if return_original_cond:
            out.append(cond)

        return out

    def forward(self, x, c, gt, mask, **kwargs):
        # t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=self.device).long()
        t = torch.randint(1, 2, (x.shape[0],), device=self.device).long()
        if self.use_dynamic_rescale:
            x = x * extract_into_tensor(self.scale_arr, t, x.shape)
        return self.p_losses(x, c, t, gt, mask, **kwargs)

    def shared_step(self, batch, random_uncond, **kwargs):
        x, c = self.get_batch_input(batch, random_uncond=random_uncond)
        loss, loss_dict = self(x, c, **kwargs)
        # loss, loss_dict = self(batch, **kwargs)

        return loss, loss_dict

    # def apply_model(self, x_noisy, t, cond, **kwargs):
    #     if isinstance(cond, dict):
    #         # hybrid case, cond is exptected to be a dict
    #         pass
    #     else:
    #         if not isinstance(cond, list):
    #             cond = [cond]
    #         key = 'c_concat' if self.model.conditioning_key == 'concat' else 'c_crossattn'
    #         cond = {key: cond}

    #     x_recon, feat_list = self.model(x_noisy, t, **cond, **kwargs)
    #     # ipdb.set_trace()

    #     if isinstance(x_recon, tuple):
    #         return x_recon[0], feat_list
    #     else:
    #         return x_recon, feat_list

    def apply_model(self, x_noisy, t, cond, **kwargs):
        # For our custom VJEPA model, we bypass the DiffusionWrapper and call the
        # actual model (VJEPA2_Direct_Wrapper) directly.
        if hasattr(self, 'parameterization') and self.parameterization == "vjepa_direct":
            # The VJEPA wrapper's forward just needs the video and the context dict.
            return self.model.diffusion_model(x_noisy, context=cond, **kwargs)

        if isinstance(cond, dict):
            # hybrid case, cond is exptected to be a dict
            pass
        else:
            if not isinstance(cond, list):
                cond = [cond]
            key = 'c_concat' if self.model.conditioning_key == 'concat' else 'c_crossattn'
            cond = {key: cond}

        x_recon, feat_list = self.model(x_noisy, t, **cond, **kwargs)
        # ipdb.set_trace()

        if isinstance(x_recon, tuple):
            return x_recon[0], feat_list
        else:
            return x_recon, feat_list

    def p_losses(self, x_start, cond, t, gt, mask, noise=None, **kwargs):
        if self.noise_strength > 0:
            b, c, f, _, _ = x_start.shape
            offset_noise = torch.randn(b, c, f, 1, 1, device=x_start.device)
            noise = default(noise, lambda: torch.randn_like(x_start) + self.noise_strength * offset_noise)
        else:
            noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)

        model_output, feat_list = self.apply_model(x_noisy, t, cond, **kwargs)
        # ipdb.set_trace()
        feat_dict = {}
        feat_dict[0] = (feat_list[0] + feat_list[1] + feat_list[2] + feat_list[-1] + feat_list[-2] + feat_list[-3]) / 6 # 72 x 128
        feat_dict[4] = feat_list[-4] # 640
        feat_dict[1] = (feat_list[4] + feat_list[5] + feat_list[-5] + feat_list[-6]) / 4 # 36 x 64
        feat_dict[5] = feat_list[-7] # 1280
        feat_dict[6] = feat_list[3] # 320
        feat_dict[2] = (feat_list[7] + feat_list[8] + feat_list[-8] + feat_list[-9] + feat_list[-10]) / 6 # 18 x 32
        feat_dict[7] = feat_list[6] # 640
        feat_dict[3] = (feat_list[9] + feat_list[10] + feat_list[11] + feat_list[-11] + feat_list[-12] + feat_list[-13]) / 6 # 9 x 16

        # ipdb.set_trace()

        p_output = self.p_decoder(feat_dict, mask)
        l1_loss = nn.L1Loss()
        # ipdb.set_trace()
        loss = l1_loss(p_output[:,0], gt)
        loss_dict = {}
        loss_dict.update({'p_l1_loss': loss})
        

        # loss_dict = {}
        # prefix = 'train' if self.training else 'val'

        # if self.parameterization == "x0":
        #     target = x_start
        # elif self.parameterization == "eps":
        #     target = noise
        # elif self.parameterization == "v":
        #     target = self.get_v(x_start, noise, t)
        # else:
        #     raise NotImplementedError()
        
        # loss_simple = self.get_loss(model_output, target, mean=False).mean([1, 2, 3, 4])
        # loss_dict.update({f'{prefix}/loss_simple': loss_simple.mean()})

        # if self.logvar.device is not self.device:
        #     self.logvar = self.logvar.to(self.device)
        # logvar_t = self.logvar[t]
        # # logvar_t = self.logvar[t.item()].to(self.device) # device conflict when ddp shared
        # loss = loss_simple / torch.exp(logvar_t) + logvar_t
        # # loss = loss_simple / torch.exp(self.logvar) + self.logvar
        # if self.learn_logvar:
        #     loss_dict.update({f'{prefix}/loss_gamma': loss.mean()})
        #     loss_dict.update({'logvar': self.logvar.data.mean()})

        # loss = self.l_simple_weight * loss.mean()

        # loss_vlb = self.get_loss(model_output, target, mean=False).mean(dim=(1, 2, 3, 4))
        # loss_vlb = (self.lvlb_weights[t] * loss_vlb).mean()
        # loss_dict.update({f'{prefix}/loss_vlb': loss_vlb})
        # loss += (self.original_elbo_weight * loss_vlb)
        # loss_dict.update({f'{prefix}/loss': loss})

        return loss, loss_dict  

    def training_step(self, batch, batch_idx):
        loss, loss_dict = self.shared_step(batch, random_uncond=self.classifier_free_guidance)
        ## sync_dist | rank_zero_only 
        self.log_dict(loss_dict, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=False)
        #self.log("epoch/global_step", self.global_step.float(), prog_bar=True, logger=True, on_step=True, on_epoch=False)
        '''
        if self.use_scheduler:
            lr = self.optimizers().param_groups[0]['lr']
            self.log('lr_abs', lr, prog_bar=True, logger=True, on_step=True, on_epoch=False, rank_zero_only=True)
        '''
        if (batch_idx+1) % self.log_every_t == 0:
            mainlogger.info(f"batch:{batch_idx}|epoch:{self.current_epoch} [globalstep:{self.global_step}]: loss={loss}")
        return loss
    
    def _get_denoise_row_from_list(self, samples, desc=''):
        denoise_row = []
        for zd in tqdm(samples, desc=desc):
            denoise_row.append(self.decode_first_stage(zd.to(self.device)))
        n_log_timesteps = len(denoise_row)

        denoise_row = torch.stack(denoise_row)  # n_log_timesteps, b, C, H, W
        
        if denoise_row.dim() == 5:
            denoise_grid = rearrange(denoise_row, 'n b c h w -> b n c h w')
            denoise_grid = rearrange(denoise_grid, 'b n c h w -> (b n) c h w')
            denoise_grid = make_grid(denoise_grid, nrow=n_log_timesteps)
        elif denoise_row.dim() == 6:
            # video, grid_size=[n_log_timesteps*bs, t]
            video_length = denoise_row.shape[3]
            denoise_grid = rearrange(denoise_row, 'n b c t h w -> b n c t h w')
            denoise_grid = rearrange(denoise_grid, 'b n c t h w -> (b n) c t h w')
            denoise_grid = rearrange(denoise_grid, 'n c t h w -> (n t) c h w')
            denoise_grid = make_grid(denoise_grid, nrow=video_length)
        else:
            raise ValueError

        return denoise_grid

    @torch.no_grad()
    def log_images(self, batch, sample=True, ddim_steps=200, ddim_eta=1., plot_denoise_rows=False, \
                    unconditional_guidance_scale=1.0, **kwargs):
        """ log images for LatentDiffusion """
        ##### control sampled imgae for logging, larger value may cause OOM
        sampled_img_num = 2
        for key in batch.keys():
            batch[key] = batch[key][:sampled_img_num]

        ## TBD: currently, classifier_free_guidance sampling is only supported by DDIM
        use_ddim = ddim_steps is not None
        log = dict()
        z, c, xrec, xc = self.get_batch_input(batch, random_uncond=False,
                                                return_first_stage_outputs=True,
                                                return_original_cond=True)

        N = xrec.shape[0]
        log["reconst"] = xrec
        log["condition"] = xc
        

        if sample:
            # get uncond embedding for classifier-free guidance sampling
            if unconditional_guidance_scale != 1.0:
                if isinstance(c, dict):
                    c_cat, c_emb = c["c_concat"][0], c["c_crossattn"][0]
                    log["condition_cat"] = c_cat
                else:
                    c_emb = c

                if self.uncond_type == "empty_seq":
                    prompts = N * [""]
                    uc = self.get_learned_conditioning(prompts)
                elif self.uncond_type == "zero_embed":
                    uc = torch.zeros_like(c_emb)
                ## hybrid case
                if isinstance(c, dict):
                    uc_hybrid = {"c_concat": [c_cat], "c_crossattn": [uc]}
                    uc = uc_hybrid
            else:
                uc = None

            with self.ema_scope("Plotting"):
                samples, z_denoise_row = self.sample_log(cond=c, batch_size=N, ddim=use_ddim,
                                                         ddim_steps=ddim_steps,eta=ddim_eta,
                                                         unconditional_guidance_scale=unconditional_guidance_scale,
                                                         unconditional_conditioning=uc, x0=z, **kwargs)
            x_samples = self.decode_first_stage(samples)
            log["samples"] = x_samples
            
            if plot_denoise_rows:
                denoise_grid = self._get_denoise_row_from_list(z_denoise_row)
                log["denoise_row"] = denoise_grid

        return log

    def p_mean_variance(self, x, c, t, clip_denoised: bool, return_x0=False, score_corrector=None, corrector_kwargs=None, **kwargs):
        t_in = t
        model_out = self.apply_model(x, t_in, c, **kwargs)

        if score_corrector is not None:
            assert self.parameterization == "eps"
            model_out = score_corrector.modify_score(self, model_out, x, t, c, **corrector_kwargs)

        if self.parameterization == "eps":
            x_recon = self.predict_start_from_noise(x, t=t, noise=model_out)
        elif self.parameterization == "x0":
            x_recon = model_out
        else:
            raise NotImplementedError()

        if clip_denoised:
            x_recon.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)

        if return_x0:
            return model_mean, posterior_variance, posterior_log_variance, x_recon
        else:
            return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, c, t, clip_denoised=False, repeat_noise=False, return_x0=False, \
                 temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None, **kwargs):
        b, *_, device = *x.shape, x.device
        outputs = self.p_mean_variance(x=x, c=c, t=t, clip_denoised=clip_denoised, return_x0=return_x0, \
                                       score_corrector=score_corrector, corrector_kwargs=corrector_kwargs, **kwargs)
        if return_x0:
            model_mean, _, model_log_variance, x0 = outputs
        else:
            model_mean, _, model_log_variance = outputs

        noise = noise_like(x.shape, device, repeat_noise) * temperature
        if noise_dropout > 0.:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))

        if return_x0:
            return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise, x0
        else:
            return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_loop(self, cond, shape, return_intermediates=False, x_T=None, verbose=True, callback=None, \
                      timesteps=None, mask=None, x0=None, img_callback=None, start_T=None, log_every_t=None, **kwargs):

        if not log_every_t:
            log_every_t = self.log_every_t
        device = self.betas.device
        b = shape[0]        
        # sample an initial noise
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
            assert x0.shape[2:3] == mask.shape[2:3]  # spatial size has to match

        for i in iterator:
            ts = torch.full((b,), i, device=device, dtype=torch.long)
            if self.shorten_cond_schedule:
                assert self.model.conditioning_key != 'hybrid'
                tc = self.cond_ids[ts].to(cond.device)
                cond = self.q_sample(x_start=cond, t=tc, noise=torch.randn_like(cond))

            img = self.p_sample(img, cond, ts, clip_denoised=self.clip_denoised, **kwargs)
            if mask is not None:
                img_orig = self.q_sample(x0, ts)
                img = img_orig * mask + (1. - mask) * img

            if i % log_every_t == 0 or i == timesteps - 1:
                intermediates.append(img)
            if callback: callback(i)
            if img_callback: img_callback(img, i)

        if return_intermediates:
            return img, intermediates
        return img

    @torch.no_grad()
    def sample(self, cond, batch_size=16, return_intermediates=False, x_T=None, \
               verbose=True, timesteps=None, mask=None, x0=None, shape=None, **kwargs):
        if shape is None:
            shape = (batch_size, self.channels, self.temporal_length, *self.image_size)
        if cond is not None:
            if isinstance(cond, dict):
                cond = {key: cond[key][:batch_size] if not isinstance(cond[key], list) else
                list(map(lambda x: x[:batch_size], cond[key])) for key in cond}
            else:
                cond = [c[:batch_size] for c in cond] if isinstance(cond, list) else cond[:batch_size]
        return self.p_sample_loop(cond,
                                  shape,
                                  return_intermediates=return_intermediates, x_T=x_T,
                                  verbose=verbose, timesteps=timesteps,
                                  mask=mask, x0=x0, **kwargs)

    @torch.no_grad()
    def sample_log(self, cond, batch_size, ddim, ddim_steps, **kwargs):
        if ddim:
            ddim_sampler = DDIMSampler(self)
            shape = (self.channels, self.temporal_length, *self.image_size)
            samples, intermediates = ddim_sampler.sample(ddim_steps, batch_size, shape, cond, verbose=False, **kwargs)

        else:
            samples, intermediates = self.sample(cond=cond, batch_size=batch_size, return_intermediates=True, **kwargs)

        return samples, intermediates

    def configure_schedulers(self, optimizer):
        assert 'target' in self.scheduler_config
        scheduler_name = self.scheduler_config.target.split('.')[-1]
        interval = self.scheduler_config.interval
        frequency = self.scheduler_config.frequency
        if scheduler_name == "LambdaLRScheduler":
            scheduler = instantiate_from_config(self.scheduler_config)
            scheduler.start_step = self.global_step
            lr_scheduler = {
                            'scheduler': LambdaLR(optimizer, lr_lambda=scheduler.schedule),
                            'interval': interval,
                            'frequency': frequency
            }
        elif scheduler_name == "CosineAnnealingLRScheduler":
            scheduler = instantiate_from_config(self.scheduler_config)
            decay_steps = scheduler.decay_steps
            last_step = -1 if self.global_step == 0 else scheduler.start_step
            lr_scheduler = {
                            'scheduler': CosineAnnealingLR(optimizer, T_max=decay_steps, last_epoch=last_step),
                            'interval': interval,
                            'frequency': frequency
            }
        else:
            raise NotImplementedError
        return lr_scheduler

class LatentVisualDiffusion(LatentDiffusion):
    def __init__(self, img_cond_stage_config, image_proj_stage_config, freeze_embedder=True, image_proj_model_trainable=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.image_proj_model_trainable = image_proj_model_trainable
        # ipdb.set_trace()
        self._init_embedder(img_cond_stage_config, freeze_embedder)
        self._init_img_ctx_projector(image_proj_stage_config, image_proj_model_trainable)
        # self._freeze_all()


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
        # self.eval()
        # self.train = disabled_train
        # ipdb.set_trace()
        for name, param in self.named_parameters():
            # print(name)
            if 'p_decoder' not in name:
                # print(name)
                param.requires_grad = False
        # ipdb.set_trace()

    def shared_step(self, batch, random_uncond, **kwargs):
        x, c, fs, gt, mask = self.get_batch_input(batch, random_uncond=random_uncond, return_fs=True)
        kwargs.update({"fs": fs.long()})
        loss, loss_dict = self(x, c, gt, mask, **kwargs)
        return loss, loss_dict
    
    def get_batch_input(self, batch, random_uncond, return_first_stage_outputs=False, return_original_cond=False, return_fs=False, return_cond_frame=False, return_original_input=False, **kwargs):
        ## x: b c t h w
        x = super().get_input(batch, self.first_stage_key)
        # ipdb.set_trace()
        ## encode video frames x to z via a 2D encoder        
        z = self.encode_first_stage(x)
        
        ## get caption condition
        cond_input = batch[self.cond_stage_key]

        if isinstance(cond_input, dict) or isinstance(cond_input, list):
            cond_emb = self.get_learned_conditioning(cond_input)
        else:
            cond_emb = self.get_learned_conditioning(cond_input.to(self.device))
                
        cond = {}
        ## to support classifier-free guidance, randomly drop out only text conditioning 5%, only image conditioning 5%, and both 5%.
        if random_uncond:
            random_num = torch.rand(x.size(0), device=x.device)
        else:
            random_num = torch.ones(x.size(0), device=x.device)  ## by doning so, we can get text embedding and complete img emb for inference
        prompt_mask = rearrange(random_num < 2 * self.uncond_prob, "n -> n 1 1")
        input_mask = 1 - rearrange((random_num >= self.uncond_prob).float() * (random_num < 3 * self.uncond_prob).float(), "n -> n 1 1 1")

        null_prompt = self.get_learned_conditioning([""])
        prompt_imb = torch.where(prompt_mask, null_prompt, cond_emb.detach())

        ## get conditioning frame
        cond_frame_index = 0
        if self.rand_cond_frame:
            cond_frame_index = random.randint(0, self.model.diffusion_model.temporal_length-1)

        img = x[:,:,cond_frame_index,...]
        img = input_mask * img
        ## img: b c h w
        img_emb = self.embedder(img) ## b l c
        img_emb = self.image_proj_model(img_emb)

        if self.model.conditioning_key == 'hybrid':
            if self.interp_mode:
                ## starting frame + (L-2 empty frames) + ending frame
                img_cat_cond = torch.zeros_like(z)
                img_cat_cond[:,:,0,:,:] = z[:,:,0,:,:]
                img_cat_cond[:,:,-1,:,:] = z[:,:,-1,:,:]
            else:
                ## simply repeat the cond_frame to match the seq_len of z
                img_cat_cond = z[:,:,cond_frame_index,:,:]
                img_cat_cond = img_cat_cond.unsqueeze(2)
                img_cat_cond = repeat(img_cat_cond, 'b c t h w -> b c (repeat t) h w', repeat=z.shape[2])

            cond["c_concat"] = [img_cat_cond] # b c t h w
        cond["c_crossattn"] = [torch.cat([prompt_imb, img_emb], dim=1)] ## concat in the seq_len dim

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
            out.append(x[:,:,cond_frame_index,...].unsqueeze(2))
        if return_original_input:
            out.append(x)

        # ipdb.set_trace()
        # gt = [float(p.split('_')[-8]) for p in batch['path']]
        gt = [p for p in batch['gt_ann']]
        # ipdb.set_trace()
        gt = torch.tensor(gt).to(out[0].device)
        out.append(gt)
        # ipdb.set_trace()

        out.append(batch['mask'])

        return out

    @torch.no_grad()
    def log_images(self, batch, sample=True, ddim_steps=50, ddim_eta=1., plot_denoise_rows=False, \
                    unconditional_guidance_scale=1.0, mask=None, **kwargs):
        """ log images for LatentVisualDiffusion """
        ##### sampled_img_num: control sampled imgae for logging, larger value may cause OOM
        sampled_img_num = 1
        for key in batch.keys():
            batch[key] = batch[key][:sampled_img_num]

        ## TBD: currently, classifier_free_guidance sampling is only supported by DDIM
        use_ddim = ddim_steps is not None
        log = dict()

        z, c, xrec, xc, fs, cond_x = self.get_batch_input(batch, random_uncond=False,
                                                return_first_stage_outputs=True,
                                                return_original_cond=True,
                                                return_fs=True,
                                                return_cond_frame=True)

        N = xrec.shape[0]
        log["image_condition"] = cond_x
        log["reconst"] = xrec
        xc_with_fs = []
        for idx, content in enumerate(xc):
            xc_with_fs.append(content + '_fs=' + str(fs[idx].item()))
        log["condition"] = xc_with_fs
        kwargs.update({"fs": fs.long()})

        c_cat = None
        if sample:
            # get uncond embedding for classifier-free guidance sampling
            if unconditional_guidance_scale != 1.0:
                if isinstance(c, dict):
                    c_emb = c["c_crossattn"][0]
                    if 'c_concat' in c.keys():
                        c_cat = c["c_concat"][0]
                else:
                    c_emb = c

                if self.uncond_type == "empty_seq":
                    prompts = N * [""]
                    uc_prompt = self.get_learned_conditioning(prompts)
                elif self.uncond_type == "zero_embed":
                    uc_prompt = torch.zeros_like(c_emb)
                
                img = torch.zeros_like(xrec[:,:,0]) ## b c h w
                ## img: b c h w
                img_emb = self.embedder(img) ## b l c
                uc_img = self.image_proj_model(img_emb)

                uc = torch.cat([uc_prompt, uc_img], dim=1)
                ## hybrid case
                if isinstance(c, dict):
                    uc_hybrid = {"c_concat": [c_cat], "c_crossattn": [uc]}
                    uc = uc_hybrid
            else:
                uc = None

            with self.ema_scope("Plotting"):
                samples, z_denoise_row = self.sample_log(cond=c, batch_size=N, ddim=use_ddim,
                                                         ddim_steps=ddim_steps,eta=ddim_eta,
                                                         unconditional_guidance_scale=unconditional_guidance_scale,
                                                         unconditional_conditioning=uc, x0=z, **kwargs)
            x_samples = self.decode_first_stage(samples)
            log["samples"] = x_samples
            
            if plot_denoise_rows:
                denoise_grid = self._get_denoise_row_from_list(z_denoise_row)
                log["denoise_row"] = denoise_grid

        return log

    def configure_optimizers(self):
        """ configure_optimizers for LatentDiffusion """
        lr = self.learning_rate

        # params = list(self.model.parameters())
        params = [p for p in self.parameters() if p.requires_grad]
        mainlogger.info(f"@Training [{len(params)}] Full Paramters.")

        # if self.cond_stage_trainable:
        #     params_cond_stage = [p for p in self.cond_stage_model.parameters() if p.requires_grad == True]
        #     mainlogger.info(f"@Training [{len(params_cond_stage)}] Paramters for Cond_stage_model.")
        #     params.extend(params_cond_stage)
        
        # if self.image_proj_model_trainable:
        #     mainlogger.info(f"@Training [{len(list(self.image_proj_model.parameters()))}] Paramters for Image_proj_model.")
        #     params.extend(list(self.image_proj_model.parameters()))   

        # if self.learn_logvar:
        #     mainlogger.info('Diffusion model optimizing logvar')
        #     if isinstance(params[0], dict):
        #         params.append({"params": [self.logvar]})
        #     else:
        #         params.append(self.logvar)

        ## optimizer
        # ipdb.set_trace()
        optimizer = torch.optim.AdamW(params, lr=lr)

        ## lr scheduler
        if self.use_scheduler:
            mainlogger.info("Setting up scheduler...")
            lr_scheduler = self.configure_schedulers(optimizer)
            return [optimizer], [lr_scheduler]
        
        return optimizer

class ImprovedScaleAwareL1Loss(nn.Module):
    def __init__(self, direct_weight=1.0, scale_weight=0.5, value_weight=0.5, eps=1e-8):
        super(ImprovedScaleAwareL1Loss, self).__init__()
        self.l1_loss = nn.L1Loss()
        self.direct_weight = direct_weight
        self.scale_weight = scale_weight
        self.value_weight = value_weight
        self.eps = eps

    def forward(self, pred, gt):
        # Ensure tensors are float type
        gt = gt.float()
        pred = pred.float()

        # print('pred: ', pred)
        # print('gt: ', gt)
        
        # Safety clipping to avoid numerical issues
        pred = torch.clamp(pred, min=self.eps)
        gt = torch.clamp(gt, min=self.eps)
        
        # Standard L1 loss component
        direct_loss = self.l1_loss(pred, gt)
        
        # Scale-aware component
        scale_gt = torch.log10(gt)
        scale_pred = torch.log10(pred)
        scale_loss = self.l1_loss(scale_gt, scale_pred)
        
        # Value-within-scale component (normalize within scale)
        norm_gt = gt / (10 ** torch.floor(scale_gt))
        norm_pred = pred / (10 ** torch.floor(scale_pred))
        value_loss = self.l1_loss(norm_gt, norm_pred)
        
        # Combine losses with weighting
        total_loss = self.direct_weight * direct_loss + self.scale_weight * scale_loss + self.value_weight * value_loss
        
        return total_loss

class LatentVisualDiffusionv9(LatentVisualDiffusion):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.p_decoder = PDecoder()
        self._freeze_all()

    def forward(self, x, c, gt, mask, **kwargs):
        # t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=self.device).long()
        t = torch.randint(51, 52, (x.shape[0],), device=self.device).long()
        if self.use_dynamic_rescale:
            x = x * extract_into_tensor(self.scale_arr, t, x.shape)
        return self.p_losses(x, c, t, gt, mask, **kwargs)
    
    def p_losses(self, x_start, cond, t, gt, mask, noise=None, **kwargs):
        # assert self.noise_strength > 0
        if self.noise_strength > 0:
            b, c, f, _, _ = x_start.shape
            offset_noise = torch.randn(b, c, f, 1, 1, device=x_start.device)
            noise = default(noise, lambda: torch.randn_like(x_start) + self.noise_strength * offset_noise)
        else:
            noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)

        model_output, feat_list = self.apply_model(x_noisy, t, cond, **kwargs)
        # ipdb.set_trace()
        feat_dict = {}
        feat_dict[0] = (feat_list[0] + feat_list[1] + feat_list[2] + feat_list[-1] + feat_list[-2] + feat_list[-3]) / 6 # 72 x 128
        feat_dict[4] = feat_list[-4] # 640
        feat_dict[1] = (feat_list[4] + feat_list[5] + feat_list[-5] + feat_list[-6]) / 4 # 36 x 64
        feat_dict[5] = feat_list[-7] # 1280
        feat_dict[6] = feat_list[3] # 320
        feat_dict[2] = (feat_list[7] + feat_list[8] + feat_list[-8] + feat_list[-9] + feat_list[-10]) / 6 # 18 x 32
        feat_dict[7] = feat_list[6] # 640
        feat_dict[3] = (feat_list[9] + feat_list[10] + feat_list[11] + feat_list[-11] + feat_list[-12] + feat_list[-13]) / 6 # 9 x 16

        # ipdb.set_trace()

        p_output = self.p_decoder(feat_dict, mask, full_resolution=0)
        l1_loss = nn.L1Loss()
        # ipdb.set_trace()
        loss = l1_loss(p_output[:,0], gt)

        # Add to p_losses method before returning loss
        if torch.isnan(loss):
            # Use a small constant value to avoid training failure
            loss = torch.tensor(0.1, device=loss.device, requires_grad=True)
            print("NaN loss detected, using fallback value")
            
        loss_dict = {}
        loss_dict.update({'p_l1_loss': loss})
    

        return loss, loss_dict 
    

class LatentVisualDiffusionv41(LatentVisualDiffusionv9):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.p_decoder = PDecoderSupElasticityv1()
        self._freeze_all()

    def shared_step(self, batch, random_uncond, **kwargs):
        x, c, fs, gt, mask, keytime_mask = self.get_batch_input(batch, random_uncond=random_uncond, return_fs=True)
        kwargs.update({"fs": fs.long()})
        loss, loss_dict = self(x, c, gt, mask, keytime_mask, **kwargs)
        return loss, loss_dict

    def forward(self, x, c, gt, mask, keytime_mask, **kwargs):
        # t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=self.device).long()
        t = torch.randint(50, 51, (x.shape[0],), device=self.device).long()
        if self.use_dynamic_rescale:
            x = x * extract_into_tensor(self.scale_arr, t, x.shape)
        return self.p_losses(x, c, t, gt, mask, keytime_mask, **kwargs)

    def p_losses(self, x_start, cond, t, gt, mask, keytime_mask, noise=None, **kwargs):
        # assert self.noise_strength > 0
        if self.noise_strength > 0:
            b, c, f, _, _ = x_start.shape
            offset_noise = torch.randn(b, c, f, 1, 1, device=x_start.device)
            noise = default(noise, lambda: torch.randn_like(x_start) + self.noise_strength * offset_noise)
        else:
            noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)

        model_output, feat_list = self.apply_model(x_noisy, t, cond, **kwargs)
        # ipdb.set_trace()
        feat_dict = {}
        feat_dict[0] = (feat_list[0] + feat_list[1] + feat_list[2] + feat_list[-1] + feat_list[-2] + feat_list[-3]) / 6 # 72 x 128
        feat_dict[4] = feat_list[-4] # 640
        feat_dict[1] = (feat_list[4] + feat_list[5] + feat_list[-5] + feat_list[-6]) / 4 # 36 x 64
        feat_dict[5] = feat_list[-7] # 1280
        feat_dict[6] = feat_list[3] # 320
        feat_dict[2] = (feat_list[7] + feat_list[8] + feat_list[-8] + feat_list[-9] + feat_list[-10]) / 6 # 18 x 32
        feat_dict[7] = feat_list[6] # 640
        feat_dict[3] = (feat_list[9] + feat_list[10] + feat_list[11] + feat_list[-11] + feat_list[-12] + feat_list[-13]) / 6 # 9 x 16

        # ipdb.set_trace()

        temporal_key_points, height_values, height_differences, normed_height, normed_height_differences, normed_sqrt_height_differences, gt_retitusion, key_masks = self.p_decoder(feat_dict, mask, full_resolution=0)
        l1_loss = nn.L1Loss()
        
        # Ensure all GT tensors are on the same device as prediction tensors
        device = temporal_key_points.device
        for key in gt:
            if isinstance(gt[key], torch.Tensor):
                gt[key] = gt[key].to(device)
            elif isinstance(gt[key], list):
                gt[key] = [item.to(device) if isinstance(item, torch.Tensor) else item for item in gt[key]]
                
        # ipdb.set_trace()
        norm_abs_loss_coef = 0.01
        loss_temporal_key_points = norm_abs_loss_coef * (l1_loss(temporal_key_points[:,0], gt['temporal_key_points'][0]) + l1_loss(temporal_key_points[:,1], gt['temporal_key_points'][1]) + l1_loss(temporal_key_points[:,2], gt['temporal_key_points'][2]))
        loss_height_values = norm_abs_loss_coef * (l1_loss(height_values[:,0], gt['height_values_of_them'][0]) + l1_loss(height_values[:,1], gt['height_values_of_them'][1]) + l1_loss(height_values[:,2], gt['height_values_of_them'][2]))
        loss_height_differences = norm_abs_loss_coef * (l1_loss(height_differences[:,0], gt['height_differences'][0]) + l1_loss(height_differences[:,1], gt['height_differences'][1]))
        loss_normed_height = l1_loss(normed_height[:,0], gt['normed_height'][0]) + l1_loss(normed_height[:,1], gt['normed_height'][1]) + l1_loss(normed_height[:,2], gt['normed_height'][2])
        loss_normed_height_differences = l1_loss(normed_height_differences[:,0], gt['normed_height_differences'][0]) + l1_loss(normed_height_differences[:,1], gt['normed_height_differences'][1])
        loss_normed_sqrt_height_differences = l1_loss(normed_sqrt_height_differences[:,0], gt['normed_sqrt_height_differences'][0]) + l1_loss(normed_sqrt_height_differences[:,1], gt['normed_sqrt_height_differences'][1])
        loss_gt_retitusion = l1_loss(gt_retitusion[:,0], gt['gt_restitution'])

        print(loss_gt_retitusion)
        
        # Binary Cross Entropy loss for key_masks supervision with keytime_mask
        # bce_loss = nn.BCELoss()
        # Interpolate keytime_mask to match the resolution of key_masks if needed
        keytime_mask = keytime_mask.squeeze(1).float()
        if key_masks.shape != keytime_mask.shape:
            # ipdb.set_trace()
            keytime_mask_interpolated = F.interpolate(keytime_mask, size=key_masks.shape[2:], mode='nearest')
            # loss_key_masks = bce_loss(key_masks, keytime_mask_interpolated)
            loss_key_masks = F.binary_cross_entropy_with_logits(key_masks, keytime_mask_interpolated)
        else:
            # loss_key_masks = bce_loss(key_masks, keytime_mask)
            loss_key_masks = F.binary_cross_entropy_with_logits(key_masks, keytime_mask_interpolated)

        extra_loss_coef = 0.1
        
        # Combine all losses
        loss = extra_loss_coef * (loss_temporal_key_points + loss_height_values + loss_height_differences + loss_normed_height + loss_normed_height_differences + loss_normed_sqrt_height_differences + loss_key_masks) + loss_gt_retitusion
        
        loss_dict = {}
        loss_dict.update({
            'total_loss': loss,
            'loss_temporal_key_points': loss_temporal_key_points,
            'loss_height_values': loss_height_values,
            'loss_height_differences': loss_height_differences,
            'loss_normed_height': loss_normed_height,
            'loss_normed_height_differences': loss_normed_height_differences,
            'loss_normed_sqrt_height_differences': loss_normed_sqrt_height_differences,
            'loss_key_masks': loss_key_masks,
            'loss_gt_retitusion': loss_gt_retitusion,
        })

        # Add to p_losses method before returning loss
        if torch.isnan(loss):
            # Use a small constant value to avoid training failure
            loss = torch.tensor(0.1, device=loss.device, requires_grad=True)
            print("NaN loss detected, using fallback value")

        for key_i in loss_dict.keys():
            if torch.isnan(loss_dict[key_i]):
                loss_dict[key_i] = torch.tensor(0.001, device=loss.device, requires_grad=True)
                print("NaN loss detected, using fallback value")
    

        return loss, loss_dict 


    def get_batch_input(self, batch, random_uncond, return_first_stage_outputs=False, return_original_cond=False, return_fs=False, return_cond_frame=False, return_original_input=False, **kwargs):
        ## x: b c t h w
        x = super().get_input(batch, self.first_stage_key)
        # ipdb.set_trace()
        ## encode video frames x to z via a 2D encoder        
        z = self.encode_first_stage(x)
        
        ## get caption condition
        cond_input = batch[self.cond_stage_key]

        if isinstance(cond_input, dict) or isinstance(cond_input, list):
            cond_emb = self.get_learned_conditioning(cond_input)
        else:
            cond_emb = self.get_learned_conditioning(cond_input.to(self.device))
                
        cond = {}
        ## to support classifier-free guidance, randomly drop out only text conditioning 5%, only image conditioning 5%, and both 5%.
        if random_uncond:
            random_num = torch.rand(x.size(0), device=x.device)
        else:
            random_num = torch.ones(x.size(0), device=x.device)  ## by doning so, we can get text embedding and complete img emb for inference
        prompt_mask = rearrange(random_num < 2 * self.uncond_prob, "n -> n 1 1")
        input_mask = 1 - rearrange((random_num >= self.uncond_prob).float() * (random_num < 3 * self.uncond_prob).float(), "n -> n 1 1 1")

        null_prompt = self.get_learned_conditioning([""])
        prompt_imb = torch.where(prompt_mask, null_prompt, cond_emb.detach())

        ## get conditioning frame
        cond_frame_index = 0
        if self.rand_cond_frame:
            cond_frame_index = random.randint(0, self.model.diffusion_model.temporal_length-1)

        img = x[:,:,cond_frame_index,...]
        img = input_mask * img
        ## img: b c h w
        img_emb = self.embedder(img) ## b l c
        img_emb = self.image_proj_model(img_emb)

        if self.model.conditioning_key == 'hybrid':
            if self.interp_mode:
                ## starting frame + (L-2 empty frames) + ending frame
                img_cat_cond = torch.zeros_like(z)
                img_cat_cond[:,:,0,:,:] = z[:,:,0,:,:]
                img_cat_cond[:,:,-1,:,:] = z[:,:,-1,:,:]
            else:
                ## simply repeat the cond_frame to match the seq_len of z
                img_cat_cond = z[:,:,cond_frame_index,:,:]
                img_cat_cond = img_cat_cond.unsqueeze(2)
                img_cat_cond = repeat(img_cat_cond, 'b c t h w -> b c (repeat t) h w', repeat=z.shape[2])

            cond["c_concat"] = [img_cat_cond] # b c t h w
        cond["c_crossattn"] = [torch.cat([prompt_imb, img_emb], dim=1)] ## concat in the seq_len dim

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
            out.append(x[:,:,cond_frame_index,...].unsqueeze(2))
        if return_original_input:
            out.append(x)

        # ipdb.set_trace()
        # gt = [float(p.split('_')[-8]) for p in batch['path']]
        # gt = [p for p in batch['gt_ann']]
        gt = batch['gt_ann']
        # ipdb.set_trace()
        # gt = torch.tensor(gt).to(out[0].device)
        out.append(gt)
        # ipdb.set_trace()

        out.append(batch['mask'])
        out.append(batch['keytime_mask'])

        return out


class LatentVisualDiffusionv43(LatentVisualDiffusionv41):
    def __init__(self, *args, ft_weights_list=None, ft_lr=1e-5, **kwargs):
        super().__init__(*args, **kwargs)
        self.p_decoder = PDecoderSupElasticityv1()
        self.ft_weights_list = ft_weights_list
        self.ft_lr = ft_lr
        self._freeze_all_but_some(self.ft_weights_list)

    def _freeze_all_but_some(self, ft_weights_list):
        for name, param in self.named_parameters():
            if 'p_decoder' not in name:
                param.requires_grad = False

        # Then, unfreeze based on patterns in ft_weights_list
        for ft_weight in ft_weights_list:
            # Check if this is a regex pattern (contains wildcard * or other regex characters)
            if any(c in ft_weight for c in ".*+?^$[](){}|\\"):
                # It's a regex pattern, compile it
                pattern = re.compile(ft_weight.replace(".", "\\.").replace("*", ".*"))
                
                # Find matching parameters and unfreeze them
                for name, param in self.named_parameters():
                    if pattern.search(name):
                        param.requires_grad = True
            else:
                # Original direct matching code
                for name, param in self.named_parameters():
                    if ft_weight in name:
                        param.requires_grad = True

    def configure_optimizers(self):
        """ configure_optimizers for LatentDiffusion """
        lr = self.learning_rate

        params_1 = [p for n, p in self.named_parameters() if p.requires_grad and 'p_decoder' in n]
        params_2 = [p for n, p in self.named_parameters() if p.requires_grad and 'p_decoder' not in n]
        mainlogger.info(f"@Training [{len(params_1)}] Full Paramters.")
        mainlogger.info(f"@Fine-tuning [{len(params_2)}] Full Paramters.")

        
        optimizer = torch.optim.AdamW([
            {"params": params_1, "lr": lr},
            {"params": params_2, "lr": self.ft_lr}
        ])

        # ipdb.set_trace()

        ## lr scheduler
        if self.use_scheduler:
            ipdb.set_trace()
            mainlogger.info("Setting up scheduler...")
            lr_scheduler = self.configure_schedulers(optimizer)
            return [optimizer], [lr_scheduler]
        
        return optimizer


class LatentVisualDiffusionv47(LatentVisualDiffusionv43):
    def __init__(self, *args, extra_loss_coef=0.1, norm_abs_loss_coef=0.01, pd_config=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.extra_loss_coef = extra_loss_coef
        self.norm_abs_loss_coef = norm_abs_loss_coef
        self.pd_config = pd_config
        self.p_decoder = PDecoderSupElasticityv2(
            self.pd_config.temporal_key_points,
            self.pd_config.height_values,
            self.pd_config.height_differences,
            self.pd_config.normed_height,
            self.pd_config.normed_height_differences,
            self.pd_config.normed_sqrt_height_differences,
            self.pd_config.keytime_masks,
            getattr(self.pd_config, 'full_resolution', 0),
            )
        self._freeze_all_but_some(self.ft_weights_list)

    def p_losses(self, x_start, cond, t, gt, mask, keytime_mask, noise=None, **kwargs):
        # assert self.noise_strength > 0
        if self.noise_strength > 0:
            b, c, f, _, _ = x_start.shape
            offset_noise = torch.randn(b, c, f, 1, 1, device=x_start.device)
            noise = default(noise, lambda: torch.randn_like(x_start) + self.noise_strength * offset_noise)
        else:
            noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)

        model_output, feat_list = self.apply_model(x_noisy, t, cond, **kwargs)
        # ipdb.set_trace()
        feat_dict = {}
        feat_dict[0] = (feat_list[0] + feat_list[1] + feat_list[2] + feat_list[-1] + feat_list[-2] + feat_list[-3]) / 6 # 72 x 128
        feat_dict[4] = feat_list[-4] # 640
        feat_dict[1] = (feat_list[4] + feat_list[5] + feat_list[-5] + feat_list[-6]) / 4 # 36 x 64
        feat_dict[5] = feat_list[-7] # 1280
        feat_dict[6] = feat_list[3] # 320
        feat_dict[2] = (feat_list[7] + feat_list[8] + feat_list[-8] + feat_list[-9] + feat_list[-10]) / 6 # 18 x 32
        feat_dict[7] = feat_list[6] # 640
        feat_dict[3] = (feat_list[9] + feat_list[10] + feat_list[11] + feat_list[-11] + feat_list[-12] + feat_list[-13]) / 6 # 9 x 16

        # ipdb.set_trace()

        temporal_key_points, height_values, height_differences, normed_height, normed_height_differences, normed_sqrt_height_differences, gt_retitusion, key_masks = self.p_decoder(feat_dict, mask)
        l1_loss = nn.L1Loss()
        
        # Ensure all GT tensors are on the same device as prediction tensors
        device = feat_dict[0].device
        for key in gt:
            if isinstance(gt[key], torch.Tensor):
                gt[key] = gt[key].to(device)
            elif isinstance(gt[key], list):
                gt[key] = [item.to(device) if isinstance(item, torch.Tensor) else item for item in gt[key]]
                
        # ipdb.set_trace()
        # norm_abs_loss_coef = 0.01
        loss_temporal_key_points = 0 if temporal_key_points is None else self.norm_abs_loss_coef * (l1_loss(temporal_key_points[:,0], gt['temporal_key_points'][0]) + l1_loss(temporal_key_points[:,1], gt['temporal_key_points'][1]) + l1_loss(temporal_key_points[:,2], gt['temporal_key_points'][2]))
        loss_height_values = 0 if height_values is None else self.norm_abs_loss_coef * (l1_loss(height_values[:,0], gt['height_values_of_them'][0]) + l1_loss(height_values[:,1], gt['height_values_of_them'][1]) + l1_loss(height_values[:,2], gt['height_values_of_them'][2]))
        loss_height_differences = 0 if height_differences is None else self.norm_abs_loss_coef * (l1_loss(height_differences[:,0], gt['height_differences'][0]) + l1_loss(height_differences[:,1], gt['height_differences'][1]))
        loss_normed_height = 0 if normed_height is None else l1_loss(normed_height[:,0], gt['normed_height'][0]) + l1_loss(normed_height[:,1], gt['normed_height'][1]) + l1_loss(normed_height[:,2], gt['normed_height'][2])
        loss_normed_height_differences = 0 if normed_height_differences is None else l1_loss(normed_height_differences[:,0], gt['normed_height_differences'][0]) + l1_loss(normed_height_differences[:,1], gt['normed_height_differences'][1])
        loss_normed_sqrt_height_differences = 0 if normed_sqrt_height_differences is None else l1_loss(normed_sqrt_height_differences[:,0], gt['normed_sqrt_height_differences'][0]) + l1_loss(normed_sqrt_height_differences[:,1], gt['normed_sqrt_height_differences'][1])
        loss_gt_retitusion = 0 if gt_retitusion is None else l1_loss(gt_retitusion[:,0], gt['gt_restitution'])

        # print('1 ', gt_retitusion)
        # print('2 ', gt['gt_restitution'])

        # print('pred: ', gt_retitusion[:,0])
        # print('target: ', gt['gt_restitution'])
        
            
        
        # Binary Cross Entropy loss for key_masks supervision with keytime_mask
        # bce_loss = nn.BCELoss()
        # Interpolate keytime_mask to match the resolution of key_masks if needed
        if key_masks is None:
            loss_key_masks = 0
        else:
            keytime_mask = keytime_mask.squeeze(1).float()
            if key_masks.shape != keytime_mask.shape:
                # ipdb.set_trace()
                keytime_mask_interpolated = F.interpolate(keytime_mask, size=key_masks.shape[2:], mode='nearest')
                # loss_key_masks = bce_loss(key_masks, keytime_mask_interpolated)
                loss_key_masks = F.binary_cross_entropy_with_logits(key_masks, keytime_mask_interpolated)
            else:
                # loss_key_masks = bce_loss(key_masks, keytime_mask)
                loss_key_masks = F.binary_cross_entropy_with_logits(key_masks, keytime_mask_interpolated)

        # extra_loss_coef = 0.1

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


        
        # Combine all losses
        loss = self.extra_loss_coef * (loss_temporal_key_points + loss_height_values + loss_height_differences + loss_normed_height + loss_normed_height_differences + loss_normed_sqrt_height_differences + loss_key_masks) + loss_gt_retitusion
        
        loss_dict = {}
        loss_dict.update({
            'total_loss': loss,
            'loss_temporal_key_points': loss_temporal_key_points,
            'loss_height_values': loss_height_values,
            'loss_height_differences': loss_height_differences,
            'loss_normed_height': loss_normed_height,
            'loss_normed_height_differences': loss_normed_height_differences,
            'loss_normed_sqrt_height_differences': loss_normed_sqrt_height_differences,
            'loss_key_masks': loss_key_masks,
            'loss_gt_retitusion': loss_gt_retitusion,
        })

        # # Add to p_losses method before returning loss
        # if torch.isnan(loss):
        #     # Use a small constant value to avoid training failure
        #     loss = torch.tensor(0.1, device=loss.device, requires_grad=True)
        #     print("NaN loss detected, using fallback value")

        for key_i in loss_dict.keys():
            if isinstance(loss_dict[key_i], torch.Tensor) and torch.isnan(loss_dict[key_i]):
                loss_dict[key_i] = torch.tensor(0.001, device=device, requires_grad=True)
                print("NaN loss detected, using fallback value")
    

        return loss, loss_dict 


class LatentVisualDiffusionv84(LatentVisualDiffusionv47):
    def __init__(self, *args, prob3_dim=0, major_specify=0, cond_specify=0, collect_map_together=0, concat_2_feat=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.concat_2_feat = concat_2_feat
        self.collect_map_together = collect_map_together
        self.major_specify = major_specify # [0, 1 mask, 2 red circle, 3 red circle only]
        self.cond_specify = cond_specify # [0, 1, 2, 3 red circle only]
        self.prob3_dim = prob3_dim
        self.prob3_learnable_vector = nn.Parameter(torch.randn((1, self.prob3_dim)) * 0.02)
        self.p_decoder = None
        
        # Initialize MLPs for cross-attention based on typical feature dimensions
        # From the output, we can see the channel dimensions: 320, 320, 320, 640, 640, 640, 1280, 1280, 1280, etc.
        feat_dims = [320, 320, 320, 320, 640, 640, 640, 1280, 1280, 1280, 1280, 1280, 1280, 1280, 1280, 1280, 1280, 1280, 1280, 640, 640, 640, 320, 320, 320]
        
        self.prob3_feat_mlps = nn.ModuleList([
            nn.Linear(self.prob3_dim, feat_dim) for feat_dim in feat_dims
        ])
        self.prob3_feat_output_mlps = nn.ModuleList([
            nn.Linear(feat_dim, self.prob3_dim) for feat_dim in feat_dims
        ])
        self.prob3_output_projection = nn.Linear(self.prob3_dim, 1)
        self.prob3_learnable_prompt = nn.Parameter(
            torch.randn((1, 77, 1024)) * 0.02  # Small standard deviation for more stable initial training
        )
        self._freeze_all_but_some(self.ft_weights_list)

    def _freeze_all_but_some(self, ft_weights_list):
        for name, param in self.named_parameters():
            if 'prob3' not in name:
                param.requires_grad = False
            if 'prob3' in name:
                param.requires_grad = True

        # Then, unfreeze based on patterns in ft_weights_list
        for ft_weight in ft_weights_list:
            # Check if this is a regex pattern (contains wildcard * or other regex characters)
            if any(c in ft_weight for c in ".*+?^$[](){}|\\"):
                # It's a regex pattern, compile it
                pattern = re.compile(ft_weight.replace(".", "\\.").replace("*", ".*"))
                
                # Find matching parameters and unfreeze them
                for name, param in self.named_parameters():
                    if pattern.search(name):
                        param.requires_grad = True
            else:
                # Original direct matching code
                for name, param in self.named_parameters():
                    if ft_weight in name:
                        param.requires_grad = True

    def configure_optimizers(self):
        """ configure_optimizers for LatentDiffusion """
        lr = self.learning_rate

        params_1 = [p for n, p in self.named_parameters() if p.requires_grad and ('prob3' in n or 'p_decoder' in n)]
        for n, p in self.named_parameters():
            if p.requires_grad:
                print(n)
        # ipdb.set_trace()
        # params_2 = [p for n, p in self.named_parameters() if p.requires_grad and 'p_decoder' not in n]
        mainlogger.info(f"@Training [{len(params_1)}] Full Paramters.")
        # mainlogger.info(f"@Fine-tuning [{len(params_2)}] Full Paramters.")

        
        optimizer = torch.optim.AdamW([
            {"params": params_1, "lr": lr},
            # {"params": params_2, "lr": self.ft_lr}
        ])

        # ipdb.set_trace()

        ## lr scheduler
        if self.use_scheduler:
            ipdb.set_trace()
            mainlogger.info("Setting up scheduler...")
            lr_scheduler = self.configure_schedulers(optimizer)
            return [optimizer], [lr_scheduler]
        
        return optimizer

    def p_losses(self, x_start, cond, t, gt, mask, keytime_mask, noise=None, **kwargs):
        # assert self.noise_strength > 0
        if self.noise_strength > 0:
            b, c, f, _, _ = x_start.shape
            offset_noise = torch.randn(b, c, f, 1, 1, device=x_start.device)
            noise = default(noise, lambda: torch.randn_like(x_start) + self.noise_strength * offset_noise)
        else:
            noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)

        cond['c_crossattn'][0][:,:77,:] = self.prob3_learnable_prompt
        model_output, feat_list = self.apply_model(x_noisy, t, cond, **kwargs)
        # for feat_i in range(len(feat_list)):
        #     print(feat_i)
        #     print(feat_list[feat_i].shape)
        # ipdb.set_trace()
        
        # Process each feature through cross-attention
        prob3_outputs = []

        for feat_i, feat in enumerate(feat_list):
            # feat shape: B x T x C x H x W
            B, T, C, H, W = feat.shape
            
            # 1. Map the learnable vector to the corresponding C dimension
            # Project learnable vector to feature dimension
            query_vector = self.prob3_feat_mlps[feat_i](self.prob3_learnable_vector)  # 1 x C
            query_vector = query_vector.expand(B, -1).unsqueeze(1)  # B x 1 x C
            
            # 2. Reshape feature for cross-attention (tokens = T x H x W)
            feat_reshaped = feat.permute(0, 1, 3, 4, 2).contiguous().view(B, T * H * W, C)  # B x (T*H*W) x C
            
            # 3. Cross-attention: Q=query_vector, K=V=feat_tokens
            # Using scaled dot-product attention
            scale = (C ** -0.5)
            
            # Compute attention scores
            attn_scores = torch.matmul(query_vector, feat_reshaped.transpose(-2, -1)) * scale  # B x 1 x (T*H*W)
            attn_weights = torch.softmax(attn_scores, dim=-1)
            
            # Apply attention to values
            attended_feat = torch.matmul(attn_weights, feat_reshaped)  # B x 1 x C
            attended_feat = attended_feat.squeeze(1)  # B x C
            
            # Map back to prob3_dim
            prob3_feat_output = self.prob3_feat_output_mlps[feat_i](attended_feat)  # B x prob3_dim
            prob3_outputs.append(prob3_feat_output)

        # 4. Average pooling over all different layers and final MLP
        if prob3_outputs:
            # Stack and average across features
            prob3_combined = torch.stack(prob3_outputs, dim=1)  # B x num_feats x prob3_dim
            prob3_averaged = torch.mean(prob3_combined, dim=1)  # B x prob3_dim
            
            # Final MLP to map to scalar
            gt_retitusion = self.prob3_output_projection(prob3_averaged)  # B x 1
        else:
            gt_retitusion = None
        # TODO: for each feat in feat_list, shape is B x T x C x H x W
        # 1. have an MLP to map the self.prob3_learnable_vector into the corresponding C dim
        # 2. use the query vector as Q, the T x H x W tokens in the feat as K and V, calculate a cross-attention
        # 3. map the output back to the self.prob3_dim, and avg pooling over all different layers
        # 4. use a simple MLP to map from B x self.prob3_dim to B x 1, and calculate the L1 loss

        
        l1_loss = nn.L1Loss()
        
        # Ensure all GT tensors are on the same device as prediction tensors
        device = feat_list[0].device
        for key in gt:
            if isinstance(gt[key], torch.Tensor):
                gt[key] = gt[key].to(device)
            elif isinstance(gt[key], list):
                gt[key] = [item.to(device) if isinstance(item, torch.Tensor) else item for item in gt[key]]
                
        loss_gt_retitusion = 0 if gt_retitusion is None else l1_loss(gt_retitusion[:,0], gt['gt_restitution'])


        if gt_retitusion is not None:
            print(loss_gt_retitusion)
        


        
        # Combine all losses
        loss = loss_gt_retitusion
        
        loss_dict = {}
        loss_dict.update({
            'total_loss': loss,
            'loss_gt_retitusion': loss_gt_retitusion,
        })

        # # Add to p_losses method before returning loss
        # if torch.isnan(loss):
        #     # Use a small constant value to avoid training failure
        #     loss = torch.tensor(0.1, device=loss.device, requires_grad=True)
        #     print("NaN loss detected, using fallback value")

        for key_i in loss_dict.keys():
            if isinstance(loss_dict[key_i], torch.Tensor) and torch.isnan(loss_dict[key_i]):
                loss_dict[key_i] = torch.tensor(0.001, device=device, requires_grad=True)
                print("NaN loss detected, using fallback value")
    

        return loss, loss_dict 

    def get_batch_input(self, batch, random_uncond, return_first_stage_outputs=False, return_original_cond=False, return_fs=False, return_cond_frame=False, return_original_input=False, **kwargs):
        ## x: b c t h w
        x = super().get_input(batch, self.first_stage_key) # b 2 c t h w
        # x = x.reshape(x.shape[0] * x.shape[1], x.shape[2], x.shape[3], x.shape[4], x.shape[5]) # b2 c t h w
        # ipdb.set_trace()
        ## encode video frames x to z via a 2D encoder 
        if self.major_specify == 0:       
            z = self.encode_first_stage(x)
        elif self.major_specify == 1:
            cur_batch_mask = batch['mask']#.reshape(batch['mask'].shape[0] * batch['mask'].shape[1], batch['mask'].shape[2], batch['mask'].shape[3], batch['mask'].shape[4], batch['mask'].shape[5])
            z = self.encode_first_stage(x * cur_batch_mask)
        elif self.major_specify == 2:
            x_red = batch['mask']
            # x_red = x_red.reshape(x_red.shape[0] * x_red.shape[1], x_red.shape[2], x_red.shape[3], x_red.shape[4], x_red.shape[5]) # b2 c t h w
            z = self.encode_first_stage(x_red)
        elif self.major_specify == 3:
            x_red = batch['mask']
            # x_red = x_red.reshape(x_red.shape[0] * x_red.shape[1], x_red.shape[2], x_red.shape[3], x_red.shape[4], x_red.shape[5]) # b2 c t h w
            x_concat_new = torch.cat([x[:,:,:-1,:,:], x_red[:,:,-1:,:,:]], dim=2) # last frame temporally is x_red
            z = self.encode_first_stage(x_concat_new)

        
        ## get caption condition
        cond_input = batch[self.cond_stage_key]
        # ipdb.set_trace()
        cond_input = cond_input # add for the relative formulation, repeat the '' captions

        if isinstance(cond_input, dict) or isinstance(cond_input, list):
            cond_emb = self.get_learned_conditioning(cond_input)
        else:
            cond_emb = self.get_learned_conditioning(cond_input.to(self.device))
                
        cond = {}
        ## to support classifier-free guidance, randomly drop out only text conditioning 5%, only image conditioning 5%, and both 5%.
        if random_uncond:
            random_num = torch.rand(x.size(0), device=x.device)
        else:
            random_num = torch.ones(x.size(0), device=x.device)  ## by doning so, we can get text embedding and complete img emb for inference
        prompt_mask = rearrange(random_num < 2 * self.uncond_prob, "n -> n 1 1")
        input_mask = 1 - rearrange((random_num >= self.uncond_prob).float() * (random_num < 3 * self.uncond_prob).float(), "n -> n 1 1 1")

        null_prompt = self.get_learned_conditioning([""])
        prompt_imb = torch.where(prompt_mask, null_prompt, cond_emb.detach())

        ## get conditioning frame
        cond_frame_index = 0
        if self.rand_cond_frame:
            cond_frame_index = random.randint(0, self.model.diffusion_model.temporal_length-1)
        if self.cond_specify == 3:
            cond_frame_index = -1

        
        if self.cond_specify == 0:
            img = x[:,:,cond_frame_index,...] # b c h w
        elif self.cond_specify == 1:
            img = x[:,:,cond_frame_index,...] # b c h w
            # ipdb.set_trace()
            cur_batch_mask = batch['mask']#.reshape(batch['mask'].shape[0] * batch['mask'].shape[1], batch['mask'].shape[2], batch['mask'].shape[3], batch['mask'].shape[4], batch['mask'].shape[5])
            img = img * cur_batch_mask[:,:,cond_frame_index,...]
        elif self.cond_specify == 2:
            cur_batch_mask = batch['mask']#.reshape(batch['mask'].shape[0] * batch['mask'].shape[1], batch['mask'].shape[2], batch['mask'].shape[3], batch['mask'].shape[4], batch['mask'].shape[5])
            img = cur_batch_mask[:,:,cond_frame_index,...]
        elif self.cond_specify == 3:
            cur_batch_mask = batch['mask']#.reshape(batch['mask'].shape[0] * batch['mask'].shape[1], batch['mask'].shape[2], batch['mask'].shape[3], batch['mask'].shape[4], batch['mask'].shape[5])
            img = cur_batch_mask[:,:,cond_frame_index,...]
        img = input_mask * img
        ## img: b c h w
        img_emb = self.embedder(img) ## b l c
        img_emb = self.image_proj_model(img_emb)

        if self.model.conditioning_key == 'hybrid':
            if self.interp_mode:
                # ipdb.set_trace()
                ## starting frame + (L-2 empty frames) + ending frame
                img_cat_cond = torch.zeros_like(z)
                img_cat_cond[:,:,0,:,:] = z[:,:,0,:,:]
                img_cat_cond[:,:,-1,:,:] = z[:,:,-1,:,:]
            else:
                ## simply repeat the cond_frame to match the seq_len of z
                if self.cond_specify == 0 or self.cond_specify == 3:
                    img_cat_cond = z[:,:,cond_frame_index,:,:]
                    img_cat_cond = img_cat_cond.unsqueeze(2)
                    img_cat_cond = repeat(img_cat_cond, 'b c t h w -> b c (repeat t) h w', repeat=z.shape[2])
                elif self.cond_specify == 1:
                    cur_batch_mask = batch['mask']#.reshape(batch['mask'].shape[0] * batch['mask'].shape[1], batch['mask'].shape[2], batch['mask'].shape[3], batch['mask'].shape[4], batch['mask'].shape[5])
                    img_cat_cond = self.encode_first_stage(x * cur_batch_mask)
                elif self.cond_specify == 2:
                    cur_batch_mask = batch['mask']#.reshape(batch['mask'].shape[0] * batch['mask'].shape[1], batch['mask'].shape[2], batch['mask'].shape[3], batch['mask'].shape[4], batch['mask'].shape[5])
                    img_cat_cond = self.encode_first_stage(cur_batch_mask)

            cond["c_concat"] = [img_cat_cond] # b c t h w
        cond["c_crossattn"] = [torch.cat([prompt_imb, img_emb], dim=1)] ## concat in the seq_len dim

        out = [z, cond]
        if return_first_stage_outputs:
            xrec = self.decode_first_stage(z)
            out.extend([xrec])

        if return_original_cond:
            out.append(cond_input)
        # ipdb.set_trace()
        if return_fs:
            if self.fps_condition_type == 'fs':
                fs = super().get_input(batch, 'frame_stride')
            elif self.fps_condition_type == 'fps':
                fs = super().get_input(batch, 'fps')
            # ipdb.set_trace()
            out.append(fs)
        if return_cond_frame:
            out.append(x[:,:,cond_frame_index,...].unsqueeze(2))
        if return_original_input:
            out.append(x)

        # ipdb.set_trace()
        # gt = [float(p.split('_')[-8]) for p in batch['path']]
        # gt = [p for p in batch['gt_ann']]
        gt = batch['gt_ann']
        # ipdb.set_trace()
        # gt = torch.tensor(gt).to(out[0].device)
        out.append(gt)
        # ipdb.set_trace()

        out.append(batch['mask'])
        out.append(batch['keytime_mask'])

        return out

    def calculate_physics_loss(self, feat_list, gt):
        """
        Calculates the physics-based loss from a list of feature maps.
        This logic is extracted from the original p_losses for reusability.
        """
        device = feat_list[0].device
        
        # Process each feature through cross-attention
        prob3_outputs = []
        for feat_i, feat in enumerate(feat_list):
            B, T, C, H, W = feat.shape
            query_vector = self.prob3_feat_mlps[feat_i](self.prob3_learnable_vector)
            query_vector = query_vector.expand(B, -1).unsqueeze(1)
            feat_reshaped = feat.permute(0, 1, 3, 4, 2).contiguous().view(B, T * H * W, C)
            scale = (C ** -0.5)
            attn_scores = torch.matmul(query_vector, feat_reshaped.transpose(-2, -1)) * scale
            attn_weights = torch.softmax(attn_scores, dim=-1)
            attended_feat = torch.matmul(attn_weights, feat_reshaped).squeeze(1)
            prob3_feat_output = self.prob3_feat_output_mlps[feat_i](attended_feat)
            prob3_outputs.append(prob3_feat_output)

        # Average pooling and final MLP
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

        loss_gt_retitusion = 0 if gt_retitusion is None else l1_loss(gt_retitusion[:,0], gt['gt_restitution'])

        if gt_retitusion is not None:
            print(loss_gt_retitusion)

        loss = loss_gt_retitusion
        loss_dict = {
            'total_loss': loss,
            'loss_gt_retitusion': loss_gt_retitusion,
        }

        for key_i in loss_dict.keys():
            if isinstance(loss_dict[key_i], torch.Tensor) and torch.isnan(loss_dict[key_i]):
                loss_dict[key_i] = torch.tensor(0.001, device=device, requires_grad=True)
                print("NaN loss detected, using fallback value")
        
        return loss, loss_dict

    def prepare_conditions(self, batch, x, random_uncond):
        """
        Prepares the conditioning dictionary `cond`.
        This logic is extracted from get_batch_input for reusability.
        `x` is the video tensor, used to determine batch size and device.
        """
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
        prompt_mask = rearrange(random_num < 2 * self.uncond_prob, "n -> n 1 1")
        input_mask = 1 - rearrange((random_num >= self.uncond_prob).float() * (random_num < 3 * self.uncond_prob).float(), "n -> n 1 1 1")

        null_prompt = self.get_learned_conditioning([""])
        prompt_imb = torch.where(prompt_mask, null_prompt, cond_emb.detach())

        cond_frame_index = 0
        if self.rand_cond_frame:
            cond_frame_index = random.randint(0, self.model.diffusion_model.temporal_length-1)
        if self.cond_specify == 3:
            cond_frame_index = -1
        
        if self.cond_specify == 0:
            img = x[:,:,cond_frame_index,...]
        elif self.cond_specify == 1:
            img = x[:,:,cond_frame_index,...]
            cur_batch_mask = batch['mask']
            img = img * cur_batch_mask[:,:,cond_frame_index,...]
        elif self.cond_specify == 2:
            cur_batch_mask = batch['mask']
            img = cur_batch_mask[:,:,cond_frame_index,...]
        elif self.cond_specify == 3:
            cur_batch_mask = batch['mask']
            img = cur_batch_mask[:,:,cond_frame_index,...]
        
        img = input_mask * img
        img_emb = self.embedder(img)
        img_emb = self.image_proj_model(img_emb)

        cond["c_crossattn"] = [torch.cat([prompt_imb, img_emb], dim=1)]
        return cond

        """
        计算训练损失，此版本为VJEPA骨干网络适配。

        此方法覆写了LatentVisualDiffusionv84的p_losses方法。
        核心的扩散损失计算保持不变。主要变化在于辅助的物理损失计算，
        该计算现在使用简化的交叉注意力探测机制和共享的MLP，以利用VJEPA的统一特征维度。

        Args:
            x_start (torch.Tensor): 初始潜变量 (z)。
            cond (dict): 条件信息。
            t (torch.Tensor): 时间步。
            gt (torch.Tensor): 物理属性的真实值。
            mask (torch.Tensor): 视频掩码。
            keytime_mask (torch.Tensor): 关键时间帧的掩码。
            noise (torch.Tensor, optional): 添加到x_start的噪声。
            **kwargs: 附加参数。

        Returns:
            tuple: 一个包含总损失和损失分量字典的元组。
            
        Calculates the training loss, adapted for a VJEPA backbone.

        This method overrides the p_losses from LatentVisualDiffusionv84.
        The core diffusion loss calculation remains the same. The primary change is in the
        auxiliary physics loss calculation, which now uses a simplified cross-attention
        probing mechanism with shared MLPs, leveraging the uniform feature dimensions of VJEPA.
        """
        if self.noise_strength > 0:
            noise_strength = self.noise_strength * torch.rand(x_start.shape[0], 1, 1, 1, 1, device=x_start.device)
            noise_gt = torch.randn_like(x_start)
            x_start = x_start + noise_strength * noise_gt
        
        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)

        cond['c_crossattn'][0][:,:77,:] = self.prob3_learnable_prompt
        model_output, feat_list = self.apply_model(x_noisy, t, cond, **kwargs)
        
        loss_dict = {}
        prefix = 'train' if self.training else 'val'

        loss_simple = self.get_loss(model_output, noise, mean=False).mean([1, 2, 3, 4])
        loss_dict.update({f'{prefix}/loss_simple': loss_simple.mean()})
        loss = self.l_simple_weight * loss_simple.mean()

        # 为VJEPA简化的物理损失计算
        # Simplified physics loss calculation for VJEPA
        if self.prob3_dim > 0:
            prob3_outputs = []
            
            # 1. 将可学习向量投影到统一的特征维度（仅需一次）
            # The original code did this inside the loop
            query_vector = self.prob3_feat_mlp(self.prob3_learnable_vector) # 1 x C

            for feat in feat_list:
                # feat shape: B x T x C x H x W
                B, T, C, H, W = feat.shape
                
                # 为批次扩展单个投影后的查询向量
                # Expand the single projected query vector for the batch
                q = query_vector.expand(B, -1).unsqueeze(1)  # B x 1 x C
                
                # 2. 为交叉注意力重塑特征
                # Reshape feature for cross-attention
                feat_reshaped = feat.permute(0, 1, 3, 4, 2).contiguous().view(B, T * H * W, C)  # B x (T*H*W) x C
                
                # 3. 交叉注意力: Q=query_vector, K=V=feat_tokens
                # Cross-attention: Q=query_vector, K=V=feat_tokens
                scale = (C ** -0.5)
                attn_scores = torch.matmul(q, feat_reshaped.transpose(-2, -1)) * scale  # B x 1 x (T*H*W)
                attn_weights = torch.softmax(attn_scores, dim=-1)
                attended_feat = torch.matmul(attn_weights, feat_reshaped).squeeze(1) # B x 1 x C -> B x C
                
                # 4. 使用共享的输出MLP映射回prob3_dim
                # Map back to prob3_dim using the shared output MLP
                prob3_feat_output = self.prob3_feat_output_mlp(attended_feat)  # B x prob3_dim
                prob3_outputs.append(prob3_feat_output)

            # 最终预测是来自所有特征级别的预测的平均值
            # Final prediction is the mean of predictions from all feature levels
            prob3_outputs = torch.stack(prob3_outputs, dim=1) # B x num_feat_levels x prob3_dim
            prob3_pred = prob3_outputs.mean(1)
            
            # 物理损失计算 (与原始代码一致，使用L1损失)
            # Physics Loss Calculation (assuming L1 loss as in the original)
            loss_prob3 = self.l1_loss(prob3_pred, gt)
            loss_dict.update({f'{prefix}/loss_prob3': loss_prob3})
            
            # 原始代码用物理损失*替换*了扩散损失。我们遵循这一行为。
            # The original code *replaces* the diffusion loss. We will do the same.
            loss = loss_prob3
        
        return loss, loss_dict

class LatentVisualDiffusionv84Test(LatentVisualDiffusionv84):

    def __init__(self, 
                 *args, 
                 save_result_dir=None,
                 **kwargs):
        super().__init__(*args, **kwargs)

        self.save_result_dir = save_result_dir
        if self.save_result_dir is not None and not os.path.exists(self.save_result_dir):
            os.makedirs(self.save_result_dir, exist_ok=True)

    def forward(self, x, c, gt, mask, keytime_mask, fns, **kwargs):
        # t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=self.device).long()
        t = torch.randint(50, 51, (x.shape[0],), device=self.device).long()
        if self.use_dynamic_rescale:
            x = x * extract_into_tensor(self.scale_arr, t, x.shape)
        return self.p_losses(x, c, t, gt, mask, keytime_mask, fns, **kwargs)

    def shared_step(self, batch, random_uncond, **kwargs):
        x, c, fs, gt, mask, keytime_mask, fns = self.get_batch_input(batch, random_uncond=random_uncond, return_fs=True)
        kwargs.update({"fs": fs.long()})
        loss, loss_dict = self(x, c, gt, mask, keytime_mask, fns, **kwargs)
        return loss, loss_dict

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        loss, loss_dict = self.shared_step(batch, random_uncond=self.classifier_free_guidance)
        ## sync_dist | rank_zero_only 
        self.log_dict(loss_dict, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=False)
        #self.log("epoch/global_step", self.global_step.float(), prog_bar=True, logger=True, on_step=True, on_epoch=False)
        '''
        if self.use_scheduler:
            lr = self.optimizers().param_groups[0]['lr']
            self.log('lr_abs', lr, prog_bar=True, logger=True, on_step=True, on_epoch=False, rank_zero_only=True)
        '''
        if (batch_idx+1) % self.log_every_t == 0:
            mainlogger.info(f"batch:{batch_idx}|epoch:{self.current_epoch} [globalstep:{self.global_step}]: loss={loss}")
        return loss

    def get_batch_input(self, batch, random_uncond, return_first_stage_outputs=False, return_original_cond=False, return_fs=False, return_cond_frame=False, return_original_input=False, **kwargs):
        ## x: b c t h w
        x = super().get_input(batch, self.first_stage_key) # b 2 c t h w
        # x = x.reshape(x.shape[0] * x.shape[1], x.shape[2], x.shape[3], x.shape[4], x.shape[5]) # b2 c t h w
        # ipdb.set_trace()
        ## encode video frames x to z via a 2D encoder 
        if self.major_specify == 0:       
            z = self.encode_first_stage(x)
        elif self.major_specify == 1:
            cur_batch_mask = batch['mask']#.reshape(batch['mask'].shape[0] * batch['mask'].shape[1], batch['mask'].shape[2], batch['mask'].shape[3], batch['mask'].shape[4], batch['mask'].shape[5])
            z = self.encode_first_stage(x * cur_batch_mask)
        elif self.major_specify == 2:
            x_red = batch['mask']
            # x_red = x_red.reshape(x_red.shape[0] * x_red.shape[1], x_red.shape[2], x_red.shape[3], x_red.shape[4], x_red.shape[5]) # b2 c t h w
            z = self.encode_first_stage(x_red)
        elif self.major_specify == 3:
            x_red = batch['mask']
            # x_red = x_red.reshape(x_red.shape[0] * x_red.shape[1], x_red.shape[2], x_red.shape[3], x_red.shape[4], x_red.shape[5]) # b2 c t h w
            x_concat_new = torch.cat([x[:,:,:-1,:,:], x_red[:,:,-1:,:,:]], dim=2) # last frame temporally is x_red
            z = self.encode_first_stage(x_concat_new)

        
        ## get caption condition
        cond_input = batch[self.cond_stage_key]
        # ipdb.set_trace()
        cond_input = cond_input # add for the relative formulation, repeat the '' captions

        if isinstance(cond_input, dict) or isinstance(cond_input, list):
            cond_emb = self.get_learned_conditioning(cond_input)
        else:
            cond_emb = self.get_learned_conditioning(cond_input.to(self.device))
                
        cond = {}
        ## to support classifier-free guidance, randomly drop out only text conditioning 5%, only image conditioning 5%, and both 5%.
        if random_uncond:
            random_num = torch.rand(x.size(0), device=x.device)
        else:
            random_num = torch.ones(x.size(0), device=x.device)  ## by doning so, we can get text embedding and complete img emb for inference
        prompt_mask = rearrange(random_num < 2 * self.uncond_prob, "n -> n 1 1")
        input_mask = 1 - rearrange((random_num >= self.uncond_prob).float() * (random_num < 3 * self.uncond_prob).float(), "n -> n 1 1 1")

        null_prompt = self.get_learned_conditioning([""])
        prompt_imb = torch.where(prompt_mask, null_prompt, cond_emb.detach())

        ## get conditioning frame
        cond_frame_index = 0
        if self.rand_cond_frame:
            cond_frame_index = random.randint(0, self.model.diffusion_model.temporal_length-1)
        if self.cond_specify == 3:
            cond_frame_index = -1

        
        if self.cond_specify == 0:
            img = x[:,:,cond_frame_index,...] # b c h w
        elif self.cond_specify == 1:
            img = x[:,:,cond_frame_index,...] # b c h w
            # ipdb.set_trace()
            cur_batch_mask = batch['mask']#.reshape(batch['mask'].shape[0] * batch['mask'].shape[1], batch['mask'].shape[2], batch['mask'].shape[3], batch['mask'].shape[4], batch['mask'].shape[5])
            img = img * cur_batch_mask[:,:,cond_frame_index,...]
        elif self.cond_specify == 2:
            cur_batch_mask = batch['mask']#.reshape(batch['mask'].shape[0] * batch['mask'].shape[1], batch['mask'].shape[2], batch['mask'].shape[3], batch['mask'].shape[4], batch['mask'].shape[5])
            img = cur_batch_mask[:,:,cond_frame_index,...]
        elif self.cond_specify == 3:
            cur_batch_mask = batch['mask']#.reshape(batch['mask'].shape[0] * batch['mask'].shape[1], batch['mask'].shape[2], batch['mask'].shape[3], batch['mask'].shape[4], batch['mask'].shape[5])
            img = cur_batch_mask[:,:,cond_frame_index,...]
        img = input_mask * img
        ## img: b c h w
        img_emb = self.embedder(img) ## b l c
        img_emb = self.image_proj_model(img_emb)

        if self.model.conditioning_key == 'hybrid':
            if self.interp_mode:
                # ipdb.set_trace()
                ## starting frame + (L-2 empty frames) + ending frame
                img_cat_cond = torch.zeros_like(z)
                img_cat_cond[:,:,0,:,:] = z[:,:,0,:,:]
                img_cat_cond[:,:,-1,:,:] = z[:,:,-1,:,:]
            else:
                ## simply repeat the cond_frame to match the seq_len of z
                if self.cond_specify == 0 or self.cond_specify == 3:
                    img_cat_cond = z[:,:,cond_frame_index,:,:]
                    img_cat_cond = img_cat_cond.unsqueeze(2)
                    img_cat_cond = repeat(img_cat_cond, 'b c t h w -> b c (repeat t) h w', repeat=z.shape[2])
                elif self.cond_specify == 1:
                    cur_batch_mask = batch['mask']#.reshape(batch['mask'].shape[0] * batch['mask'].shape[1], batch['mask'].shape[2], batch['mask'].shape[3], batch['mask'].shape[4], batch['mask'].shape[5])
                    img_cat_cond = self.encode_first_stage(x * cur_batch_mask)
                elif self.cond_specify == 2:
                    cur_batch_mask = batch['mask']#.reshape(batch['mask'].shape[0] * batch['mask'].shape[1], batch['mask'].shape[2], batch['mask'].shape[3], batch['mask'].shape[4], batch['mask'].shape[5])
                    img_cat_cond = self.encode_first_stage(cur_batch_mask)

            cond["c_concat"] = [img_cat_cond] # b c t h w
        cond["c_crossattn"] = [torch.cat([prompt_imb, img_emb], dim=1)] ## concat in the seq_len dim

        out = [z, cond]
        if return_first_stage_outputs:
            xrec = self.decode_first_stage(z)
            out.extend([xrec])

        if return_original_cond:
            out.append(cond_input)
        # ipdb.set_trace()
        if return_fs:
            if self.fps_condition_type == 'fs':
                fs = super().get_input(batch, 'frame_stride')
            elif self.fps_condition_type == 'fps':
                fs = super().get_input(batch, 'fps')
            # ipdb.set_trace()
            out.append(fs)
        if return_cond_frame:
            out.append(x[:,:,cond_frame_index,...].unsqueeze(2))
        if return_original_input:
            out.append(x)

        # ipdb.set_trace()
        # gt = [float(p.split('_')[-8]) for p in batch['path']]
        # gt = [p for p in batch['gt_ann']]
        gt = batch['gt_ann']
        # ipdb.set_trace()
        # gt = torch.tensor(gt).to(out[0].device)
        out.append(gt)
        # ipdb.set_trace()

        out.append(batch['mask'])
        out.append(batch['keytime_mask'])

        fn_list = [p.split('/')[-1] for p in batch['path']]
        out.append(fn_list)

        return out

    def p_losses(self, x_start, cond, t, gt, mask, keytime_mask, fns, noise=None, **kwargs):
        # assert self.noise_strength > 0
        if self.noise_strength > 0:
            b, c, f, _, _ = x_start.shape
            offset_noise = torch.randn(b, c, f, 1, 1, device=x_start.device)
            noise = default(noise, lambda: torch.randn_like(x_start) + self.noise_strength * offset_noise)
        else:
            noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)

        cond['c_crossattn'][0][:,:77,:] = self.prob3_learnable_prompt
        model_output, feat_list = self.apply_model(x_noisy, t, cond, **kwargs)
        # for feat_i in range(len(feat_list)):
        #     print(feat_i)
        #     print(feat_list[feat_i].shape)
        # ipdb.set_trace()
        
        # Process each feature through cross-attention
        prob3_outputs = []

        for feat_i, feat in enumerate(feat_list):
            # feat shape: B x T x C x H x W
            B, T, C, H, W = feat.shape
            
            # 1. Map the learnable vector to the corresponding C dimension
            # Project learnable vector to feature dimension
            query_vector = self.prob3_feat_mlps[feat_i](self.prob3_learnable_vector)  # 1 x C
            query_vector = query_vector.expand(B, -1).unsqueeze(1)  # B x 1 x C
            
            # 2. Reshape feature for cross-attention (tokens = T x H x W)
            feat_reshaped = feat.permute(0, 1, 3, 4, 2).contiguous().view(B, T * H * W, C)  # B x (T*H*W) x C
            
            # 3. Cross-attention: Q=query_vector, K=V=feat_tokens
            # Using scaled dot-product attention
            scale = (C ** -0.5)
            
            # Compute attention scores
            attn_scores = torch.matmul(query_vector, feat_reshaped.transpose(-2, -1)) * scale  # B x 1 x (T*H*W)
            attn_weights = torch.softmax(attn_scores, dim=-1)
            
            # Apply attention to values
            attended_feat = torch.matmul(attn_weights, feat_reshaped)  # B x 1 x C
            attended_feat = attended_feat.squeeze(1)  # B x C
            
            # Map back to prob3_dim
            prob3_feat_output = self.prob3_feat_output_mlps[feat_i](attended_feat)  # B x prob3_dim
            prob3_outputs.append(prob3_feat_output)

        # 4. Average pooling over all different layers and final MLP
        if prob3_outputs:
            # Stack and average across features
            prob3_combined = torch.stack(prob3_outputs, dim=1)  # B x num_feats x prob3_dim
            prob3_averaged = torch.mean(prob3_combined, dim=1)  # B x prob3_dim
            
            # Final MLP to map to scalar
            gt_retitusion = self.prob3_output_projection(prob3_averaged)  # B x 1
        else:
            gt_retitusion = None
        # TODO: for each feat in feat_list, shape is B x T x C x H x W
        # 1. have an MLP to map the self.prob3_learnable_vector into the corresponding C dim
        # 2. use the query vector as Q, the T x H x W tokens in the feat as K and V, calculate a cross-attention
        # 3. map the output back to the self.prob3_dim, and avg pooling over all different layers
        # 4. use a simple MLP to map from B x self.prob3_dim to B x 1, and calculate the L1 loss

        
        l1_loss = nn.L1Loss()
        
        # Ensure all GT tensors are on the same device as prediction tensors
        device = feat_list[0].device
        for key in gt:
            if isinstance(gt[key], torch.Tensor):
                gt[key] = gt[key].to(device)
            elif isinstance(gt[key], list):
                gt[key] = [item.to(device) if isinstance(item, torch.Tensor) else item for item in gt[key]]
                
        loss_gt_retitusion = 0 if gt_retitusion is None else l1_loss(torch.sigmoid(gt_retitusion[:,0])*0.2, gt['gt_restitution']) / 0.2
        # loss_gt_retitusion = 0 if gt_retitusion is None else l1_loss(torch.sigmoid(gt_retitusion[:,0]), gt['gt_restitution']) / 0.01


        if gt_retitusion is not None:
            print(loss_gt_retitusion)
            print('pred: ', torch.sigmoid(gt_retitusion[:,0]))
            print('gt: ', gt['gt_restitution'])

        
        # Combine all losses
        loss = loss_gt_retitusion
        
        loss_dict = {}
        loss_dict.update({
            'total_loss': loss,
            'loss_gt_retitusion': loss_gt_retitusion,
        })

        fn_2_pred = {}
        for fn_i in range(gt_retitusion.shape[0]):
            fn_2_pred[fns[fn_i]] = {}
            # ipdb.set_trace()
            fn_2_pred[fns[fn_i]]['pred'] =  gt_retitusion[fn_i][0].detach().cpu().numpy().tolist()
            fn_2_pred[fns[fn_i]]['gt'] = gt['gt_restitution'][fn_i].detach().cpu().numpy().tolist()

        
        # ipdb.set_trace()
        with open(os.path.join(self.save_result_dir, fns[0] + '.json'), 'w') as dump_f:
            json.dump(fn_2_pred, dump_f)
    

        return loss, loss_dict 
    

        """
        Override shared_step to handle the new output of get_batch_input.
        """
        forward_kwargs = {}
        signature = inspect.signature(self.forward)
        
        if 'fs' in signature.parameters:
             kwargs['return_fs'] = True
             forward_kwargs['fs'] = True
        if 'fns' in signature.parameters:
            kwargs['return_fns'] = True
            forward_kwargs['fns'] = True

        full_out = self.get_batch_input(batch, random_uncond, **kwargs)
        
        x, c, gt, mask, keytime_mask, *rest = full_out
        
        # Pass optional arguments to forward
        idx = 0
        if forward_kwargs.get('fs'):
            forward_kwargs['fs'] = rest[idx]
            idx += 1
        if forward_kwargs.get('fns'):
            forward_kwargs['fns'] = rest[idx]
            idx += 1

        loss, loss_dict = self(x, c, gt, mask, keytime_mask, **forward_kwargs)
        return loss, loss_dict


class PhysicsPredictionVJEPA(LatentVisualDiffusionv84):
    """
    This model uses VJEPA to directly extract features from raw video
    to predict physical properties. It inherits from v84 but overrides
    the core data processing and loss calculation logic to bypass
    VAE encoding and the diffusion process entirely. This version is
    specifically adapted to work with uniform-dimension features from VJEPA.
    """
    def __init__(self, *args, vjepa_feat_dim=768, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.parameterization = "vjepa_direct"
        self.noise_strength = 0
        self.l1_loss = nn.L1Loss()

        mainlogger.info(f"Re-initializing MLPs for PhysicsPredictionVJEPA to match v84 structure with uniform dimensions.")
        
        num_features = self.model.diffusion_model.num_output_features
        uniform_feat_dim = self.model.diffusion_model.vjepa_hidden_dim
        
        # To keep consistency with v84, we use ModuleLists, but all layers will have the same dimensions.
        self.prob3_feat_mlps = nn.ModuleList([
            nn.Linear(self.prob3_dim, uniform_feat_dim) for _ in range(num_features)
        ])
        self.prob3_feat_output_mlps = nn.ModuleList([
            nn.Linear(uniform_feat_dim, self.prob3_dim) for _ in range(num_features)
        ])
        
        # We still need the final projection layer from v84
        self.prob3_output_projection = nn.Linear(self.prob3_dim, 1)

        # Nullify the single MLPs to avoid confusion
        self.prob3_feat_mlp = None
        self.prob3_feat_output_mlp = None

    def get_batch_input(self, batch, random_uncond, **kwargs):
        """
        Overrides get_batch_input to bypass VAE encoding.
        The first element of the returned list is now the raw video `x`.
        """
        # 1. Get the raw video directly from the batch
        x = batch[self.first_stage_key]

        # 2. Prepare conditions using the helper method from the parent class
        cond = self.prepare_conditions(batch, x, random_uncond)

        # 3. Get other necessary data from the batch
        gt = batch['gt_ann']
        mask = batch['mask']
        keytime_mask = batch['keytime_mask']

        # The 'c_concat' key is not needed as we are not using a U-Net architecture
        if 'c_concat' in cond:
            del cond['c_concat']

        # 4. Return a list where the first element is the raw video `x`
        out = [x, cond, gt, mask, keytime_mask]
        
        # Add other optional returns for compatibility if needed by training scripts
        if kwargs.get('return_fs', False):
             if self.fps_condition_type == 'fs':
                fs = super(LatentVisualDiffusion, self).get_input(batch, 'frame_stride')
             elif self.fps_condition_type == 'fps':
                fs = super(LatentVisualDiffusion, self).get_input(batch, 'fps')
             out.append(fs)
        
        if kwargs.get('return_original_input', False):
            out.append(x) # It's just x itself
        
        # This part handles filename returns for test/validation steps
        if 'fns' in inspect.signature(self.forward).parameters or kwargs.get('return_fns', False):
            fn_list = [p.split('/')[-1] for p in batch['path']]
            out.append(fn_list)

        return out

    def p_losses(self, video_input, cond, t, gt, mask, keytime_mask, noise=None, **kwargs):
        """
        Overrides the loss function to bypass the diffusion process.
        'video_input' is now the raw video `x`. 't' and 'noise' are ignored.
        """
        # 1. Call `apply_model` which should point to VJEPA2_Direct_Wrapper.
        # `t` is passed for interface compatibility but may be ignored inside the wrapper.
        _, feat_list = self.apply_model(video_input, t, cond, **kwargs)

        # 2. Calculate the physics loss using the now-overridden helper method.
        loss, loss_dict = self.calculate_physics_loss(feat_list, gt)
        return loss, loss_dict

    def forward(self, x, c, gt, mask, keytime_mask, **kwargs):
        """
        The forward pass now takes the raw video `x` directly.
        A dummy timestep `t` is created to maintain interface compatibility.
        """
        # Create a dummy timestep `t` as it's required by the function signature
        # of p_losses, but it will be ignored.
        t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=self.device).long()
        return self.p_losses(x, c, t, gt, mask, keytime_mask, **kwargs)
    
    def shared_step(self, batch, random_uncond, **kwargs):
        '''
        We need to make sure that the returned values from get_batch_input match
        what this method expects. Let's inspect the parent method if needed.
        The base `shared_step` in DDPM calls `self.get_input` and then `self.forward`.
        LatentDiffusion overrides it, and LatentVisualDiffusionv84 does not.
        So we should follow LatentDiffusion's `shared_step`.

        It expects `x, c` from `get_input`, and then calls `self(x, c)`.
        Our `get_batch_input` returns more things. Let's adapt.
        
        For test/val steps, `LatentVisualDiffusionv84Test` uses a more complex call.
        # Let's align with the `forward` signature.
        '''
        if 'fns' in inspect.signature(self.forward).parameters:
            x, c, gt, mask, keytime_mask, fns = self.get_batch_input(batch, random_uncond=random_uncond, return_fns=True)
            loss, loss_dict = self(x, c, gt, mask, keytime_mask, fns=fns, **kwargs)
        else:
            x, c, gt, mask, keytime_mask = self.get_batch_input(batch, random_uncond=random_uncond)
            loss, loss_dict = self(x, c, gt, mask, keytime_mask, **kwargs)

        return loss, loss_dict

class PhysicsPredictionVJEPA_Base(LatentVisualDiffusionv84):
    """
    Base class for models that use VJEPA to directly extract features from raw video
    for physics prediction. This class encapsulates the common logic for bypassing
    the VAE and diffusion process.
    """
    def __init__(self, *args, **kwargs):
        # Force EMA to be disabled as it's incompatible with the VJEPA backbone.
        kwargs['use_ema'] = False
        self.use_red_circle = kwargs.pop('use_red_circle', False)
        super().__init__(*args, **kwargs)

        self.linear_start = kwargs.get('linear_start', 0.00085)
        self.linear_end = kwargs.get('linear_end', 0.012)
        self.cosine_s = kwargs.get('cosine_s', 8e-3)
        self.num_timesteps = kwargs.get('num_timesteps', 1000)
        self.parameterization = "vjepa_direct"
        self.noise_strength = 0
        self.l1_loss = nn.L1Loss()
        # self.use_red_circle = kwargs.get('use_red_circle', False)

        mainlogger.info(f"Initializing PhysicsPredictionVJEPA_Base: Setting up VJEPA-specific components.")
        
        # This check ensures that the underlying model is the VJEPA wrapper
        if not hasattr(self.model.diffusion_model, 'num_output_features') or not hasattr(self.model.diffusion_model, 'vjepa_hidden_dim'):
             raise TypeError("The underlying model for PhysicsPredictionVJEPA_Base must be VJEPA2_Direct_Wrapper with 'num_output_features' and 'vjepa_hidden_dim' attributes.")

        num_features = self.model.diffusion_model.num_output_features
        uniform_feat_dim = self.model.diffusion_model.vjepa_hidden_dim
        
        # To keep consistency with v84, we use ModuleLists for the physics prediction head
        self.prob3_feat_mlps = nn.ModuleList([
            nn.Linear(self.prob3_dim, uniform_feat_dim) for _ in range(num_features)
        ])
        self.prob3_feat_output_mlps = nn.ModuleList([
            nn.Linear(uniform_feat_dim, self.prob3_dim) for _ in range(num_features)
        ])
        
        self.prob3_output_projection = nn.Linear(self.prob3_dim, 1)

        # Nullify unused single MLPs from parent classes to avoid confusion
        self.prob3_feat_mlp = None
        self.prob3_feat_output_mlp = None

    def register_schedule(self, *args, **kwargs):
        """
        Overrides the parent DDPM's schedule registration.
        This model does not use a diffusion schedule, so this method does nothing.
        This prevents errors from the training script trying to set up a schedule.
        """
        mainlogger.info("PhysicsPredictionVJEPA_Base: Skipping diffusion schedule registration.")
        self.num_timesteps = kwargs.get('num_timesteps', 1000)
        pass

    def get_batch_input(self, batch, random_uncond, **kwargs):
        """
        Overrides get_batch_input to bypass VAE encoding.
        The first element of the returned list is now the raw video `x`.
        """
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
            print("use red circle video")
        
        if kwargs.get('return_fs', False):
             fs_key = 'frame_stride' if self.fps_condition_type == 'fs' else 'fps'
             fs = super(LatentVisualDiffusion, self).get_input(batch, fs_key)
             out.append(fs)
        
        if 'fns' in inspect.signature(self.forward).parameters or kwargs.get('return_fns', False):
            fn_list = [p.split('/')[-1] for p in batch['path']]
            out.append(fn_list)

        return out

    def forward(self, x, c, gt, mask, keytime_mask, **kwargs):
        """
        The forward pass now takes the raw video `x` directly.
        A dummy timestep `t` is created to maintain interface compatibility.
        """
        t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=self.device).long()
        return self.p_losses(x, c, t, gt, mask, keytime_mask, **kwargs)

    def shared_step(self, batch, random_uncond, **kwargs):
        """
        Standard shared step, ensures the correct inputs are fetched and passed to forward.
        """
        if 'fns' in inspect.signature(self.forward).parameters:
            x, c, gt, mask, keytime_mask, fns = self.get_batch_input(batch, random_uncond=random_uncond, return_fns=True)
            loss, loss_dict = self(x, c, gt, mask, keytime_mask, fns=fns, **kwargs)
        else:
            x, c, gt, mask, keytime_mask = self.get_batch_input(batch, random_uncond=random_uncond)
            loss, loss_dict = self(x, c, gt, mask, keytime_mask, **kwargs)

        return loss, loss_dict

class PhysicsPredictionVJEPA_Base_probe(PhysicsPredictionVJEPA_Base):
    """
    Base class for models that use VJEPA to directly extract features from raw video
    for physics prediction. This class encapsulates the common logic for bypassing
    the VAE and diffusion process.
    """
    def __init__(self, attentive_config: dict = None, direct_weight: float = 1.0, scale_weight: float = 0.5, value_weight: float = 0.0, vjepa_hidden_dim: int = 1024, num_output_features: int = 25, use_all_feat_in_attentive_probe: bool = False, *args, **kwargs):
        # 从kwargs中提取不被父类接受的参数
        self.direct_weight = direct_weight
        self.scale_weight = scale_weight
        self.value_weight = value_weight
        self.vjepa_hidden_dim = vjepa_hidden_dim
        self.num_output_features = num_output_features
        self.use_all_feat_in_attentive_probe = use_all_feat_in_attentive_probe
        
        # Force EMA to be disabled as it's incompatible with the VJEPA backbone.
        kwargs['use_ema'] = False
        super().__init__(*args, **kwargs)

        self.linear_start = kwargs.get('linear_start', 0.00085)
        self.linear_end = kwargs.get('linear_end', 0.012)
        self.cosine_s = kwargs.get('cosine_s', 8e-3)
        self.num_timesteps = kwargs.get('num_timesteps', 1000)
        self.parameterization = "vjepa_direct"
        self.noise_strength = 0
        self.l1_loss = nn.L1Loss()

        if self.use_all_feat_in_attentive_probe:
            self.prob3_feat_mlps = nn.ModuleList([
                nn.Linear(self.vjepa_hidden_dim, self.vjepa_hidden_dim) 
                for _ in range(self.num_output_features)
            ]) 
        else:
            self.prob3_feat_mlps = nn.ModuleList([
                nn.Linear(self.prob3_dim, self.vjepa_hidden_dim) 
                for _ in range(self.num_output_features)
            ]) 

        mainlogger.info(f"Initializing PhysicsPredictionVJEPA_Base: Setting up VJEPA-specific components.")
        
        # This check ensures that the underlying model is the VJEPA wrapper
        if not hasattr(self.model.diffusion_model, 'num_output_features') or not hasattr(self.model.diffusion_model, 'vjepa_hidden_dim'):
             raise TypeError("The underlying model for PhysicsPredictionVJEPA_Base must be VJEPA2_Direct_Wrapper with 'num_output_features' and 'vjepa_hidden_dim' attributes.")

        # 使用显式传递的参数而不是从diffusion_model中获取
        num_features = self.num_output_features
        uniform_feat_dim = self.vjepa_hidden_dim
        
        self.l1_loss = nn.L1Loss()
        self.bce_loss = nn.BCEWithLogitsLoss()
        
        # 设置AttentiveClassifier的默认配置
        default_attentive_config = {
            'verb_classes': {},  # 不使用verb分类
            'noun_classes': {},  # 不使用noun分类
            'action_classes': {f'restitution_{i}': i for i in range(2)},  # 物理预测类别
            'embed_dim': uniform_feat_dim,  # 使用实际的特征维度
            'num_heads': 8,
            'depth': 2,
            'use_activation_checkpointing': False
        }
        
        if attentive_config:
            default_attentive_config.update(attentive_config)
        
        # 创建AttentiveClassifier来替换原有的MLP机制
        self.attentive_classifier = AttentiveClassifier(**default_attentive_config)
        
        # 移除父类中的MLP组件，因为我们用AttentiveClassifier替换了
        # 保留prob3_learnable_vector用于兼容性（虽然AttentiveClassifier有自己的learnable queries）
        
        mainlogger.info(f"初始化PhysicsPredictionVJEPA_Base_probe: 使用AttentiveClassifier替换MLP机制")

    def calculate_vjepa_physics_loss(self, feat_list: list, gt: dict, return_prediction: bool = False) -> tuple:
        """
        使用AttentiveClassifier计算基于VJEPA特征的物理损失
        
        Args:
            feat_list: 现在是最后一层feature，维度为 [B, N, D]
            gt: 包含真实标签的字典
            return_prediction: 是否返回预测值
            
        Returns:
            tuple: (loss, loss_dict) 或 (loss, loss_dict, prediction)
        """


        if feat_list is None:
            device = self.device
        else: # feat_list现在是model_output，直接使用
            device = feat_list.device

        for key in gt:
            if isinstance(gt[key], torch.Tensor):
                gt[key] = gt[key].to(device)
            elif isinstance(gt[key], list):
                gt[key] = [item.to(device) if isinstance(item, torch.Tensor) else item for item in gt[key]]

        # bce_loss = nn.BCEWithLogitsLoss()
        l1_loss = nn.L1Loss()
        
        # 处理特征
        if feat_list is None:
            gt_retitusion = None
        else:
            # 获取最后一层特征，维度应该是 [B, N, D]
            last_layer_feat = feat_list
            
            # 直接使用AttentiveClassifier处理特征
            # last_layer_feat的维度 [B, N, D] 正好符合AttentiveClassifier的输入要求
            classifier_output = self.attentive_classifier(last_layer_feat)
            
            # 获取action分类结果（用于物理预测）
            action_logits = classifier_output['action']  # [B, num_classes]
            
            gt_retitusion = action_logits
        # import pdb; pdb.set_trace()

        # 计算损失
        if gt_retitusion is None:
            loss_gt_retitusion = torch.tensor(0.0, device=device, requires_grad=True)
        else:
            # 根据输出维度选择损失计算方式
            if gt_retitusion.shape[1] > 1:
                loss_gt_retitusion = l1_loss(gt_retitusion[:, 0], gt['gt_restitution'].float())
            else:
                loss_gt_retitusion = l1_loss(gt_retitusion.squeeze(), gt['gt_restitution'].float())
        
        loss = loss_gt_retitusion
        
        loss_dict = {
            'total_loss': loss,
            'loss_gt_retitusion': loss_gt_retitusion,
        }
        
        # NaN和Inf检查
        for key_i in loss_dict.keys():
            if isinstance(loss_dict[key_i], torch.Tensor) and (torch.isnan(loss_dict[key_i]) or torch.isinf(loss_dict[key_i])):
                mainlogger.info(f"NaN or Inf loss detected in {key_i}, using fallback value")
                loss_dict[key_i] = torch.tensor(0.001, device=device, requires_grad=True)
        
        if return_prediction:
            return loss, loss_dict, gt_retitusion
            
        return loss, loss_dict

    def shared_step(self, batch: dict, **kwargs) -> tuple[torch.Tensor, dict]:
        """
        重写后的核心训练/验证步骤，使用精简的数据流和直接的模型调用。
        此方法现在返回损失张量和包含详细损失信息的字典。
        """
        x, cond, gt, mask, keytime_mask = self.get_batch_input(batch, random_uncond=False)
        
        t = torch.zeros(x.shape[0], device=self.device, dtype=torch.long)
        model_output, feat_list = self.model.diffusion_model(x, t=t, context=None)

        if self.use_all_feat_in_attentive_probe:
            # 将feat_list中的所有特征重塑为[B, N_patches, D]，通过线性变换后再平均
            if feat_list and len(feat_list) > 0:
                # 获取model_output的形状作为目标形状
                target_shape = model_output.shape  # [B, N_patches, D]
                B, N_patches, D = target_shape
                
                # 将每个feat重塑为[B, N_patches, D]，并通过对应的线性变换
                transformed_feats = []
                for i, feat in enumerate(feat_list):
                    # feat shape: [B, T, C, H, W] -> [B, N_patches, D]
                    B_feat, T, C, H, W = feat.shape
                    # 重塑为[B, T*H*W, C]，相当于[B, N_patches, D]
                    feat_reshaped = feat.permute(0, 2, 1, 3, 4).contiguous().view(B_feat, C, T*H*W).transpose(1, 2)
                    # 现在feat_reshaped的形状是[B, T*H*W, C]，即[B, N_patches, D]
                    
                    # 通过对应的线性变换层
                    feat_idx = min(i, len(self.prob3_feat_mlps) - 1)  # 防止索引越界
                    feat_transformed = self.prob3_feat_mlps[feat_idx](feat_reshaped)
                    transformed_feats.append(feat_transformed)
                
                # 对所有经过线性变换的特征进行平均
                model_output = torch.stack(transformed_feats, dim=0).mean(dim=0)  # [B, N_patches, D]
            # 否则使用原始的model_output
        else:
            model_output = model_output

        if feat_list is None or not feat_list:
            loss = torch.tensor(0.0, device=self.device, requires_grad=True)
            loss_dict = {'total_loss': loss.detach()}
        else:
            loss, loss_dict = self.calculate_vjepa_physics_loss(model_output, gt)
        
        
        return loss, loss_dict

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        """
        标准的训练步骤，调用shared_step并记录返回的损失
        """
        loss, loss_dict = self.shared_step(batch)
        
        # 在这里检查总损失
        if not torch.isfinite(loss):
            mainlogger.warning(f"Non-finite loss detected at batch {batch_idx}: {loss}")
            # 返回一个小的有效损失值
            loss = torch.tensor(0.001, device=loss.device, requires_grad=True)
        
        prefix = 'train'
        log_dict_prefixed = {f'{prefix}/{k}': v for k, v in loss_dict.items()}
        
        # 记录损失
        if (batch_idx+1) % self.log_every_t == 0:
            mainlogger.info(f"batch:{batch_idx}|epoch:{self.current_epoch} [globalstep:{self.global_step}]: loss={loss}")
        
        self.log_dict(log_dict_prefixed, prog_bar=True, logger=True, on_step=True, on_epoch=False)
        
        return loss

    def register_schedule(self, *args, **kwargs):
        """
        Overrides the parent DDPM's schedule registration.
        This model does not use a diffusion schedule, so this method does nothing.
        This prevents errors from the training script trying to set up a schedule.
        """
        mainlogger.info("PhysicsPredictionVJEPA_Base: Skipping diffusion schedule registration.")
        self.num_timesteps = kwargs.get('num_timesteps', 1000)
        pass

class PhysicsPredictionVJEPA_Base_Test_probe(PhysicsPredictionVJEPA_Base_probe):
    """
    专门用于评估 `PhysicsPredictionVJEPA_Base_probe` 模型性能的测试类。

    该类继承了 `PhysicsPredictionVJEPA_Base_probe` 的核心VJEPA预测框架，
    并通过重写验证步骤 `validation_step` 来增加结果保存功能。
    它遵循了标准的测试类设计模式：在初始化时指定结果保存目录，
    在评估时运行预测并将预测值、真实值与样本文件名一一对应地
    保存为文本文件，以便于进行离线的定量分析。
    """
    def __init__(self, *args, save_result_dir=None, **kwargs):
        """
        初始化测试类。

        Args:
            save_result_dir (str, optional): 保存测试结果的目录路径。
            *args, **kwargs: 传递给父类的参数。
        """
        super().__init__(*args, **kwargs)
        self.save_result_dir = save_result_dir
        if self.save_result_dir is not None:
            os.makedirs(self.save_result_dir, exist_ok=True)
            self.result_file_path = os.path.join(self.save_result_dir, 'results.txt')
            # 清空旧的结果文件，以便于新的测试运行
            if os.path.exists(self.result_file_path):
                os.remove(self.result_file_path)
            mainlogger.info(f"Test results for PhysicsPredictionVJEPA_Base_Test_probe will be saved to {self.result_file_path}")

    def get_batch_input_for_test(self, batch: dict) -> tuple:
        """
        为测试专门准备的数据输入方法，额外返回文件名列表 `fns` 用于结果保存。
        
        Args:
            batch: 输入批次数据
            
        Returns:
            tuple: (x, cond, gt, mask, keytime_mask, fns) 包含所有必要的数据和文件名
        """
        # 复用父类的核心逻辑来获取基本数据
        x, cond, gt, mask, keytime_mask = self.get_batch_input(batch, random_uncond=False)
        
        # 新增逻辑：提取并返回文件名
        # The path in batch is a list of paired paths, e.g., ['a1.mp4', 'a2.mp4', 'b1.mp4', 'b2.mp4']
        fns = batch['path']
        
        return x, cond, gt, mask, keytime_mask, fns

    def run_prediction_step(self, batch: dict) -> tuple:
        """
        运行单步预测，并返回包括预测值和文件名在内的所有相关信息。
        
        Args:
            batch: 输入批次数据
            
        Returns:
            tuple: (loss, loss_dict, predictions, gt, fns) 包含损失、预测值和相关信息
        """
        x, cond, gt, mask, keytime_mask, fns = self.get_batch_input_for_test(batch)
        
        t = torch.zeros(x.shape[0], device=self.device, dtype=torch.long)
        model_output, feat_list = self.model.diffusion_model(x, t=t, context=None)

        if self.use_all_feat_in_attentive_probe:
            # 将所有层的特征重塑为 [B, N_patches, D]，通过线性变换后再平均
            if feat_list and isinstance(feat_list, list):
                transformed_features = []
                for i, feat in enumerate(feat_list):
                    # 重塑为 [B, N_patches, D]
                    B, T, C, H, W = feat.shape
                    feat_reshaped = feat.permute(0, 1, 3, 4, 2).contiguous().view(B, T * H * W, C)
                    
                    # 通过对应的线性变换层
                    feat_idx = min(i, len(self.prob3_feat_mlps) - 1)  # 防止索引越界
                    feat_transformed = self.prob3_feat_mlps[feat_idx](feat_reshaped)
                    transformed_features.append(feat_transformed)
                
                # 平均所有经过线性变换的特征
                model_output = torch.stack(transformed_features, dim=0).mean(dim=0)  # [B, N_patches, D]
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
        
        return loss, loss_dict, predictions, gt, fns

    @torch.no_grad()
    def validation_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        """
        在验证集上执行一个步骤，记录损失并保存详细的预测结果。
        此方法的保存逻辑严格遵循 PhysicsPredictionVJEPA_Base_Test 中的标准。
        
        Args:
            batch: 输入批次数据
            batch_idx: 批次索引
            
        Returns:
            torch.Tensor: 计算得到的损失值
        """
        loss, loss_dict, predictions, ground_truth, fns = self.run_prediction_step(batch)

        # 记录损失到logger
        prefix = 'val'
        num_pairs = len(fns) // 2 if fns else 0
        self.log_dict(loss_dict, prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True, batch_size=num_pairs)

        # 保存预测结果到文本文件，严格遵循 PhysicsPredictionVJEPA_Base_Test 的格式
        if self.save_result_dir and predictions is not None and fns:
            # 处理预测值，确保格式正确
            if predictions.dim() > 1 and predictions.shape[1] > 1:
                # 如果预测结果有多个维度，取第一个
                preds = predictions[:, 0].squeeze().cpu().tolist()
            else:
                # 如果只有一个维度，直接使用
                preds = predictions.squeeze().cpu().tolist()
            
            gts = ground_truth['gt_restitution'].cpu().tolist()
            
            # 确保preds和gts都是列表格式
            if not isinstance(preds, list): 
                preds = [preds]
            if not isinstance(gts, list): 
                gts = [gts]

            # 保存到文本文件，每行一个结果
            with open(self.result_file_path, 'a') as f:
                for i in range(len(gts)):
                    f.write(f"File: {fns[i]}, Prediction: {preds[i]:.6f}, GroundTruth: {gts[i]}\n")

        print('save_result_dir: ', self.save_result_dir)

        return loss

class PhysicsPredictionVJEPA_Base_Test(PhysicsPredictionVJEPA_Base):
    """
    一个专门用于测试和验证的 PhysicsPredictionVJEPA_Base 版本。

    它继承了所有核心的VJEPA物理预测逻辑，并增加了以下测试功能：
    1.  初始化时能够指定一个目录用于保存结果。
    2.  能够获取并处理每个样本的文件名。
    3.  能够将每个样本的预测结果和真实标签保存到指定文件中。
    4.  提供更详细的验证步骤日志。
    """
    def __init__(self, *args, save_result_dir=None, **kwargs):
        """
        初始化函数。

        Args:
            save_result_dir (str, optional): 保存测试结果的目录路径。 Defaults to None.
            *args, **kwargs: 传递给父类的参数。
        """
        super().__init__(*args, **kwargs)
        self.save_result_dir = save_result_dir
        if self.save_result_dir:
            os.makedirs(self.save_result_dir, exist_ok=True)
            self.result_file_path = os.path.join(self.save_result_dir, 'results.txt')
            # 清空旧的结果文件，以便于新的测试运行
            if os.path.exists(self.result_file_path):
                os.remove(self.result_file_path)
            mainlogger.info(f"Test results will be saved to {self.result_file_path}")

    def get_batch_input(self, batch, random_uncond, **kwargs):
        """
        重写基类方法，以确保文件名列表 `fns` 总能被返回，用于测试。
        """
        kwargs['return_fns'] = True
        return super().get_batch_input(batch, random_uncond, **kwargs)
    
    def forward(self, x, c, gt, mask, keytime_mask, fns, **kwargs):
        """
        重写 forward 以接收文件名列表 `fns`。
        """
        t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=self.device).long()
        return self.p_losses(x, c, t, gt, mask, keytime_mask, fns=fns, **kwargs)

    def p_losses(self, x_start, cond, t, gt, mask, keytime_mask, fns, **kwargs):
        """
        重写 p_losses 以接收并处理文件名列表 `fns`。
        """
        model_output, feat_list = self.model.diffusion_model(x_start, t, **cond)

        prob3_outputs = []
        if feat_list:
            for i, feat in enumerate(feat_list):
                B, T, C, H, W = feat.shape
                query_vector = self.prob3_feat_mlps[i](self.prob3_learnable_vector)
                q = query_vector.expand(B, -1).unsqueeze(1)
                feat_reshaped = feat.permute(0, 1, 3, 4, 2).contiguous().view(B, T * H * W, C)
                scale = (C ** -0.5)
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

            # --- 新增的日志记录逻辑 ---
            if self.save_result_dir and fns:
                preds = gt_prediction.squeeze().cpu().tolist()
                gts = gt['gt_restitution'].cpu().tolist()
                
                if not isinstance(preds, list): preds = [preds]
                if not isinstance(gts, list): gts = [gts]

                with open(self.result_file_path, 'a') as f:
                    for i in range(len(gts)):
                        f.write(f"File: {fns[i]}, Prediction: {preds[i]:.6f}, GroundTruth: {gts[i]}\n")
        
        loss_dict['total_loss'] = loss.item()
        return loss, loss_dict

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        """
        专门的验证步骤，用于打印详细日志。
        """
        loss, loss_dict = self.shared_step(batch, random_uncond=False)
        self.log_dict(loss_dict, prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        
        if (batch_idx + 1) % 10 == 0: # 每10个batch打印一次日志
            mainlogger.info(f"validation batch:{batch_idx}|epoch:{self.current_epoch} [globalstep:{self.global_step}]: loss={loss.item():.6f}")

        print('save_result_dir: ', self.save_result_dir)
        return loss

class PhysicsPredictionVJEPAElasticityLoss(PhysicsPredictionVJEPA_Base):
    """
    This model uses VJEPA to directly extract features from raw video
    to predict physical properties. It inherits from the VJEPA base class
    and uses the standard physics loss calculation from LatentVisualDiffusionv84.
    """
    def p_losses(self, video_input, cond, t, gt, mask, keytime_mask, noise=None, **kwargs):
        """
        Overrides the loss function to bypass the diffusion process.
        'video_input' is now the raw video `x`. 't' and 'noise' are ignored.
        """

        _, feat_list = self.apply_model(video_input, t, cond, **kwargs)
        # This will call the calculate_physics_loss from LatentVisualDiffusionv84
        return self.calculate_physics_loss(feat_list, gt)

class LatentVisualDiffusionv84LogLoss(LatentVisualDiffusionv84):
    def __init__(self, direct_weight=1.0, scale_weight=0.5, value_weight=0.5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.direct_weight = direct_weight
        self.scale_weight = scale_weight
        self.value_weight = value_weight
    
    def p_losses(self, x_start, cond, t, gt, mask, keytime_mask, noise=None, **kwargs):
        # assert self.noise_strength > 0
        if self.noise_strength > 0:
            b, c, f, _, _ = x_start.shape
            offset_noise = torch.randn(b, c, f, 1, 1, device=x_start.device)
            noise = default(noise, lambda: torch.randn_like(x_start) + self.noise_strength * offset_noise)
        else:
            noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)

        cond['c_crossattn'][0][:,:77,:] = self.prob3_learnable_prompt
        model_output, feat_list = self.apply_model(x_noisy, t, cond, **kwargs)
        # for feat_i in range(len(feat_list)):
        #     print(feat_i)
        #     print(feat_list[feat_i].shape)
        # ipdb.set_trace()
        
        # Process each feature through cross-attention
        prob3_outputs = []

        for feat_i, feat in enumerate(feat_list):
            # feat shape: B x T x C x H x W
            B, T, C, H, W = feat.shape
            
            # 1. Map the learnable vector to the corresponding C dimension
            # Project learnable vector to feature dimension
            query_vector = self.prob3_feat_mlps[feat_i](self.prob3_learnable_vector)  # 1 x C
            query_vector = query_vector.expand(B, -1).unsqueeze(1)  # B x 1 x C
            
            # 2. Reshape feature for cross-attention (tokens = T x H x W)
            feat_reshaped = feat.permute(0, 1, 3, 4, 2).contiguous().view(B, T * H * W, C)  # B x (T*H*W) x C
            
            # 3. Cross-attention: Q=query_vector, K=V=feat_tokens
            # Using scaled dot-product attention
            scale = (C ** -0.5)
            
            # Compute attention scores
            attn_scores = torch.matmul(query_vector, feat_reshaped.transpose(-2, -1)) * scale  # B x 1 x (T*H*W)
            attn_weights = torch.softmax(attn_scores, dim=-1)
            
            # Apply attention to values
            attended_feat = torch.matmul(attn_weights, feat_reshaped)  # B x 1 x C
            attended_feat = attended_feat.squeeze(1)  # B x C
            
            # Map back to prob3_dim
            prob3_feat_output = self.prob3_feat_output_mlps[feat_i](attended_feat)  # B x prob3_dim
            prob3_outputs.append(prob3_feat_output)

        # 4. Average pooling over all different layers and final MLP
        if prob3_outputs:
            # Stack and average across features
            prob3_combined = torch.stack(prob3_outputs, dim=1)  # B x num_feats x prob3_dim
            prob3_averaged = torch.mean(prob3_combined, dim=1)  # B x prob3_dim
            
            # Final MLP to map to scalar
            gt_retitusion = self.prob3_output_projection(prob3_averaged)  # B x 1
        else:
            gt_retitusion = None
        # TODO: for each feat in feat_list, shape is B x T x C x H x W
        # 1. have an MLP to map the self.prob3_learnable_vector into the corresponding C dim
        # 2. use the query vector as Q, the T x H x W tokens in the feat as K and V, calculate a cross-attention
        # 3. map the output back to the self.prob3_dim, and avg pooling over all different layers
        # 4. use a simple MLP to map from B x self.prob3_dim to B x 1, and calculate the L1 loss

        
        # l1_loss = nn.L1Loss()
        l1_loss = ImprovedScaleAwareL1Loss(direct_weight=self.direct_weight, scale_weight=self.scale_weight, value_weight=self.value_weight)
        
        # Ensure all GT tensors are on the same device as prediction tensors
        device = feat_list[0].device
        for key in gt:
            if isinstance(gt[key], torch.Tensor):
                gt[key] = gt[key].to(device)
            elif isinstance(gt[key], list):
                gt[key] = [item.to(device) if isinstance(item, torch.Tensor) else item for item in gt[key]]
                
        loss_gt_retitusion = 0 if gt_retitusion is None else l1_loss(torch.sigmoid(gt_retitusion[:,0]) * 0.01, gt['gt_restitution'])
        print('pred: ', torch.sigmoid(gt_retitusion[:,0]))
        print('gt: ', gt['gt_restitution'] / 0.01)

        if self.scale_weight == 0 and self.value_weight == 0:
            loss_gt_retitusion = loss_gt_retitusion / 0.01

        if gt_retitusion is not None:
            print(loss_gt_retitusion)
        


        
        # Combine all losses
        loss = loss_gt_retitusion
        
        loss_dict = {}
        loss_dict.update({
            'total_loss': loss,
            'loss_gt_retitusion': loss_gt_retitusion,
        })

        # # Add to p_losses method before returning loss
        # if torch.isnan(loss):
        #     # Use a small constant value to avoid training failure
        #     loss = torch.tensor(0.1, device=loss.device, requires_grad=True)
        #     print("NaN loss detected, using fallback value")

        for key_i in loss_dict.keys():
            if isinstance(loss_dict[key_i], torch.Tensor) and torch.isnan(loss_dict[key_i]):
                loss_dict[key_i] = torch.tensor(0.001, device=device, requires_grad=True)
                print("NaN loss detected, using fallback value")
    

        return loss, loss_dict

    def calculate_physics_loss(self, feat_list, gt):
        """
        Calculates the physics-based loss, including the specific Scale-Aware L1 Loss for this class.
        """
        # Process each feature through cross-attention
        prob3_outputs = []

        for feat_i, feat in enumerate(feat_list):
            B, T, C, H, W = feat.shape
            query_vector = self.prob3_feat_mlps[feat_i](self.prob3_learnable_vector)
            query_vector = query_vector.expand(B, -1).unsqueeze(1)
            feat_reshaped = feat.permute(0, 1, 3, 4, 2).contiguous().view(B, T * H * W, C)
            scale = (C ** -0.5)
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
                
        loss_gt_retitusion = 0 if gt_retitusion is None else l1_loss(torch.sigmoid(gt_retitusion[:,0]) * 0.01, gt['gt_restitution'])
        
        if self.training:
            print('pred: ', torch.sigmoid(gt_retitusion[:,0]))
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
                print("NaN loss detected, using fallback value")
    
        return loss, loss_dict

class PhysicsPredictionVJEPALogLoss(PhysicsPredictionVJEPA_Base, LatentVisualDiffusionv84LogLoss):
    """
    VJEPA-based physics prediction model that incorporates the LogL1Loss logic
    from LatentVisualDiffusionv84LogLoss.
    """
    def p_losses(self, video_input, cond, t, gt, mask, keytime_mask, noise=None, **kwargs):
        """
        Gets VJEPA features and calculates the specific LogLoss.
        """
        _, feat_list = self.apply_model(video_input, t, cond, **kwargs)
        # Due to MRO, this calls calculate_physics_loss from LatentVisualDiffusionv84LogLoss
        return self.calculate_physics_loss(feat_list, gt)

class PhysicsPredictionVJEPALogLoss_probe(PhysicsPredictionVJEPA_Base_probe, LatentVisualDiffusionv84LogLoss):
    """
    VJEPA-based physics prediction model that incorporates the LogL1Loss logic
    from LatentVisualDiffusionv84LogLoss.
    """
    def calculate_vjepa_physics_loss(self, feat_list: list, gt: dict, return_prediction: bool = False) -> tuple:
        """
        使用AttentiveClassifier计算基于VJEPA特征的物理损失
        
        Args:
            feat_list: 现在是最后一层feature，维度为 [B, N, D]
            gt: 包含真实标签的字典
            return_prediction: 是否返回预测值
            
        Returns:
            tuple: (loss, loss_dict) 或 (loss, loss_dict, prediction)
        """


        if feat_list is None:
            device = self.device
        else: # feat_list现在是model_output，直接使用
            device = feat_list.device

        for key in gt:
            if isinstance(gt[key], torch.Tensor):
                gt[key] = gt[key].to(device)
            elif isinstance(gt[key], list):
                gt[key] = [item.to(device) if isinstance(item, torch.Tensor) else item for item in gt[key]]

        # bce_loss = nn.BCEWithLogitsLoss()
        l1_loss = ImprovedScaleAwareL1Loss(direct_weight=self.direct_weight, scale_weight=self.scale_weight, value_weight=self.value_weight)
        
        # 处理特征
        if feat_list is None:
            gt_retitusion = None
        else:
            # 获取最后一层特征，维度应该是 [B, N, D]
            last_layer_feat = feat_list
            
            # 直接使用AttentiveClassifier处理特征
            # last_layer_feat的维度 [B, N, D] 正好符合AttentiveClassifier的输入要求
            classifier_output = self.attentive_classifier(last_layer_feat)
            
            # 获取action分类结果（用于物理预测）
            action_logits = classifier_output['action']  # [B, num_classes]
            
            gt_retitusion = action_logits
        # import pdb; pdb.set_trace()

        # 计算损失
        # if gt_retitusion is None:
        #     loss_gt_retitusion = torch.tensor(0.0, device=device, requires_grad=True)
        # else:
        #     # 根据输出维度选择损失计算方式
        #     if gt_retitusion.shape[1] > 1:
        #         loss_gt_retitusion = l1_loss(gt_retitusion[:, 0], gt['gt_restitution'].float())
        #     else:
        #         loss_gt_retitusion = l1_loss(gt_retitusion.squeeze(), gt['gt_restitution'].float())

        loss_gt_retitusion = 0 if gt_retitusion is None else l1_loss(torch.sigmoid(gt_retitusion[:,0]) * 0.01, gt['gt_restitution'])
        
        if self.training:
            print('pred: ', torch.sigmoid(gt_retitusion[:,0]))
            print('gt: ', gt['gt_restitution'] / 0.01)
        
        loss = loss_gt_retitusion
        
        loss_dict = {
            'total_loss': loss,
            'loss_gt_retitusion': loss_gt_retitusion,
        }
        
        # NaN和Inf检查
        for key_i in loss_dict.keys():
            if isinstance(loss_dict[key_i], torch.Tensor) and (torch.isnan(loss_dict[key_i]) or torch.isinf(loss_dict[key_i])):
                mainlogger.info(f"NaN or Inf loss detected in {key_i}, using fallback value")
                loss_dict[key_i] = torch.tensor(0.001, device=device, requires_grad=True)
        
        if return_prediction:
            return loss, loss_dict, gt_retitusion
            
        return loss, loss_dict


class LatentVisualDiffusionv84FrictionLoss(LatentVisualDiffusionv84):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def p_losses(self, x_start, cond, t, gt, mask, keytime_mask, noise=None, **kwargs):
        # assert self.noise_strength > 0
        if self.noise_strength > 0:
            b, c, f, _, _ = x_start.shape
            offset_noise = torch.randn(b, c, f, 1, 1, device=x_start.device)
            noise = default(noise, lambda: torch.randn_like(x_start) + self.noise_strength * offset_noise)
        else:
            noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)

        cond['c_crossattn'][0][:,:77,:] = self.prob3_learnable_prompt
        model_output, feat_list = self.apply_model(x_noisy, t, cond, **kwargs)
        # for feat_i in range(len(feat_list)):
        #     print(feat_i)
        #     print(feat_list[feat_i].shape)
        # ipdb.set_trace()
        
        # Process each feature through cross-attention
        prob3_outputs = []

        for feat_i, feat in enumerate(feat_list):
            # feat shape: B x T x C x H x W
            B, T, C, H, W = feat.shape
            
            # 1. Map the learnable vector to the corresponding C dimension
            # Project learnable vector to feature dimension
            query_vector = self.prob3_feat_mlps[feat_i](self.prob3_learnable_vector)  # 1 x C
            query_vector = query_vector.expand(B, -1).unsqueeze(1)  # B x 1 x C
            
            # 2. Reshape feature for cross-attention (tokens = T x H x W)
            feat_reshaped = feat.permute(0, 1, 3, 4, 2).contiguous().view(B, T * H * W, C)  # B x (T*H*W) x C
            
            # 3. Cross-attention: Q=query_vector, K=V=feat_tokens
            # Using scaled dot-product attention
            scale = (C ** -0.5)
            
            # Compute attention scores
            attn_scores = torch.matmul(query_vector, feat_reshaped.transpose(-2, -1)) * scale  # B x 1 x (T*H*W)
            attn_weights = torch.softmax(attn_scores, dim=-1)
            
            # Apply attention to values
            attended_feat = torch.matmul(attn_weights, feat_reshaped)  # B x 1 x C
            attended_feat = attended_feat.squeeze(1)  # B x C
            
            # Map back to prob3_dim
            prob3_feat_output = self.prob3_feat_output_mlps[feat_i](attended_feat)  # B x prob3_dim
            prob3_outputs.append(prob3_feat_output)

        # 4. Average pooling over all different layers and final MLP
        if prob3_outputs:
            # Stack and average across features
            prob3_combined = torch.stack(prob3_outputs, dim=1)  # B x num_feats x prob3_dim
            prob3_averaged = torch.mean(prob3_combined, dim=1)  # B x prob3_dim
            
            # Final MLP to map to scalar
            gt_retitusion = self.prob3_output_projection(prob3_averaged)  # B x 1
        else:
            gt_retitusion = None
        # TODO: for each feat in feat_list, shape is B x T x C x H x W
        # 1. have an MLP to map the self.prob3_learnable_vector into the corresponding C dim
        # 2. use the query vector as Q, the T x H x W tokens in the feat as K and V, calculate a cross-attention
        # 3. map the output back to the self.prob3_dim, and avg pooling over all different layers
        # 4. use a simple MLP to map from B x self.prob3_dim to B x 1, and calculate the L1 loss

        
        l1_loss = nn.L1Loss()
        
        # Ensure all GT tensors are on the same device as prediction tensors
        device = feat_list[0].device
        for key in gt:
            if isinstance(gt[key], torch.Tensor):
                gt[key] = gt[key].to(device)
            elif isinstance(gt[key], list):
                gt[key] = [item.to(device) if isinstance(item, torch.Tensor) else item for item in gt[key]]
                
        loss_gt_retitusion = 0 if gt_retitusion is None else l1_loss(torch.sigmoid(gt_retitusion[:,0]) * 0.2, gt['gt_restitution']) / 0.2
        print('pred: ', torch.sigmoid(gt_retitusion[:,0]))
        print('gt: ', gt['gt_restitution'] / 0.2)

        if gt_retitusion is not None:
            print(loss_gt_retitusion)
        


        
        # Combine all losses
        loss = loss_gt_retitusion
        
        loss_dict = {}
        loss_dict.update({
            'total_loss': loss,
            'loss_gt_retitusion': loss_gt_retitusion,
        })

        # # Add to p_losses method before returning loss
        # if torch.isnan(loss):
        #     # Use a small constant value to avoid training failure
        #     loss = torch.tensor(0.1, device=loss.device, requires_grad=True)
        #     print("NaN loss detected, using fallback value")

        for key_i in loss_dict.keys():
            if isinstance(loss_dict[key_i], torch.Tensor) and torch.isnan(loss_dict[key_i]):
                loss_dict[key_i] = torch.tensor(0.001, device=device, requires_grad=True)
                print("NaN loss detected, using fallback value")
    

        return loss, loss_dict

    def calculate_physics_loss(self, feat_list, gt):
        """
        Calculates the physics-based loss, including the specific Friction Loss for this class.
        """
        prob3_outputs = []
        for feat_i, feat in enumerate(feat_list):
            B, T, C, H, W = feat.shape
            query_vector = self.prob3_feat_mlps[feat_i](self.prob3_learnable_vector)
            query_vector = query_vector.expand(B, -1).unsqueeze(1)
            feat_reshaped = feat.permute(0, 1, 3, 4, 2).contiguous().view(B, T * H * W, C)
            scale = (C ** -0.5)
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
                
        loss_gt_retitusion = 0 if gt_retitusion is None else l1_loss(torch.sigmoid(gt_retitusion[:,0]) * 0.2, gt['gt_restitution']) / 0.2
        
        if self.training:
            print('pred: ', torch.sigmoid(gt_retitusion[:,0]))
            print('gt: ', gt['gt_restitution'] / 0.2)

        if gt_retitusion is not None and self.training:
            print(loss_gt_retitusion)
        
        loss = loss_gt_retitusion
        loss_dict = {'total_loss': loss, 'loss_gt_retitusion': loss_gt_retitusion}

        for key_i in loss_dict.keys():
            if isinstance(loss_dict[key_i], torch.Tensor) and torch.isnan(loss_dict[key_i]):
                loss_dict[key_i] = torch.tensor(0.001, device=device, requires_grad=True)
                print("NaN loss detected, using fallback value")
    
        return loss, loss_dict

class PhysicsPredictionVJEPAFrictionLoss(PhysicsPredictionVJEPA_Base, LatentVisualDiffusionv84FrictionLoss):
    """
    VJEPA-based physics prediction model that incorporates the friction loss logic
    from LatentVisualDiffusionv84FrictionLoss.
    """
    def p_losses(self, video_input, cond, t, gt, mask, keytime_mask, noise=None, **kwargs):
        """
        Gets VJEPA features and calculates the specific FrictionLoss.
        """
        _, feat_list = self.apply_model(video_input, t, cond, **kwargs)
        # Due to MRO, this calls calculate_physics_loss from LatentVisualDiffusionv84FrictionLoss
        return self.calculate_physics_loss(feat_list, gt)


    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_result_dir = '/work/guanqi/DynamiCrafter/2.11_result_test4_1800itr'
        if not os.path.exists(self.save_result_dir):
            os.makedirs(self.save_result_dir, exist_ok=True)

class PhysicsPredictionVJEPAFrictionLoss_probe(PhysicsPredictionVJEPA_Base_probe, LatentVisualDiffusionv84FrictionLoss):
    """
    VJEPA-based physics prediction model that incorporates the friction loss logic
    from LatentVisualDiffusionv84FrictionLoss.
    """
    def calculate_vjepa_physics_loss(self, feat_list: list, gt: dict, return_prediction: bool = False) -> tuple:
        """
        使用AttentiveClassifier计算基于VJEPA特征的物理损失
        
        Args:
            feat_list: 现在是最后一层feature，维度为 [B, N, D]
            gt: 包含真实标签的字典
            return_prediction: 是否返回预测值
            
        Returns:
            tuple: (loss, loss_dict) 或 (loss, loss_dict, prediction)
        """


        if feat_list is None:
            device = self.device
        else: # feat_list现在是model_output，直接使用
            device = feat_list.device

        for key in gt:
            if isinstance(gt[key], torch.Tensor):
                gt[key] = gt[key].to(device)
            elif isinstance(gt[key], list):
                gt[key] = [item.to(device) if isinstance(item, torch.Tensor) else item for item in gt[key]]

        # bce_loss = nn.BCEWithLogitsLoss()
        l1_loss = nn.L1Loss()
        # l1_loss = ImprovedScaleAwareL1Loss(direct_weight=self.direct_weight, scale_weight=self.scale_weight, value_weight=self.value_weight)
        
        # 处理特征
        if feat_list is None:
            gt_retitusion = None
        else:
            # 获取最后一层特征，维度应该是 [B, N, D]
            last_layer_feat = feat_list
            
            # 直接使用AttentiveClassifier处理特征
            # last_layer_feat的维度 [B, N, D] 正好符合AttentiveClassifier的输入要求
            classifier_output = self.attentive_classifier(last_layer_feat)
            
            # 获取action分类结果（用于物理预测）
            action_logits = classifier_output['action']  # [B, num_classes]
            
            gt_retitusion = action_logits
        # import pdb; pdb.set_trace()

        # 计算损失
        # if gt_retitusion is None:
        #     loss_gt_retitusion = torch.tensor(0.0, device=device, requires_grad=True)
        # else:
        #     # 根据输出维度选择损失计算方式
        #     if gt_retitusion.shape[1] > 1:
        #         loss_gt_retitusion = l1_loss(gt_retitusion[:, 0], gt['gt_restitution'].float())
        #     else:
        #         loss_gt_retitusion = l1_loss(gt_retitusion.squeeze(), gt['gt_restitution'].float())

        loss_gt_retitusion = 0 if gt_retitusion is None else l1_loss(torch.sigmoid(gt_retitusion[:,0]) * 0.2, gt['gt_restitution']) / 0.2
        
        if self.training:
            print('pred: ', torch.sigmoid(gt_retitusion[:,0]))
            print('gt: ', gt['gt_restitution'] / 0.2)

        if gt_retitusion is not None and self.training:
            print(loss_gt_retitusion)

        loss = loss_gt_retitusion
        
        loss_dict = {
            'total_loss': loss,
            'loss_gt_retitusion': loss_gt_retitusion,
        }
        
        # NaN和Inf检查
        for key_i in loss_dict.keys():
            if isinstance(loss_dict[key_i], torch.Tensor) and (torch.isnan(loss_dict[key_i]) or torch.isinf(loss_dict[key_i])):
                mainlogger.info(f"NaN or Inf loss detected in {key_i}, using fallback value")
                loss_dict[key_i] = torch.tensor(0.001, device=device, requires_grad=True)
        
        if return_prediction:
            return loss, loss_dict, gt_retitusion
            
        return loss, loss_dict



class LatentVisualDiffusionv87(LatentVisualDiffusionv47):
    def __init__(self, *args, prob3_dim=0, major_specify=0, cond_specify=0, collect_map_together=0, concat_2_feat=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.concat_2_feat = concat_2_feat
        self.collect_map_together = collect_map_together
        self.major_specify = major_specify # [0, 1 mask, 2 red circle, 3 red circle only]
        self.cond_specify = cond_specify # [0, 1, 2, 3 red circle only]
        self.prob3_dim = prob3_dim
        self.prob3_learnable_vector = nn.Parameter(torch.randn((1, self.prob3_dim)) * 0.02)
        self.p_decoder = None
        
        # Initialize MLPs for cross-attention based on typical feature dimensions
        # From the output, we can see the channel dimensions: 320, 320, 320, 640, 640, 640, 1280, 1280, 1280, etc.
        feat_dims = [320, 320, 320, 320, 640, 640, 640, 1280, 1280, 1280, 1280, 1280, 1280, 1280, 1280, 1280, 1280, 1280, 1280, 640, 640, 640, 320, 320, 320]
        
        self.prob3_feat_mlps = nn.ModuleList([
            nn.Linear(self.prob3_dim, feat_dim) for feat_dim in feat_dims
        ])
        if self.collect_map_together == 0:
            self.prob3_feat_output_mlps = nn.ModuleList([
                nn.Linear(feat_dim, self.prob3_dim) for feat_dim in feat_dims
            ])
        elif self.collect_map_together == 1:
            self.prob3_feat_output_mlps = nn.Linear(sum(feat_dims), self.prob3_dim)
        else:
            assert self.collect_map_together == 2
            self.prob3_feat_output_mlps = nn.Linear(sum(feat_dims)*2, 1)
        # self.prob3_output_projection = nn.Linear(self.prob3_dim*2, 1)
        if self.collect_map_together != 2:
            if concat_2_feat == False: # minus
                self.prob3_output_projection = nn.Sequential(
                    nn.Linear(self.prob3_dim, 256),
                    nn.ReLU(),
                    nn.Linear(256, 128),
                    nn.ReLU(), 
                    nn.Linear(128, 1)
                )
            else:
                self.prob3_output_projection = nn.Sequential(
                    nn.Linear(self.prob3_dim*2, 256),
                    nn.ReLU(),
                    nn.Linear(256, 128),
                    nn.ReLU(), 
                    nn.Linear(128, 1)
                )
        self.prob3_learnable_prompt = nn.Parameter(
            torch.randn((1, 77, 1024)) * 0.02  # Small standard deviation for more stable initial training
        )
        self._freeze_all_but_some(self.ft_weights_list)

    def _freeze_all_but_some(self, ft_weights_list):
        for name, param in self.named_parameters():
            if 'prob3' not in name:
                param.requires_grad = False
            if 'prob3' in name:
                param.requires_grad = True

        # Then, unfreeze based on patterns in ft_weights_list
        for ft_weight in ft_weights_list:
            # Check if this is a regex pattern (contains wildcard * or other regex characters)
            if any(c in ft_weight for c in ".*+?^$[](){}|\\"):
                # It's a regex pattern, compile it
                pattern = re.compile(ft_weight.replace(".", "\\.").replace("*", ".*"))
                
                # Find matching parameters and unfreeze them
                for name, param in self.named_parameters():
                    if pattern.search(name):
                        param.requires_grad = True
            else:
                # Original direct matching code
                for name, param in self.named_parameters():
                    if ft_weight in name:
                        param.requires_grad = True

    def configure_optimizers(self):
        """ configure_optimizers for LatentDiffusion """
        lr = self.learning_rate

        params_1 = [p for n, p in self.named_parameters() if p.requires_grad and ('prob3' in n or 'p_decoder' in n)]
        for n, p in self.named_parameters():
            if p.requires_grad:
                print(n)
        # ipdb.set_trace()
        # params_2 = [p for n, p in self.named_parameters() if p.requires_grad and 'p_decoder' not in n]
        mainlogger.info(f"@Training [{len(params_1)}] Full Paramters.")
        # mainlogger.info(f"@Fine-tuning [{len(params_2)}] Full Paramters.")

        
        optimizer = torch.optim.AdamW([
            {"params": params_1, "lr": lr},
            # {"params": params_2, "lr": self.ft_lr}
        ])

        # ipdb.set_trace()

        ## lr scheduler
        if self.use_scheduler:
            ipdb.set_trace()
            mainlogger.info("Setting up scheduler...")
            lr_scheduler = self.configure_schedulers(optimizer)
            return [optimizer], [lr_scheduler]
        
        return optimizer

    def p_losses(self, x_start, cond, t, gt, mask, keytime_mask, noise=None, **kwargs):
        # assert self.noise_strength > 0
        if self.noise_strength > 0:
            b, c, f, _, _ = x_start.shape
            offset_noise = torch.randn(b, c, f, 1, 1, device=x_start.device)
            noise = default(noise, lambda: torch.randn_like(x_start) + self.noise_strength * offset_noise)
        else:
            noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)

        cond['c_crossattn'][0][:,:77,:] = self.prob3_learnable_prompt
        model_output, feat_list = self.apply_model(x_noisy, t, cond, **kwargs)
        # for feat_i in range(len(feat_list)):
        #     print(feat_i)
        #     print(feat_list[feat_i].shape)
        # ipdb.set_trace()
        
        # Process each feature through cross-attention
        prob3_outputs = []

        for feat_i, feat in enumerate(feat_list):
            # feat shape: B x T x C x H x W
            B, T, C, H, W = feat.shape
            
            # 1. Map the learnable vector to the corresponding C dimension
            # Project learnable vector to feature dimension
            query_vector = self.prob3_feat_mlps[feat_i](self.prob3_learnable_vector)  # 1 x C
            query_vector = query_vector.expand(B, -1).unsqueeze(1)  # B x 1 x C
            
            # 2. Reshape feature for cross-attention (tokens = T x H x W)
            feat_reshaped = feat.permute(0, 1, 3, 4, 2).contiguous().view(B, T * H * W, C)  # B x (T*H*W) x C
            
            # 3. Cross-attention: Q=query_vector, K=V=feat_tokens
            # Using scaled dot-product attention
            scale = (C ** -0.5)
            
            # Compute attention scores
            attn_scores = torch.matmul(query_vector, feat_reshaped.transpose(-2, -1)) * scale  # B x 1 x (T*H*W)
            attn_weights = torch.softmax(attn_scores, dim=-1)
            
            # Apply attention to values
            attended_feat = torch.matmul(attn_weights, feat_reshaped)  # B x 1 x C
            attended_feat = attended_feat.squeeze(1)  # B x C
            
            # Map back to prob3_dim
            if self.collect_map_together == 0:
                prob3_feat_output = self.prob3_feat_output_mlps[feat_i](attended_feat)  # B x prob3_dim
                prob3_outputs.append(prob3_feat_output)
            else:
                prob3_outputs.append(attended_feat)

        # 4. Average pooling over all different layers and final MLP
        if prob3_outputs:
            if self.collect_map_together == 0:
                # Stack and average across features
                prob3_combined = torch.stack(prob3_outputs, dim=1)  # B x num_feats x prob3_dim
                prob3_averaged = torch.mean(prob3_combined, dim=1)  # B x prob3_dim

                # TODO: reshape to B/2 (real B) x 2 prob3_dim
                prob3_averaged = prob3_averaged.reshape(int(prob3_averaged.shape[0]/2), 2, prob3_averaged.shape[1]) # b x 2 x prob3_dim
                if self.concat_2_feat == False:
                    prob3_diff = prob3_averaged[:, 0, :] - prob3_averaged[:, 1, :]
                    # Final MLP to map to scalar
                    gt_retitusion = self.prob3_output_projection(prob3_diff)  # B/2 x 1
                else:
                    prob3_diff = torch.cat([prob3_averaged[:, 0, :], prob3_averaged[:, 1, :]], dim=1)
                    # Final MLP to map to scalar
                    gt_retitusion = self.prob3_output_projection(prob3_diff)  # B/2 x 1
            elif self.collect_map_together == 1:
                prob3_combined = torch.cat(prob3_outputs, dim=1) # B x L
                prob3_averaged = self.prob3_feat_output_mlps(prob3_combined) # B x prob3_dim
                prob3_averaged = prob3_averaged.reshape(int(prob3_averaged.shape[0]/2), 2, prob3_averaged.shape[1]) # b x 2 x prob3_dim
                if self.concat_2_feat == False:
                    prob3_diff = prob3_averaged[:, 0, :] - prob3_averaged[:, 1, :]
                    # Final MLP to map to scalar
                    gt_retitusion = self.prob3_output_projection(prob3_diff)  # B/2 x 1
                else:
                    prob3_diff = torch.cat([prob3_averaged[:, 0, :], prob3_averaged[:, 1, :]], dim=1)
                    # Final MLP to map to scalar
                    gt_retitusion = self.prob3_output_projection(prob3_diff)  # B/2 x 1
            else:
                assert self.collect_map_together == 2
                prob3_combined = torch.cat(prob3_outputs, dim=1) # B x L
                prob3_combined = prob3_combined.reshape(int(prob3_combined.shape[0]/2), 2 * prob3_combined.shape[1]) # b x 2 L
                gt_retitusion = self.prob3_feat_output_mlps(prob3_combined) # b x 1

        else:
            gt_retitusion = None
        # TODO: for each feat in feat_list, shape is B x T x C x H x W
        # 1. have an MLP to map the self.prob3_learnable_vector into the corresponding C dim
        # 2. use the query vector as Q, the T x H x W tokens in the feat as K and V, calculate a cross-attention
        # 3. map the output back to the self.prob3_dim, and avg pooling over all different layers
        # 4. use a simple MLP to map from B x self.prob3_dim to B x 1, and calculate the L1 loss

        
        # l1_loss = nn.L1Loss()
        bce_loss = nn.BCEWithLogitsLoss()
        
        # Ensure all GT tensors are on the same device as prediction tensors
        device = feat_list[0].device
        for key in gt:
            if isinstance(gt[key], torch.Tensor):
                gt[key] = gt[key].to(device)
            elif isinstance(gt[key], list):
                gt[key] = [item.to(device) if isinstance(item, torch.Tensor) else item for item in gt[key]]
        loss_gt_retitusion = 0 if gt_retitusion is None else bce_loss(gt_retitusion[:,0], gt['gt_restitution'].float())


        if gt_retitusion is not None:
            print(loss_gt_retitusion)
        


        
        # Combine all losses
        loss = loss_gt_retitusion
        
        loss_dict = {}
        loss_dict.update({
            'total_loss': loss,
            'loss_gt_retitusion': loss_gt_retitusion,
        })

        # # Add to p_losses method before returning loss
        # if torch.isnan(loss):
        #     # Use a small constant value to avoid training failure
        #     loss = torch.tensor(0.1, device=loss.device, requires_grad=True)
        #     print("NaN loss detected, using fallback value")

        for key_i in loss_dict.keys():
            if isinstance(loss_dict[key_i], torch.Tensor) and torch.isnan(loss_dict[key_i]):
                loss_dict[key_i] = torch.tensor(0.001, device=device, requires_grad=True)
                print("NaN loss detected, using fallback value")
    

        return loss, loss_dict 

    def get_batch_input(self, batch, random_uncond, return_first_stage_outputs=False, return_original_cond=False, return_fs=False, return_cond_frame=False, return_original_input=False, **kwargs):
        ## x: b c t h w
        x = super().get_input(batch, self.first_stage_key) # b 2 c t h w
        x = x.reshape(x.shape[0] * x.shape[1], x.shape[2], x.shape[3], x.shape[4], x.shape[5]) # b2 c t h w
        # ipdb.set_trace()
        ## encode video frames x to z via a 2D encoder 
        if self.major_specify == 0:       
            z = self.encode_first_stage(x)
        elif self.major_specify == 1:
            cur_batch_mask = batch['mask'].reshape(batch['mask'].shape[0] * batch['mask'].shape[1], batch['mask'].shape[2], batch['mask'].shape[3], batch['mask'].shape[4], batch['mask'].shape[5])
            z = self.encode_first_stage(x * cur_batch_mask)
        elif self.major_specify == 2:
            x_red = batch['mask']
            x_red = x_red.reshape(x_red.shape[0] * x_red.shape[1], x_red.shape[2], x_red.shape[3], x_red.shape[4], x_red.shape[5]) # b2 c t h w
            z = self.encode_first_stage(x_red)
        elif self.major_specify == 3:
            x_red = batch['mask']
            x_red = x_red.reshape(x_red.shape[0] * x_red.shape[1], x_red.shape[2], x_red.shape[3], x_red.shape[4], x_red.shape[5]) # b2 c t h w
            x_concat_new = torch.cat([x[:,:,:-1,:,:], x_red[:,:,-1:,:,:]], dim=2) # last frame temporally is x_red
            z = self.encode_first_stage(x_concat_new)

        
        ## get caption condition
        cond_input = batch[self.cond_stage_key]
        # ipdb.set_trace()
        cond_input = cond_input + cond_input # add for the relative formulation, repeat the '' captions

        if isinstance(cond_input, dict) or isinstance(cond_input, list):
            cond_emb = self.get_learned_conditioning(cond_input)
        else:
            cond_emb = self.get_learned_conditioning(cond_input.to(self.device))
                
        cond = {}
        ## to support classifier-free guidance, randomly drop out only text conditioning 5%, only image conditioning 5%, and both 5%.
        if random_uncond:
            random_num = torch.rand(x.size(0), device=x.device)
        else:
            random_num = torch.ones(x.size(0), device=x.device)  ## by doning so, we can get text embedding and complete img emb for inference
        prompt_mask = rearrange(random_num < 2 * self.uncond_prob, "n -> n 1 1")
        input_mask = 1 - rearrange((random_num >= self.uncond_prob).float() * (random_num < 3 * self.uncond_prob).float(), "n -> n 1 1 1")

        null_prompt = self.get_learned_conditioning([""])
        prompt_imb = torch.where(prompt_mask, null_prompt, cond_emb.detach())

        ## get conditioning frame
        cond_frame_index = 0
        if self.rand_cond_frame:
            cond_frame_index = random.randint(0, self.model.diffusion_model.temporal_length-1)
        if self.cond_specify == 3:
            cond_frame_index = -1

        
        if self.cond_specify == 0:
            img = x[:,:,cond_frame_index,...] # b c h w
        elif self.cond_specify == 1:
            img = x[:,:,cond_frame_index,...] # b c h w
            # ipdb.set_trace()
            cur_batch_mask = batch['mask'].reshape(batch['mask'].shape[0] * batch['mask'].shape[1], batch['mask'].shape[2], batch['mask'].shape[3], batch['mask'].shape[4], batch['mask'].shape[5])
            img = img * cur_batch_mask[:,:,cond_frame_index,...]
        elif self.cond_specify == 2:
            cur_batch_mask = batch['mask'].reshape(batch['mask'].shape[0] * batch['mask'].shape[1], batch['mask'].shape[2], batch['mask'].shape[3], batch['mask'].shape[4], batch['mask'].shape[5])
            img = cur_batch_mask[:,:,cond_frame_index,...]
        elif self.cond_specify == 3:
            cur_batch_mask = batch['mask'].reshape(batch['mask'].shape[0] * batch['mask'].shape[1], batch['mask'].shape[2], batch['mask'].shape[3], batch['mask'].shape[4], batch['mask'].shape[5])
            img = cur_batch_mask[:,:,cond_frame_index,...]
        img = input_mask * img
        ## img: b c h w
        img_emb = self.embedder(img) ## b l c
        img_emb = self.image_proj_model(img_emb)

        if self.model.conditioning_key == 'hybrid':
            if self.interp_mode:
                # ipdb.set_trace()
                ## starting frame + (L-2 empty frames) + ending frame
                img_cat_cond = torch.zeros_like(z)
                img_cat_cond[:,:,0,:,:] = z[:,:,0,:,:]
                img_cat_cond[:,:,-1,:,:] = z[:,:,-1,:,:]
            else:
                ## simply repeat the cond_frame to match the seq_len of z
                if self.cond_specify == 0 or self.cond_specify == 3:
                    img_cat_cond = z[:,:,cond_frame_index,:,:]
                    img_cat_cond = img_cat_cond.unsqueeze(2)
                    img_cat_cond = repeat(img_cat_cond, 'b c t h w -> b c (repeat t) h w', repeat=z.shape[2])
                elif self.cond_specify == 1:
                    cur_batch_mask = batch['mask'].reshape(batch['mask'].shape[0] * batch['mask'].shape[1], batch['mask'].shape[2], batch['mask'].shape[3], batch['mask'].shape[4], batch['mask'].shape[5])
                    img_cat_cond = self.encode_first_stage(x * cur_batch_mask)
                elif self.cond_specify == 2:
                    cur_batch_mask = batch['mask'].reshape(batch['mask'].shape[0] * batch['mask'].shape[1], batch['mask'].shape[2], batch['mask'].shape[3], batch['mask'].shape[4], batch['mask'].shape[5])
                    img_cat_cond = self.encode_first_stage(cur_batch_mask)

            cond["c_concat"] = [img_cat_cond] # b c t h w
        cond["c_crossattn"] = [torch.cat([prompt_imb, img_emb], dim=1)] ## concat in the seq_len dim

        out = [z, cond]
        if return_first_stage_outputs:
            xrec = self.decode_first_stage(z)
            out.extend([xrec])

        if return_original_cond:
            out.append(cond_input)
        # ipdb.set_trace()
        if return_fs:
            if self.fps_condition_type == 'fs':
                fs = super().get_input(batch, 'frame_stride')
            elif self.fps_condition_type == 'fps':
                fs = super().get_input(batch, 'fps')
            # ipdb.set_trace()
            out.append(torch.cat([fs, fs], dim=0)) # repeat twice for the relative formulation
        if return_cond_frame:
            out.append(x[:,:,cond_frame_index,...].unsqueeze(2))
        if return_original_input:
            out.append(x)

        # ipdb.set_trace()
        # gt = [float(p.split('_')[-8]) for p in batch['path']]
        # gt = [p for p in batch['gt_ann']]
        gt = batch['gt_ann']
        # ipdb.set_trace()
        # gt = torch.tensor(gt).to(out[0].device)
        out.append(gt)
        # ipdb.set_trace()

        out.append(batch['mask'].reshape(batch['mask'].shape[0] * batch['mask'].shape[1], batch['mask'].shape[2], batch['mask'].shape[3], batch['mask'].shape[4], batch['mask'].shape[5]))
        out.append(batch['keytime_mask'].reshape(batch['keytime_mask'].shape[0] * batch['keytime_mask'].shape[1], batch['keytime_mask'].shape[2], batch['keytime_mask'].shape[3], batch['keytime_mask'].shape[4], batch['keytime_mask'].shape[5]))

        return out

class LatentVisualDiffusionv87Test(LatentVisualDiffusionv87):

    def __init__(self, 
                 *args, 
                 save_result_dir=None,
                 **kwargs):
        super().__init__(*args, **kwargs)

        self.save_result_dir = save_result_dir
        if self.save_result_dir is not None and not os.path.exists(self.save_result_dir):
            os.makedirs(self.save_result_dir, exist_ok=True)

    def forward(self, x, c, gt, mask, keytime_mask, fns, **kwargs):
        # t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=self.device).long()
        t = torch.randint(50, 51, (x.shape[0],), device=self.device).long()
        if self.use_dynamic_rescale:
            x = x * extract_into_tensor(self.scale_arr, t, x.shape)
        return self.p_losses(x, c, t, gt, mask, keytime_mask, fns, **kwargs)

    def shared_step(self, batch, random_uncond, **kwargs):
        x, c, fs, gt, mask, keytime_mask, fns = self.get_batch_input(batch, random_uncond=random_uncond, return_fs=True)
        kwargs.update({"fs": fs.long()})
        loss, loss_dict = self(x, c, gt, mask, keytime_mask, fns, **kwargs)
        return loss, loss_dict

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        loss, loss_dict = self.shared_step(batch, random_uncond=self.classifier_free_guidance)
        ## sync_dist | rank_zero_only 
        self.log_dict(loss_dict, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=False)
        #self.log("epoch/global_step", self.global_step.float(), prog_bar=True, logger=True, on_step=True, on_epoch=False)
        '''
        if self.use_scheduler:
            lr = self.optimizers().param_groups[0]['lr']
            self.log('lr_abs', lr, prog_bar=True, logger=True, on_step=True, on_epoch=False, rank_zero_only=True)
        '''
        if (batch_idx+1) % self.log_every_t == 0:
            mainlogger.info(f"batch:{batch_idx}|epoch:{self.current_epoch} [globalstep:{self.global_step}]: loss={loss}")
        return loss

    def get_batch_input(self, batch, random_uncond, return_first_stage_outputs=False, return_original_cond=False, return_fs=False, return_cond_frame=False, return_original_input=False, **kwargs):
        ## x: b c t h w
        x = super().get_input(batch, self.first_stage_key) # b 2 c t h w
        x = x.reshape(x.shape[0] * x.shape[1], x.shape[2], x.shape[3], x.shape[4], x.shape[5]) # b2 c t h w
        # ipdb.set_trace()
        ## encode video frames x to z via a 2D encoder 
        if self.major_specify == 0:       
            z = self.encode_first_stage(x)
        elif self.major_specify == 1:
            cur_batch_mask = batch['mask'].reshape(batch['mask'].shape[0] * batch['mask'].shape[1], batch['mask'].shape[2], batch['mask'].shape[3], batch['mask'].shape[4], batch['mask'].shape[5])
            z = self.encode_first_stage(x * cur_batch_mask)
        elif self.major_specify == 2:
            x_red = batch['mask']
            x_red = x_red.reshape(x_red.shape[0] * x_red.shape[1], x_red.shape[2], x_red.shape[3], x_red.shape[4], x_red.shape[5]) # b2 c t h w
            z = self.encode_first_stage(x_red)
        elif self.major_specify == 3:
            x_red = batch['mask']
            x_red = x_red.reshape(x_red.shape[0] * x_red.shape[1], x_red.shape[2], x_red.shape[3], x_red.shape[4], x_red.shape[5]) # b2 c t h w
            x_concat_new = torch.cat([x[:,:,:-1,:,:], x_red[:,:,-1:,:,:]], dim=2) # last frame temporally is x_red
            z = self.encode_first_stage(x_concat_new)

        
        ## get caption condition
        cond_input = batch[self.cond_stage_key]
        # ipdb.set_trace()
        cond_input = cond_input + cond_input # add for the relative formulation, repeat the '' captions

        if isinstance(cond_input, dict) or isinstance(cond_input, list):
            cond_emb = self.get_learned_conditioning(cond_input)
        else:
            cond_emb = self.get_learned_conditioning(cond_input.to(self.device))
                
        cond = {}
        ## to support classifier-free guidance, randomly drop out only text conditioning 5%, only image conditioning 5%, and both 5%.
        if random_uncond:
            random_num = torch.rand(x.size(0), device=x.device)
        else:
            random_num = torch.ones(x.size(0), device=x.device)  ## by doning so, we can get text embedding and complete img emb for inference
        prompt_mask = rearrange(random_num < 2 * self.uncond_prob, "n -> n 1 1")
        input_mask = 1 - rearrange((random_num >= self.uncond_prob).float() * (random_num < 3 * self.uncond_prob).float(), "n -> n 1 1 1")

        null_prompt = self.get_learned_conditioning([""])
        prompt_imb = torch.where(prompt_mask, null_prompt, cond_emb.detach())

        ## get conditioning frame
        cond_frame_index = 0
        if self.rand_cond_frame:
            cond_frame_index = random.randint(0, self.model.diffusion_model.temporal_length-1)
        if self.cond_specify == 3:
            cond_frame_index = -1
        
        if self.cond_specify == 0:
            img = x[:,:,cond_frame_index,...] # b c h w
        elif self.cond_specify == 1:
            img = x[:,:,cond_frame_index,...] # b c h w
            # ipdb.set_trace()
            cur_batch_mask = batch['mask'].reshape(batch['mask'].shape[0] * batch['mask'].shape[1], batch['mask'].shape[2], batch['mask'].shape[3], batch['mask'].shape[4], batch['mask'].shape[5])
            img = img * cur_batch_mask[:,:,cond_frame_index,...]
        elif self.cond_specify == 2:
            cur_batch_mask = batch['mask'].reshape(batch['mask'].shape[0] * batch['mask'].shape[1], batch['mask'].shape[2], batch['mask'].shape[3], batch['mask'].shape[4], batch['mask'].shape[5])
            img = cur_batch_mask[:,:,cond_frame_index,...]
        elif self.cond_specify == 3:
            cur_batch_mask = batch['mask'].reshape(batch['mask'].shape[0] * batch['mask'].shape[1], batch['mask'].shape[2], batch['mask'].shape[3], batch['mask'].shape[4], batch['mask'].shape[5])
            img = cur_batch_mask[:,:,cond_frame_index,...]
        img = input_mask * img
        ## img: b c h w
        img_emb = self.embedder(img) ## b l c
        img_emb = self.image_proj_model(img_emb)

        if self.model.conditioning_key == 'hybrid':
            if self.interp_mode:
                # ipdb.set_trace()
                ## starting frame + (L-2 empty frames) + ending frame
                img_cat_cond = torch.zeros_like(z)
                img_cat_cond[:,:,0,:,:] = z[:,:,0,:,:]
                img_cat_cond[:,:,-1,:,:] = z[:,:,-1,:,:]
            else:
                ## simply repeat the cond_frame to match the seq_len of z
                if self.cond_specify == 0 or self.cond_specify == 3:
                    img_cat_cond = z[:,:,cond_frame_index,:,:]
                    img_cat_cond = img_cat_cond.unsqueeze(2)
                    img_cat_cond = repeat(img_cat_cond, 'b c t h w -> b c (repeat t) h w', repeat=z.shape[2])
                elif self.cond_specify == 1:
                    cur_batch_mask = batch['mask'].reshape(batch['mask'].shape[0] * batch['mask'].shape[1], batch['mask'].shape[2], batch['mask'].shape[3], batch['mask'].shape[4], batch['mask'].shape[5])
                    img_cat_cond = self.encode_first_stage(x * cur_batch_mask)
                elif self.cond_specify == 2:
                    cur_batch_mask = batch['mask'].reshape(batch['mask'].shape[0] * batch['mask'].shape[1], batch['mask'].shape[2], batch['mask'].shape[3], batch['mask'].shape[4], batch['mask'].shape[5])
                    img_cat_cond = self.encode_first_stage(cur_batch_mask)

            cond["c_concat"] = [img_cat_cond] # b c t h w
        cond["c_crossattn"] = [torch.cat([prompt_imb, img_emb], dim=1)] ## concat in the seq_len dim

        out = [z, cond]
        if return_first_stage_outputs:
            xrec = self.decode_first_stage(z)
            out.extend([xrec])

        if return_original_cond:
            out.append(cond_input)
        # ipdb.set_trace()
        if return_fs:
            if self.fps_condition_type == 'fs':
                fs = super().get_input(batch, 'frame_stride')
            elif self.fps_condition_type == 'fps':
                fs = super().get_input(batch, 'fps')
            # ipdb.set_trace()
            out.append(torch.cat([fs, fs], dim=0)) # repeat twice for the relative formulation
        if return_cond_frame:
            out.append(x[:,:,cond_frame_index,...].unsqueeze(2))
        if return_original_input:
            out.append(x)

        # ipdb.set_trace()
        # gt = [float(p.split('_')[-8]) for p in batch['path']]
        # gt = [p for p in batch['gt_ann']]
        gt = batch['gt_ann']
        # ipdb.set_trace()
        # gt = torch.tensor(gt).to(out[0].device)
        out.append(gt)
        # ipdb.set_trace()

        out.append(batch['mask'].reshape(batch['mask'].shape[0] * batch['mask'].shape[1], batch['mask'].shape[2], batch['mask'].shape[3], batch['mask'].shape[4], batch['mask'].shape[5]))
        out.append(batch['keytime_mask'].reshape(batch['keytime_mask'].shape[0] * batch['keytime_mask'].shape[1], batch['keytime_mask'].shape[2], batch['keytime_mask'].shape[3], batch['keytime_mask'].shape[4], batch['keytime_mask'].shape[5]))

        fn_list = [p for p in batch['path']]
        out.append(fn_list)

        return out

    def p_losses(self, x_start, cond, t, gt, mask, keytime_mask, fns, noise=None, **kwargs):
        # assert self.noise_strength > 0
        if self.noise_strength > 0:
            b, c, f, _, _ = x_start.shape
            offset_noise = torch.randn(b, c, f, 1, 1, device=x_start.device)
            noise = default(noise, lambda: torch.randn_like(x_start) + self.noise_strength * offset_noise)
        else:
            noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)

        cond['c_crossattn'][0][:,:77,:] = self.prob3_learnable_prompt
        model_output, feat_list = self.apply_model(x_noisy, t, cond, **kwargs)
        # for feat_i in range(len(feat_list)):
        #     print(feat_i)
        #     print(feat_list[feat_i].shape)
        # ipdb.set_trace()
        
        # Process each feature through cross-attention
        prob3_outputs = []

        for feat_i, feat in enumerate(feat_list):
            # feat shape: B x T x C x H x W
            B, T, C, H, W = feat.shape
            
            # 1. Map the learnable vector to the corresponding C dimension
            # Project learnable vector to feature dimension
            query_vector = self.prob3_feat_mlps[feat_i](self.prob3_learnable_vector)  # 1 x C
            query_vector = query_vector.expand(B, -1).unsqueeze(1)  # B x 1 x C
            
            # 2. Reshape feature for cross-attention (tokens = T x H x W)
            feat_reshaped = feat.permute(0, 1, 3, 4, 2).contiguous().view(B, T * H * W, C)  # B x (T*H*W) x C
            
            # 3. Cross-attention: Q=query_vector, K=V=feat_tokens
            # Using scaled dot-product attention
            scale = (C ** -0.5)
            
            # Compute attention scores
            attn_scores = torch.matmul(query_vector, feat_reshaped.transpose(-2, -1)) * scale  # B x 1 x (T*H*W)
            attn_weights = torch.softmax(attn_scores, dim=-1)
            
            # Apply attention to values
            attended_feat = torch.matmul(attn_weights, feat_reshaped)  # B x 1 x C
            attended_feat = attended_feat.squeeze(1)  # B x C
            
            # Map back to prob3_dim
            if self.collect_map_together == 0:
                prob3_feat_output = self.prob3_feat_output_mlps[feat_i](attended_feat)  # B x prob3_dim
                prob3_outputs.append(prob3_feat_output)
            else:
                prob3_outputs.append(attended_feat)

        # 4. Average pooling over all different layers and final MLP
        if prob3_outputs:
            if self.collect_map_together == 0:
                # Stack and average across features
                prob3_combined = torch.stack(prob3_outputs, dim=1)  # B x num_feats x prob3_dim
                prob3_averaged = torch.mean(prob3_combined, dim=1)  # B x prob3_dim

                # TODO: reshape to B/2 (real B) x 2 prob3_dim
                prob3_averaged = prob3_averaged.reshape(int(prob3_averaged.shape[0]/2), 2, prob3_averaged.shape[1]) # b x 2 x prob3_dim
                if self.concat_2_feat == False:
                    prob3_diff = prob3_averaged[:, 0, :] - prob3_averaged[:, 1, :]
                    # Final MLP to map to scalar
                    gt_retitusion = self.prob3_output_projection(prob3_diff)  # B/2 x 1
                else:
                    prob3_diff = torch.cat([prob3_averaged[:, 0, :], prob3_averaged[:, 1, :]], dim=1)
                    # Final MLP to map to scalar
                    gt_retitusion = self.prob3_output_projection(prob3_diff)  # B/2 x 1
            elif self.collect_map_together == 1:
                prob3_combined = torch.cat(prob3_outputs, dim=1) # B x L
                prob3_averaged = self.prob3_feat_output_mlps(prob3_combined) # B x prob3_dim
                prob3_averaged = prob3_averaged.reshape(int(prob3_averaged.shape[0]/2), 2, prob3_averaged.shape[1]) # b x 2 x prob3_dim
                if self.concat_2_feat == False:
                    prob3_diff = prob3_averaged[:, 0, :] - prob3_averaged[:, 1, :]
                    # Final MLP to map to scalar
                    gt_retitusion = self.prob3_output_projection(prob3_diff)  # B/2 x 1
                else:
                    prob3_diff = torch.cat([prob3_averaged[:, 0, :], prob3_averaged[:, 1, :]], dim=1)
                    # Final MLP to map to scalar
                    gt_retitusion = self.prob3_output_projection(prob3_diff)  # B/2 x 1
            else:
                assert self.collect_map_together == 2
                prob3_combined = torch.cat(prob3_outputs, dim=1) # B x L
                prob3_combined = prob3_combined.reshape(int(prob3_combined.shape[0]/2), 2 * prob3_combined.shape[1]) # b x 2 L
                gt_retitusion = self.prob3_feat_output_mlps(prob3_combined) # b x 1

        else:
            gt_retitusion = None
        # TODO: for each feat in feat_list, shape is B x T x C x H x W
        # 1. have an MLP to map the self.prob3_learnable_vector into the corresponding C dim
        # 2. use the query vector as Q, the T x H x W tokens in the feat as K and V, calculate a cross-attention
        # 3. map the output back to the self.prob3_dim, and avg pooling over all different layers
        # 4. use a simple MLP to map from B x self.prob3_dim to B x 1, and calculate the L1 loss

        
        # l1_loss = nn.L1Loss()
        bce_loss = nn.BCEWithLogitsLoss()
        
        # Ensure all GT tensors are on the same device as prediction tensors
        device = feat_list[0].device
        for key in gt:
            if isinstance(gt[key], torch.Tensor):
                gt[key] = gt[key].to(device)
            elif isinstance(gt[key], list):
                gt[key] = [item.to(device) if isinstance(item, torch.Tensor) else item for item in gt[key]]
                
        loss_gt_retitusion = 0 if gt_retitusion is None else bce_loss(gt_retitusion[:,0], gt['gt_restitution'].float())


        if gt_retitusion is not None:
            print(loss_gt_retitusion)
        
        fn_2_pred = {}
        for fn_i in range(gt_retitusion.shape[0]):
            fn_2_pred[fns[fn_i]] = {}
            # ipdb.set_trace()
            fn_2_pred[fns[fn_i]]['pred'] =  gt_retitusion[fn_i][0].detach().cpu().numpy().tolist()
            fn_2_pred[fns[fn_i]]['gt'] = gt['gt_restitution'][fn_i].detach().cpu().numpy().tolist()

        with open(os.path.join(self.save_result_dir, fns[0] + '.json'), 'w') as dump_f:
            json.dump(fn_2_pred, dump_f)
        
        # Combine all losses
        loss = loss_gt_retitusion
        
        loss_dict = {}
        loss_dict.update({
            'total_loss': loss,
            'loss_gt_retitusion': loss_gt_retitusion,
        })

        # # Add to p_losses method before returning loss
        # if torch.isnan(loss):
        #     # Use a small constant value to avoid training failure
        #     loss = torch.tensor(0.1, device=loss.device, requires_grad=True)
        #     print("NaN loss detected, using fallback value")

        for key_i in loss_dict.keys():
            if isinstance(loss_dict[key_i], torch.Tensor) and torch.isnan(loss_dict[key_i]):
                loss_dict[key_i] = torch.tensor(0.001, device=device, requires_grad=True)
                print("NaN loss detected, using fallback value")
    

        return loss, loss_dict 


class PhysicsPredictionVJEPAv87_original(LatentVisualDiffusionv87):
    """
    该模型将 LatentVisualDiffusionv87 的物理预测框架与 VJEPA 骨干网络相结合。
    它继承了v87处理成对输入（paired inputs）的数据准备逻辑，但用VJEPA替换了
    U-Net作为特征提取器，并相应地调整了训练和损失计算流程。

    此版本精确地复刻了父类中 `collect_map_together` 的所有模式。
    """
    def __init__(self, vjepa_hidden_dim: int, num_output_features: int, *args, **kwargs):
        """
        初始化函数。

        Args:
            vjepa_hidden_dim (int): VJEPA模型输出的特征的统一维度。
            num_output_features (int): VJEPA模型输出的特征层的数量。
            *args, **kwargs: 传递给父类 LatentVisualDiffusionv87 的参数，
                             其中必须包含 `collect_map_together`。
        """
        # 在调用父类__init__之前，先提取并移除当前类特有的参数
        self.use_red_circle = kwargs.pop('use_red_circle', False)
        self.linear_start = kwargs.pop('linear_start', 0.00085)
        self.linear_end = kwargs.pop('linear_end', 0.012)
        self.cosine_s = kwargs.pop('cosine_s', 8e-3)
        self.num_timesteps = kwargs.pop('num_timesteps', 1000)
        
        super().__init__(*args, **kwargs)
        self.vjepa_hidden_dim = vjepa_hidden_dim
        self.num_output_features = num_output_features

        self.parameterization = "vjepa_direct"
        self.noise_strength = 0
        self.l1_loss = nn.L1Loss()
        self.bce_loss = nn.BCEWithLogitsLoss()

        # --- 复刻父类中基于 `collect_map_together` 的MLP初始化逻辑 ---

        # 1. 输入MLP列表，将探针向量投影到特征空间。该逻辑对所有模式通用。
        self.prob3_feat_mlps = nn.ModuleList([
            nn.Linear(self.prob3_dim, self.vjepa_hidden_dim) 
            for _ in range(self.num_output_features)
        ])
        
        # 2. 输出MLP，其结构由 `collect_map_together` 决定。
        vjepa_total_feat_dim = self.num_output_features * self.vjepa_hidden_dim
        if self.collect_map_together == 0:
            self.prob3_feat_output_mlps = nn.ModuleList([
                nn.Linear(self.vjepa_hidden_dim, self.prob3_dim)
                for _ in range(self.num_output_features)
            ])
        elif self.collect_map_together == 1:
            self.prob3_feat_output_mlps = nn.Linear(vjepa_total_feat_dim, self.prob3_dim)
        else: # self.collect_map_together == 2
            assert self.collect_map_together == 2
            # 输入维度是来自一对视频的所有聚合特征，因此是两倍
            self.prob3_feat_output_mlps = nn.Linear(vjepa_total_feat_dim * 2, 1)

        # 3. `self.prob3_output_projection` 由父类的 `__init__` 调用自动创建，
        #    它的逻辑被完全继承和复用，此处无需重复定义。

    def get_batch_input_for_vjepa(self, batch: dict) -> tuple:
        """
        一个为VJEPA模型定制的、精简的数据准备方法。
        """
        x = super().get_input(batch, self.first_stage_key)
        red_video = super().get_input(batch, 'mask')
        if self.use_red_circle:
            x = red_video
            print("use red circle video")
        if len(x.shape) == 6:
            x = x.reshape(x.shape[0] * x.shape[1], x.shape[2], x.shape[3], x.shape[4], x.shape[5])
        gt = batch['gt_ann']
        return x, gt
    
    def register_schedule(self, *args, **kwargs):
        """
        重写父类的 schedule 注册方法。
        由于VJEPA模型不使用扩散过程的采样器（sampler），因此我们不需要
        注册任何beta schedule。此方法什么都不做，以绕过父类中会引发
        NotImplementedError的逻辑。
        """
        print("PhysicsPredictionVJEPAv87: Bypassing DDPM schedule registration.")
        self.num_timesteps = kwargs.get('num_timesteps', 1000)
        pass

    def calculate_vjepa_physics_loss(self, feat_list: list, gt: dict, return_prediction: bool = False) -> tuple:
        """
        计算基于VJEPA特征的物理损失。

        Args:
            feat_list (list): 从VJEPA骨干网络得到的特征列表。
            gt (dict): 包含了真实标签的字典。
            return_prediction (bool, optional): 如果为True，则额外返回预测值。Defaults to False.

        Returns:
            tuple: 返回 (loss, loss_dict) 或 (loss, loss_dict, prediction)。
        """
        # Ensure all GT tensors are on the same device as prediction tensors
        device = feat_list[0].device
        for key in gt:
            if isinstance(gt[key], torch.Tensor):
                gt[key] = gt[key].to(device)
            elif isinstance(gt[key], list):
                gt[key] = [item.to(device) if isinstance(item, torch.Tensor) else item for item in gt[key]]

        bce_loss = nn.BCEWithLogitsLoss()
        
        # Process features
        prob3_outputs = []
        if self.collect_map_together in [0, 1]:
            for feat_i, feat in enumerate(feat_list):
                B, T, C, H, W = feat.shape
                query_vector = self.prob3_feat_mlps[feat_i](self.prob3_learnable_vector)
                query_vector = query_vector.expand(B, -1).unsqueeze(1)
                
                feat_reshaped = feat.permute(0, 1, 3, 4, 2).contiguous().view(B, T * H * W, C)
                
                scale = (C ** -0.5)
                attn_scores = torch.matmul(query_vector, feat_reshaped.transpose(-2, -1)) * scale
                attn_weights = torch.softmax(attn_scores, dim=-1)
                
                attended_feat = torch.matmul(attn_weights, feat_reshaped).squeeze(1)
                
                if self.collect_map_together == 0:
                    prob3_feat_output = self.prob3_feat_output_mlps[feat_i](attended_feat)
                    prob3_outputs.append(prob3_feat_output)
                else: # self.collect_map_together == 1
                    prob3_outputs.append(attended_feat)

        elif self.collect_map_together == 2:
            # Similar logic but without individual output MLPs yet
            for feat_i, feat in enumerate(feat_list):
                B, T, C, H, W = feat.shape
                query_vector = self.prob3_feat_mlps[feat_i](self.prob3_learnable_vector)
                query_vector = query_vector.expand(B, -1).unsqueeze(1)
                feat_reshaped = feat.permute(0, 1, 3, 4, 2).contiguous().view(B, T * H * W, C)
                scale = C ** -0.5
                attn_scores = torch.matmul(query_vector, feat_reshaped.transpose(-2, -1)) * scale
                attn_weights = torch.softmax(attn_scores, dim=-1)
                attended_feat = torch.matmul(attn_weights, feat_reshaped).squeeze(1)
                prob3_outputs.append(attended_feat)

        # Final projection
        gt_retitusion = None
        if prob3_outputs:
            if self.collect_map_together == 0:
                prob3_combined = torch.stack(prob3_outputs, dim=1)
                prob3_averaged = torch.mean(prob3_combined, dim=1)
                prob3_averaged = prob3_averaged.reshape(int(prob3_averaged.shape[0]/2), 2, -1)
                if not self.concat_2_feat:
                    prob3_diff = prob3_averaged[:, 0, :] - prob3_averaged[:, 1, :]
                else:
                    prob3_diff = torch.cat([prob3_averaged[:, 0, :], prob3_averaged[:, 1, :]], dim=1)
                gt_retitusion = self.prob3_output_projection(prob3_diff)
            
            elif self.collect_map_together == 1:
                prob3_combined = torch.cat(prob3_outputs, dim=1)
                prob3_averaged = self.prob3_feat_output_mlps(prob3_combined)
                prob3_averaged = prob3_averaged.reshape(int(prob3_averaged.shape[0]/2), 2, -1)
                if not self.concat_2_feat:
                    prob3_diff = prob3_averaged[:, 0, :] - prob3_averaged[:, 1, :]
                else:
                    prob3_diff = torch.cat([prob3_averaged[:, 0, :], prob3_averaged[:, 1, :]], dim=1)
                gt_retitusion = self.prob3_output_projection(prob3_diff)
            
            elif self.collect_map_together == 2:
                prob3_combined = torch.cat(prob3_outputs, dim=1)
                prob3_reshaped = prob3_combined.reshape(int(prob3_combined.shape[0]/2), 2 * prob3_combined.shape[1])
                gt_retitusion = self.prob3_feat_output_mlps(prob3_reshaped)

        loss_gt_retitusion = 0 if gt_retitusion is None else bce_loss(gt_retitusion[:,0], gt['gt_restitution'].float())
        
        # loss_gt_retitusion = 0 if gt_retitusion is None else bce_loss(gt_retitusion.squeeze(), gt['gt_restitution'].float())
        loss = loss_gt_retitusion
        
        loss_dict = {
            'total_loss': loss,
            'loss_gt_retitusion': loss_gt_retitusion,
        }
        
        # NaN and inf check
        for key_i in loss_dict.keys():
            if isinstance(loss_dict[key_i], torch.Tensor) and (torch.isnan(loss_dict[key_i]) or torch.isinf(loss_dict[key_i])):
                mainlogger.info(f"NaN or Inf loss detected in {key_i}, using fallback value")
                loss_dict[key_i] = torch.tensor(0.001, device=device, requires_grad=True)
        
        if return_prediction:
            return loss, loss_dict, gt_retitusion
            
        return loss, loss_dict

    def calculate_vjepa_physics_loss_train(self, feat_list: list, gt: dict) -> tuple[torch.Tensor, dict]:
        """
        使用从VJEPA获得的特征列表来计算物理预测损失, 并支持所有 `collect_map_together` 模式。
        """
        # 通用步骤: 从每一层提取聚合后的特征
        attended_features = []
        for i, feat in enumerate(feat_list):
            B, T, C, H, W = feat.shape
            query_vector = self.prob3_feat_mlps[i](self.prob3_learnable_vector)
            q = query_vector.expand(B, -1).unsqueeze(1)
            feat_reshaped = feat.permute(0, 1, 3, 4, 2).contiguous().view(B, T * H * W, C)
            scale = C ** -0.5
            attn_scores = torch.matmul(q, feat_reshaped.transpose(-2, -1)) * scale
            attn_weights = torch.softmax(attn_scores, dim=-1)
            attended_feat = torch.matmul(attn_weights, feat_reshaped).squeeze(1)
            attended_features.append(attended_feat)

        # 分支逻辑: 根据 `collect_map_together` 处理聚合后的特征
        gt_prediction = None
        gt_retitusion = None
        if attended_features:
            if self.collect_map_together == 0:
                prob3_outputs = [self.prob3_feat_output_mlps[i](att_feat) for i, att_feat in enumerate(attended_features)]
                prob3_averaged = torch.mean(torch.stack(prob3_outputs, dim=1), dim=1)
                
                real_B = prob3_averaged.shape[0] // 2
                prob3_averaged = prob3_averaged.reshape(real_B, 2, prob3_averaged.shape[1])
                prob3_diff = (prob3_averaged[:, 0, :] - prob3_averaged[:, 1, :]) if not self.concat_2_feat else torch.cat([prob3_averaged[:, 0, :], prob3_averaged[:, 1, :]], dim=1)
                gt_prediction = self.prob3_output_projection(prob3_diff)

            elif self.collect_map_together == 1:
                prob3_combined = torch.cat(attended_features, dim=1)
                prob3_averaged = self.prob3_feat_output_mlps(prob3_combined)

                real_B = prob3_averaged.shape[0] // 2
                prob3_averaged = prob3_averaged.reshape(real_B, 2, prob3_averaged.shape[1])
                prob3_diff = (prob3_averaged[:, 0, :] - prob3_averaged[:, 1, :]) if not self.concat_2_feat else torch.cat([prob3_averaged[:, 0, :], prob3_averaged[:, 1, :]], dim=1)
                gt_prediction = self.prob3_output_projection(prob3_diff)

            else: # self.collect_map_together == 2
                assert self.collect_map_together == 2
                prob3_combined = torch.cat(attended_features, dim=1)
                real_B = prob3_combined.shape[0] // 2
                prob3_combined = prob3_combined.reshape(real_B, 2 * prob3_combined.shape[1])
                gt_prediction = self.prob3_feat_output_mlps(prob3_combined)

        # 损失计算逻辑 (所有模式通用)
        gt_retitusion = gt_prediction
        bce_loss = nn.BCEWithLogitsLoss()
        # Ensure all GT tensors are on the same device as prediction tensors
        device = feat_list[0].device
        for key in gt:
            if isinstance(gt[key], torch.Tensor):
                gt[key] = gt[key].to(device)
            elif isinstance(gt[key], list):
                gt[key] = [item.to(device) if isinstance(item, torch.Tensor) else item for item in gt[key]]
        loss_gt_retitusion = 0 if gt_retitusion is None else bce_loss(gt_retitusion[:,0], gt['gt_restitution'].float())


        if gt_retitusion is not None:
            print(loss_gt_retitusion)
        


        
        # Combine all losses
        loss = loss_gt_retitusion
        
        loss_dict = {}
        loss_dict.update({
            'total_loss': loss,
            'loss_gt_retitusion': loss_gt_retitusion,
        })

        # # Add to p_losses method before returning loss
        # if torch.isnan(loss):
        #     # Use a small constant value to avoid training failure
        #     loss = torch.tensor(0.1, device=loss.device, requires_grad=True)
        #     print("NaN loss detected, using fallback value")

        for key_i in loss_dict.keys():
            if isinstance(loss_dict[key_i], torch.Tensor) and torch.isnan(loss_dict[key_i]):
                loss_dict[key_i] = torch.tensor(0.001, device=device, requires_grad=True)
                print("NaN loss detected, using fallback value")
    

        return loss, loss_dict 

        # device = feat_list[0].device if feat_list else self.device
        # for key in gt:
        #     if isinstance(gt[key], torch.Tensor): gt[key] = gt[key].to(device)
        
        # loss_gt = torch.tensor(0.0, device=device)
        # if gt_prediction is not None:
        #     loss_gt = bce_loss(gt_prediction.squeeze(-1), gt['gt_restitution'].float())

        # loss = loss_gt
        # loss_dict = {'total_loss': loss, 'loss_bce': loss_gt}
        # for key, value in loss_dict.items():
        #     if isinstance(value, torch.Tensor) and torch.isnan(value):
        #         print(f"NaN loss detected in {key}, replacing with 0.0.")
        #         loss_dict[key] = torch.tensor(0.0, device=device)
        
        # return loss, loss_dict

    def shared_step(self, batch: dict, **kwargs) -> tuple[torch.Tensor, dict]:
        """
        重写后的核心训练/验证步骤，使用精简的数据流和直接的模型调用。
        此方法现在返回损失张量和包含详细损失信息的字典。
        """
        x, gt = self.get_batch_input_for_vjepa(batch)
        
        t = torch.zeros(x.shape[0], device=self.device, dtype=torch.long)
        model_output, feat_list = self.model.diffusion_model(x, t=t, context=None)

        if feat_list is None or not feat_list:
            loss = torch.tensor(0.0, device=self.device, requires_grad=True)
            loss_dict = {'total_loss': loss.detach()}
        else:
            loss, loss_dict = self.calculate_vjepa_physics_loss(feat_list, gt)
        
        return loss, loss_dict

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        """
        标准的训练步骤，调用 shared_step 并记录返回的损失。
        """
        loss, loss_dict = self.shared_step(batch)

        prefix = 'train'
        log_dict_prefixed = {f'{prefix}/{k}': v for k, v in loss_dict.items()}
        self.log_dict(log_dict_prefixed, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        
        self.log("global_step", self.global_step,
                 prog_bar=True, logger=True, on_step=True, on_epoch=False)

        if (batch_idx+1) % self.log_every_t == 0:
            mainlogger.info(f"batch:{batch_idx}|epoch:{self.current_epoch} [globalstep:{self.global_step}]: loss={loss}")

        return loss

    def validation_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        return self.shared_step(batch)

    def p_losses(self, *args, **kwargs):
        raise NotImplementedError("p_losses is obsolete for this model.")

    def get_batch_input(self, *args, **kwargs):
        raise NotImplementedError("Use get_batch_input_for_vjepa for this model.")


class PhysicsPredictionVJEPAv87_attentive_probe(PhysicsPredictionVJEPAv87_original):
    """
    该模型将 LatentVisualDiffusionv87 的物理预测框架与 VJEPA 骨干网络相结合。
    它继承了v87处理成对输入（paired inputs）的数据准备逻辑，但用VJEPA替换了
    U-Net作为特征提取器，并相应地调整了训练和损失计算流程。

    此版本精确地复刻了父类中 `collect_map_together` 的所有模式。
    """
    def __init__(self, 
                 vjepa_hidden_dim: int, 
                 num_output_features: int = 1,
                 attentive_config: dict = None,
                 use_all_feat_in_attentive_probe: bool = False,
                 *args, **kwargs):
        """
        初始化AttentiveClassifier版本
        
        Args:
            vjepa_hidden_dim: VJEPA特征的隐藏维度 (对应D)
            num_output_features: 输出特征数量，默认为1（恢复系数预测）
            attentive_config: AttentiveClassifier的配置参数
        """
        # 调用父类初始化，但我们会替换掉MLP部分
        super().__init__(vjepa_hidden_dim, num_output_features, *args, **kwargs)

        self.vjepa_hidden_dim = vjepa_hidden_dim
        self.num_output_features = num_output_features
        self.use_all_feat_in_attentive_probe = use_all_feat_in_attentive_probe
        self.linear_start = kwargs.get('linear_start', 0.00085)
        self.linear_end = kwargs.get('linear_end', 0.012)
        self.cosine_s = kwargs.get('cosine_s', 8e-3)
        self.num_timesteps = kwargs.get('num_timesteps', 1000)

        self.parameterization = "vjepa_direct"
        self.noise_strength = 0
        self.l1_loss = nn.L1Loss()
        self.bce_loss = nn.BCEWithLogitsLoss()

        self.prob3_feat_mlps = nn.ModuleList([
            nn.Linear(self.vjepa_hidden_dim, self.vjepa_hidden_dim) 
            for _ in range(self.num_output_features)
        ]) 
        
        # 设置AttentiveClassifier的默认配置
        default_attentive_config = {
            'verb_classes': {},  # 不使用verb分类
            'noun_classes': {},  # 不使用noun分类
            'action_classes': {f'restitution_{i}': i for i in range(1)},  # 物理预测类别
            'embed_dim': vjepa_hidden_dim,  # 对应特征维度D
            'num_heads': 8,
            'depth': 4,
            'use_activation_checkpointing': False
        }
        
        if attentive_config:
            default_attentive_config.update(attentive_config)
        
        # 创建AttentiveClassifier来替换原有的MLP机制
        self.attentive_classifier = AttentiveClassifier(**default_attentive_config)

        # 根据concat_2_feat参数动态确定MLP输入维度
        mlp_input_dim = vjepa_hidden_dim * 2 if kwargs.get('concat_2_feat', False) else vjepa_hidden_dim
        self.final_projection_feature_mlp = nn.Linear(mlp_input_dim, 1)
        
        # 移除父类中的MLP组件，因为我们用AttentiveClassifier替换了
        # 保留prob3_learnable_vector用于兼容性（虽然AttentiveClassifier有自己的learnable queries）
        
        mainlogger.info(f"初始化PhysicsPredictionVJEPAv87_AttentiveClassifier: 使用AttentiveClassifier替换MLP机制")

    def calculate_vjepa_physics_loss(self, feat_list: list, gt: dict, return_prediction: bool = False) -> tuple:
        """
        使用AttentiveClassifier计算基于VJEPA特征的物理损失
        
        Args:
            feat_list: 现在是最后一层feature，维度为 [B, N, D]
            gt: 包含真实标签的字典
            return_prediction: 是否返回预测值
            
        Returns:
            tuple: (loss, loss_dict) 或 (loss, loss_dict, prediction)
        """

        if feat_list is None:
            device = self.device
        else: # feat_list现在是model_output，直接使用
            device = feat_list.device

        for key in gt:
            if isinstance(gt[key], torch.Tensor):
                gt[key] = gt[key].to(device)
            elif isinstance(gt[key], list):
                gt[key] = [item.to(device) if isinstance(item, torch.Tensor) else item for item in gt[key]]

        bce_loss = nn.BCEWithLogitsLoss()
        
        # 处理特征
        if feat_list is None:
            gt_retitusion = None
        else:
            # 获取最后一层特征，维度应该是 [B, N, D]
            last_layer_feat = feat_list
            
            # 直接使用AttentiveClassifier处理特征
            # last_layer_feat的维度 [B, N, D] 正好符合AttentiveClassifier的输入要求
            classifier_output = self.attentive_classifier(last_layer_feat)
            
            # 获取action分类结果（用于物理预测）
            action_logits = classifier_output['skip_projection_feature']  # [B, num_classes]

            # print('action_logits of skip_projection_feature: ', action_logits.shape)
            
            # 处理成对数据（假设批次大小是偶数，用于比较两个状态）
            B = action_logits.shape[0]
            if B % 2 == 0:
                # 重塑为成对格式
                real_batch_size = B // 2
                action_reshaped = action_logits.reshape(real_batch_size, 2, -1)
                
                # 根据concat_2_feat决定如何处理成对特征
                if hasattr(self, 'concat_2_feat') and self.concat_2_feat:
                    # 拼接两个状态的特征
                    gt_retitusion = torch.cat([action_reshaped[:, 0, :], action_reshaped[:, 1, :]], dim=1)
                else:
                    # 计算两个状态的差值
                    gt_retitusion = action_reshaped[:, 0, :] - action_reshaped[:, 1, :]
            else:
                # 如果不是成对数据，直接使用
                gt_retitusion = action_logits
        # import pdb; pdb.set_trace()
        gt_retitusion = self.final_projection_feature_mlp(gt_retitusion)

        loss_gt_retitusion = 0 if gt_retitusion is None else bce_loss(gt_retitusion[:,0], gt['gt_restitution'].float())
        
        if self.training:
            print('gt_retitusion: ', gt_retitusion)
            print('gt: ', gt['gt_restitution'])
        
        loss = loss_gt_retitusion
        
        loss_dict = {
            'total_loss': loss,
            'loss_gt_retitusion': loss_gt_retitusion,
        }
        
        # NaN和Inf检查
        for key_i in loss_dict.keys():
            if isinstance(loss_dict[key_i], torch.Tensor) and (torch.isnan(loss_dict[key_i]) or torch.isinf(loss_dict[key_i])):
                mainlogger.info(f"NaN or Inf loss detected in {key_i}, using fallback value")
                loss_dict[key_i] = torch.tensor(0.001, device=device, requires_grad=True)
        
        if return_prediction:
            return loss, loss_dict, gt_retitusion
            
        return loss, loss_dict

    def shared_step(self, batch: dict, **kwargs) -> tuple[torch.Tensor, dict]:
        """
        重写后的核心训练/验证步骤，使用精简的数据流和直接的模型调用。
        此方法现在返回损失张量和包含详细损失信息的字典。
        """
        x, gt = self.get_batch_input_for_vjepa(batch)
        
        t = torch.zeros(x.shape[0], device=self.device, dtype=torch.long)
        model_output, feat_list = self.model.diffusion_model(x, t=t, context=None)

        if self.use_all_feat_in_attentive_probe:
            # 将feat_list中的所有特征重塑为[B, N_patches, D]，通过线性变换后再平均
            if feat_list and len(feat_list) > 0:
                # 获取model_output的形状作为目标形状
                target_shape = model_output.shape  # [B, N_patches, D]
                B, N_patches, D = target_shape
                
                # 将每个feat重塑为[B, N_patches, D]，并通过对应的线性变换
                transformed_feats = []
                for i, feat in enumerate(feat_list):
                    # feat shape: [B, T, C, H, W] -> [B, N_patches, D]
                    B_feat, T, C, H, W = feat.shape
                    # 重塑为[B, T*H*W, C]，相当于[B, N_patches, D]
                    feat_reshaped = feat.permute(0, 2, 1, 3, 4).contiguous().view(B_feat, C, T*H*W).transpose(1, 2)
                    # 现在feat_reshaped的形状是[B, T*H*W, C]，即[B, N_patches, D]
                    
                    # 通过对应的线性变换层
                    feat_idx = min(i, len(self.prob3_feat_mlps) - 1)  # 防止索引越界
                    feat_transformed = self.prob3_feat_mlps[feat_idx](feat_reshaped)
                    transformed_feats.append(feat_transformed)
                
                # 对所有经过线性变换的特征进行平均
                model_output = torch.stack(transformed_feats, dim=0).mean(dim=0)  # [B, N_patches, D]
            # 否则使用原始的model_output
        else:
            model_output = model_output

        if feat_list is None or not feat_list:
            loss = torch.tensor(0.0, device=self.device, requires_grad=True)
            loss_dict = {'total_loss': loss.detach()}
        else:
            loss, loss_dict = self.calculate_vjepa_physics_loss(model_output, gt)
        
        return loss, loss_dict

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        """
        标准的训练步骤，调用shared_step并记录返回的损失
        """
        loss, loss_dict = self.shared_step(batch)
        
        # 在这里检查总损失
        if not torch.isfinite(loss):
            mainlogger.warning(f"Non-finite loss detected at batch {batch_idx}: {loss}")
            # 返回一个小的有效损失值
            loss = torch.tensor(0.001, device=loss.device, requires_grad=True)
        
        prefix = 'train'
        log_dict_prefixed = {f'{prefix}/{k}': v for k, v in loss_dict.items()}
        
        # 记录损失
        if (batch_idx+1) % self.log_every_t == 0:
            mainlogger.info(f"batch:{batch_idx}|epoch:{self.current_epoch} [globalstep:{self.global_step}]: loss={loss}")
        
        self.log_dict(log_dict_prefixed, prog_bar=True, logger=True, on_step=True, on_epoch=False)
        
        return loss


class PhysicsPredictionVJEPAv87Test_original(PhysicsPredictionVJEPAv87_original):
    """
    专门用于评估 `PhysicsPredictionVJEPAv87_original` 模型性能的测试类。

    该类继承了 `PhysicsPredictionVJEPAv87_original` 的核心VJEPA预测框架，
    并通过重写验证步骤 `validation_step` 来增加结果保存功能。
    它遵循了标准的测试类设计模式：在初始化时指定结果保存目录，
    在评估时运行预测并将预测值、真实值与样本文件名一一对应地
    保存为JSON文件，以便于进行离线的定量分析。
    """
    def __init__(self, *args, save_result_dir=None, **kwargs):
        """
        初始化测试类。

        Args:
            save_result_dir (str, optional): 保存测试结果的目录路径。
            *args, **kwargs: 传递给父类的参数。
        """
        super().__init__(*args, **kwargs)
        self.save_result_dir = save_result_dir
        if self.save_result_dir is not None:
            os.makedirs(self.save_result_dir, exist_ok=True)
            mainlogger.info(f"Test results for PhysicsPredictionVJEPAv87Test_original will be saved to {self.save_result_dir}")

    def get_batch_input_for_vjepa(self, batch: dict) -> tuple:
        """
        重写父类的数据准备方法，额外返回文件名列表 `fns` 用于结果保存。
        """
        # 复用父类的核心逻辑来获取 x 和 gt
        x, gt = super().get_batch_input_for_vjepa(batch)
        
        # 新增逻辑：提取并返回文件名
        # The path in batch is a list of paired paths, e.g., ['a1.mp4', 'a2.mp4', 'b1.mp4', 'b2.mp4']
        fns = batch['path']
        
        return x, gt, fns

    def run_prediction_step(self, batch: dict) -> tuple:
        """
        运行单步预测，并返回包括预测值和文件名在内的所有相关信息。
        """
        x, gt, fns = self.get_batch_input_for_vjepa(batch)
        
        t = torch.zeros(x.shape[0], device=self.device, dtype=torch.long)
        _, feat_list = self.model.diffusion_model(x, timesteps=t, context=None)

        if feat_list is None or not feat_list:
            loss = torch.tensor(0.0, device=self.device, requires_grad=True)
            loss_dict = {'total_loss': loss.detach()}
            predictions = None
        else:
            loss, loss_dict, predictions = self.calculate_vjepa_physics_loss(feat_list, gt, return_prediction=True)
        
        return loss, loss_dict, predictions, gt, fns

    @torch.no_grad()
    def validation_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        """
        在验证集上执行一个步骤，记录损失并保存详细的预测结果。
        此方法的保存逻辑严格遵循 LatentVisualDiffusionv87Test 中的标准。
        """
        loss, loss_dict, predictions, ground_truth, fns = self.run_prediction_step(batch)

        # 记录损失到logger
        prefix = 'val'
        num_pairs = len(fns) // 2 if fns else 0
        self.log_dict(loss_dict, prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True, batch_size=num_pairs)

        # 保存预测结果到JSON文件，严格遵循 LatentVisualDiffusionv87Test 的逻辑
        if self.save_result_dir and predictions is not None and fns:
            fn_2_pred = {}
            # 遍历批次中的所有样本
            for i in range(predictions.shape[0]):
                fn_2_pred[fns[i]] = {}
                # `predictions` is the raw logit tensor. The reference saves this raw value.
                fn_2_pred[fns[i]]['pred'] = predictions[i][0].detach().cpu().numpy().tolist()
                fn_2_pred[fns[i]]['gt'] = ground_truth['gt_restitution'][i].detach().cpu().numpy().tolist()

            # 将整个批次的结果保存到一个JSON文件，以批次中第一个文件名命名
            save_path = os.path.join(self.save_result_dir, fns[0] + '.json')
            with open(save_path, 'w') as dump_f:
                json.dump(fn_2_pred, dump_f)

        print('save_result_dir: ', self.save_result_dir)

        return loss


class PhysicsPredictionVJEPAv87Test_attentive_probe(PhysicsPredictionVJEPAv87_attentive_probe):
    """
    专门用于评估 `PhysicsPredictionVJEPAv87_attentive_probe` 模型性能的测试类。

    该类继承了 `PhysicsPredictionVJEPAv87_attentive_probe` 的核心VJEPA预测框架，
    并通过重写验证步骤 `validation_step` 来增加结果保存功能。
    它遵循了标准的测试类设计模式：在初始化时指定结果保存目录，
    在评估时运行预测并将预测值、真实值与样本文件名一一对应地
    保存为JSON文件，以便于进行离线的定量分析。
    """
    def __init__(self, *args, save_result_dir=None, **kwargs):
        """
        初始化测试类。

        Args:
            save_result_dir (str, optional): 保存测试结果的目录路径。
            *args, **kwargs: 传递给父类的参数。
        """
        super().__init__(*args, **kwargs)
        self.save_result_dir = save_result_dir
        if self.save_result_dir is not None:
            os.makedirs(self.save_result_dir, exist_ok=True)
            mainlogger.info(f"Test results for PhysicsPredictionVJEPAv87Test_original will be saved to {self.save_result_dir}")

    def get_batch_input_for_vjepa(self, batch: dict) -> tuple:
        """
        重写父类的数据准备方法，额外返回文件名列表 `fns` 用于结果保存。
        """
        # 复用父类的核心逻辑来获取 x 和 gt
        x, gt = super().get_batch_input_for_vjepa(batch)
        
        # 新增逻辑：提取并返回文件名
        # The path in batch is a list of paired paths, e.g., ['a1.mp4', 'a2.mp4', 'b1.mp4', 'b2.mp4']
        fns = batch['path']
        
        return x, gt, fns

    def run_prediction_step(self, batch: dict) -> tuple:
        """
        运行单步预测，并返回包括预测值和文件名在内的所有相关信息。
        """
        x, gt, fns = self.get_batch_input_for_vjepa(batch)
        
        t = torch.zeros(x.shape[0], device=self.device, dtype=torch.long)
        model_output, feat_list = self.model.diffusion_model(x, timesteps=t, context=None)

        if self.use_all_feat_in_attentive_probe:
            # 将所有层的特征重塑为 [B, N_patches, D]，通过线性变换后再平均
            if feat_list and isinstance(feat_list, list):
                transformed_features = []
                for i, feat in enumerate(feat_list):
                    # 重塑为 [B, N_patches, D]
                    B, T, C, H, W = feat.shape
                    feat_reshaped = feat.permute(0, 1, 3, 4, 2).contiguous().view(B, T * H * W, C)
                    
                    # 通过对应的线性变换层
                    feat_idx = min(i, len(self.prob3_feat_mlps) - 1)  # 防止索引越界
                    feat_transformed = self.prob3_feat_mlps[feat_idx](feat_reshaped)
                    transformed_features.append(feat_transformed)
                
                # 平均所有经过线性变换的特征
                model_output = torch.stack(transformed_features, dim=0).mean(dim=0)  # [B, N_patches, D]
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
        
        return loss, loss_dict, predictions, gt, fns

    @torch.no_grad()
    def validation_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        """
        在验证集上执行一个步骤，记录损失并保存详细的预测结果。
        此方法的保存逻辑严格遵循 LatentVisualDiffusionv87Test 中的标准。
        """
        loss, loss_dict, predictions, ground_truth, fns = self.run_prediction_step(batch)

        # 记录损失到logger
        prefix = 'val'
        num_pairs = len(fns) // 2 if fns else 0
        self.log_dict(loss_dict, prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True, batch_size=num_pairs)

        # 保存预测结果到JSON文件，严格遵循 LatentVisualDiffusionv87Test 的逻辑
        if self.save_result_dir and predictions is not None and fns:
            fn_2_pred = {}
            # 遍历批次中的所有样本
            for i in range(predictions.shape[0]):
                fn_2_pred[fns[i]] = {}
                # `predictions` is the raw logit tensor. The reference saves this raw value.
                fn_2_pred[fns[i]]['pred'] = predictions[i][0].detach().cpu().numpy().tolist()
                fn_2_pred[fns[i]]['gt'] = ground_truth['gt_restitution'][i].detach().cpu().numpy().tolist()

            # 将整个批次的结果保存到一个JSON文件，以批次中第一个文件名命名
            save_path = os.path.join(self.save_result_dir, fns[0] + '.json')
            with open(save_path, 'w') as dump_f:
                json.dump(fn_2_pred, dump_f)

        print('save_result_dir: ', self.save_result_dir)
        return loss



class PhysicsPredictionVJEPAv87_v1(LatentVisualDiffusionv87):
    """
    使用VJEPA进行物理预测的模型。
    这个版本经过重构，使其数据输入方法对训练和测试都更加友好和健壮。

    主要改动:
    - `__init__`: 手动添加了 linear_start 等属性，以确保与外部训练脚本的兼容性。
    - `get_batch_input_for_vjepa` 被增强，可以通过 `return_full_batch` 参数
      来控制返回内容的详细程度，为测试子类提供了必要的数据接口。
    - `shared_step` 被更新，以使用新版 `get_batch_input_for_vjepa` 的训练模式。
    """
    def __init__(self, vjepa_hidden_dim: int, num_output_features: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # 为兼容 trainer 脚本，手动设置这些属性
        self.linear_start = kwargs.get('linear_start', 0.00085)
        self.linear_end = kwargs.get('linear_end', 0.012)
        self.cosine_s = kwargs.get('cosine_s', 8e-3)
        
        mainlogger.info(f"Initializing PhysicsPredictionVJEPAv87: Setting up VJEPA-specific components.")
        self.vjepa_predictor = self.model.diffusion_model
        
        uniform_feat_dim = vjepa_hidden_dim

        self.prob3_feat_mlps = nn.ModuleList([
            nn.Linear(self.prob3_dim, uniform_feat_dim) for _ in range(num_output_features)
        ])
        
        if self.collect_map_together in [0, 1]:
            if self.collect_map_together == 0:
                 self.prob3_feat_output_mlps = nn.ModuleList([
                    nn.Linear(uniform_feat_dim, self.prob3_dim) for _ in range(num_output_features)
                ])
            else: # collect_map_together == 1
                self.prob3_feat_output_mlps = nn.Linear(uniform_feat_dim * num_output_features, self.prob3_dim)
            
            output_proj_dim = self.prob3_dim * 2 if self.concat_2_feat else self.prob3_dim
            self.prob3_output_projection = nn.Linear(output_proj_dim, 1)

        elif self.collect_map_together == 2:
            self.prob3_feat_output_mlps = nn.Linear(uniform_feat_dim * num_output_features * 2, 1)
        
        self.prob3_learnable_vector = nn.Parameter(torch.randn(1, self.prob3_dim))

    def get_batch_input_for_vjepa(self, batch: dict, return_full_batch: bool = False) -> tuple:
        """
        为 VJEPA 模型准备批处理输入。
        """
        _, _, _, cond, gt, mask, _, keytime_mask, fns = \
            super().get_batch_input(batch, random_uncond=False)
        
        video_input = batch['video_input']
        
        if return_full_batch:
            return video_input, cond, gt, mask, keytime_mask, fns
        else:
            return video_input, gt

    def register_schedule(self, *args, **kwargs):
        """
        重写父类的 schedule 注册方法。VJEPA模型不使用扩散过程。
        """
        mainlogger.info("PhysicsPredictionVJEPAv87: Bypassing DDPM schedule registration.")
        self.num_timesteps = kwargs.get('num_timesteps', 1000)
        pass

    def calculate_vjepa_physics_loss(self, feat_list: list, gt: dict, return_prediction: bool = False) -> tuple:
        """
        计算基于VJEPA特征的物理损失。
        """
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
            scale = C ** -0.5
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
                prob3_diff = (prob3_averaged[:, 0, :] - prob3_averaged[:, 1, :]) if not self.concat_2_feat else torch.cat([prob3_averaged[:, 0, :], prob3_averaged[:, 1, :]], dim=1)
                gt_prediction = self.prob3_output_projection(prob3_diff)
            else: # self.collect_map_together == 2
                prob3_combined = torch.cat(attended_features, dim=1)
                real_B = prob3_combined.shape[0] // 2
                prob3_combined = prob3_combined.reshape(real_B, 2 * prob3_combined.shape[1])
                gt_prediction = self.prob3_feat_output_mlps(prob3_combined)

        loss_val = torch.tensor(0.0, device=device)
        if gt_prediction is not None:
            loss_val = bce_loss(gt_prediction.squeeze(), gt['gt_restitution'].float())

        loss_dict = {'total_loss': loss_val.item(), 'loss_restitution': loss_val.item()}
        
        if torch.isnan(loss_val) or torch.isinf(loss_val):
            mainlogger.warning(f"NaN or Inf loss detected, using fallback value 0.001")
            loss_val = torch.tensor(0.001, device=device, requires_grad=True)

        if return_prediction:
            return loss_val, loss_dict, gt_prediction
        return loss_val, loss_dict

    def calculate_vjepa_physics_loss_train(self, feat_list: list, gt: dict) -> tuple[torch.Tensor, dict]:
        """
        一个为训练优化的损失计算版本。
        """
        loss, loss_dict, _ = self.calculate_vjepa_physics_loss(feat_list, gt, return_prediction=True)
        return loss, loss_dict

    def shared_step(self, batch: dict, **kwargs) -> tuple[torch.Tensor, dict]:
        """
        标准的共享步骤，为训练流程获取输入并进行前向传播。
        """
        video_input, gt = self.get_batch_input_for_vjepa(batch, return_full_batch=False)
        feat_list = self.vjepa_predictor(video_input)
        loss, loss_dict = self.calculate_vjepa_physics_loss_train(feat_list, gt)
        return loss, loss_dict

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        """
        训练循环中的单步操作。
        """
        loss, loss_dict = self.shared_step(batch)
        self.log("train_loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log_dict(loss_dict, prog_bar=False, logger=True, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        """
        验证循环中的单步操作。
        """
        with torch.no_grad():
            loss, loss_dict = self.shared_step(batch)
        self.log("val_loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log_dict(loss_dict, prog_bar=False, logger=True, on_step=True, on_epoch=True)
        return loss

    def p_losses(self, *args, **kwargs):
        raise NotImplementedError("p_losses is obsolete for this model.")

class PhysicsPredictionVJEPAv87(LatentVisualDiffusionv87):
    """
    使用VJEPA进行物理预测的模型，此版本修正为正确地从 LatentVisualDiffusionv87 继承。

    此版本遵循了必须从 `LatentVisualDiffusionv87` 继承的要求，并通过完全重写数据输入逻辑来解决由此产生的冲突。

    主要修正:
    - 继承关系: 恢复为继承自 `LatentVisualDiffusionv87`，确保框架兼容性。
    - `get_batch_input_for_vjepa`: 实现了一个完全独立的版本，直接从 `batch` 字典中提取所有需要的数据
      （视频、真值、掩码、文件名），彻底绕过了父类中导致 `KeyError: 'image'` 的 `get_batch_input` 方法。
      这是解决核心矛盾的关键。
    - `run_prediction_step` 和 `shared_step`: 调整为在调用 `vjepa_predictor` 时，
      正确地传递了所需的 `timesteps` 参数。
    - `register_schedule`: 仍然被重写，以确保框架兼容性所需的所有属性 (`linear_start` 等)
      都被设置，同时避免创建不必要的扩散调度器。
    """
    def __init__(
        self,
        vjepa_cfg,
        vjepa_predictor_cfg,
        vjepa_hidden_dim,
        *args, **kwargs
    ):
        # 调用父类构造函数，以正确初始化所有框架所需的功能
        super().__init__(*args, **kwargs)

        # 初始化 VJEPA 编码器和预测器
        self.vjepa = instantiate_from_config(vjepa_cfg)
        self.vjepa_predictor = instantiate_from_config(vjepa_predictor_cfg)
        
        # 用于从 VJEPA 特征预测物理参数的线性层
        self.physics_head = torch.nn.Linear(vjepa_hidden_dim, 1)

    def register_schedule(self, beta_schedule="linear", timesteps=1000,
                          linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
        """
        重写父类方法以避免创建不必要的扩散调度器，
        但仍然设置必要的属性以确保框架兼容性。
        """
        self.num_timesteps = timesteps
        self.linear_start = linear_start
        self.linear_end = linear_end

    @torch.no_grad()
    def get_batch_input_for_vjepa(self, batch: dict, return_full_batch: bool = False) -> tuple:
        """
        一个完全独立的、用于准备批处理输入的方法。
        此方法不调用父类的 get_batch_input，以避免 `KeyError: 'image'`。
        """
        # self.get_input 是在更上层的 DDPM 中定义的，可以安全使用
        video_input = self.get_input(batch, self.first_stage_key)
        cond = None  # 本模型没有条件输入

        if return_full_batch:
            gt = batch['gt_ann']
            mask = batch['mask']
            keytime_mask = batch['keytime_mask']
            if 'rel_path' in batch:
                fns = [os.path.basename(p) for p in batch['rel_path']]
            else:
                fns = [f'sample_{i}' for i in range(video_input.shape[0])]
            return video_input, cond, gt, mask, keytime_mask, fns
        
        return video_input, cond

    def predict_physics_from_vjepa(self, feat_list: list, mask: torch.Tensor) -> dict:
        """
        从VJEPA特征预测物理属性。
        """
        vjepa_feat = feat_list[0]
        restitution = self.physics_head(vjepa_feat).squeeze(-1)
        return {'restitution': restitution}

    def shared_step(self, batch: dict) -> tuple:
        """
        训练步骤，计算 VJEPA 物理预测的损失。
        """
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
        return loss, loss_dict

    def run_prediction_step(self, batch: dict) -> tuple:
        """
        在验证/测试集上运行单步预测。
        """
        video_input, cond, gt, mask, keytime_mask, fns = self.get_batch_input_for_vjepa(
            batch, return_full_batch=True
        )

        batch_size = video_input.shape[0]
        timesteps = torch.zeros(batch_size, device=self.device).long()

        feat_list = self.vjepa_predictor(video_input, timesteps)
        predictions = self.predict_physics_from_vjepa(feat_list, mask)
        ground_truth = {'restitution': gt}

        loss_func = torch.nn.L1Loss()
        loss = loss_func(predictions['restitution'], ground_truth['restitution'])
        loss_dict = {'val/l1_loss': loss.item()}

        return loss, loss_dict, predictions, ground_truth, fns

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

    def forward(self, x, t, c_concat: list = None, c_crossattn: list = None,
                c_adm=None, s=None, mask=None, **kwargs):
        # temporal_context = fps is foNone
        if self.conditioning_key is None:
            out = self.diffusion_model(x, t)
        elif self.conditioning_key == 'concat':
            xc = torch.cat([x] + c_concat, dim=1)
            out = self.diffusion_model(xc, t, **kwargs)
        elif self.conditioning_key == 'crossattn':
            cc = torch.cat(c_crossattn, 1)
            out = self.diffusion_model(x, t, context=cc, **kwargs)
        elif self.conditioning_key == 'hybrid':
            ## it is just right [b,c,t,h,w]: concatenate in channel dim
            xc = torch.cat([x] + c_concat, dim=1)
            cc = torch.cat(c_crossattn, 1)
            # ipdb.set_trace()
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
            # assert s is not None
            xc = torch.cat([x] + c_concat, dim=1)
            out = self.diffusion_model(xc, t, context=None, s=s, mask=mask)
        elif self.conditioning_key == 'concat-adm-mask':
            # assert s is not None
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
        elif self.conditioning_key == 'hybrid-time-adm': # adm means y, e.g., class index
            # assert s is not None
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