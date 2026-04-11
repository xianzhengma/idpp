"""
LatentVisualDiffusion模型的LoRA版本
该模型支持使用LoRA微调UNet
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from contextlib import contextmanager
from functools import partial
import os
from lvdm.models.ddpm3d import LatentVisualDiffusionv14
from lvdm.modules.lora_modules import (
    replace_unet_modules_with_lora, 
    get_lora_parameters, 
    get_pdecoder_parameters, 
    save_trained_parameters, 
    load_trained_parameters
)
from lvdm.common import (
    extract_into_tensor,
    noise_like,
    exists,
    default
)
from einops import rearrange, repeat
import random
from lvdm.models.p_decoder import PDecoderSupElasticity, PDecoderSupElasticityv1, PDecoder, PDecoderCondMask
import ipdb
import json



# v0: multiply mask on feat + no extra supervision
class LatentVisualDiffusionv14Lora(LatentVisualDiffusionv14):
    """
    对LatentVisualDiffusionv14模型添加LoRA支持
    
    该模型支持：
    1. 使用LoRA微调UNet
    2. 同时训练PDecoder
    3. 保存和加载训练参数
    """
    def __init__(self, 
                 *args, 
                 use_lora=True, 
                 lora_config=None,
                 trained_params_path=None,  # 新增：训练参数路径
                 save_only_trained_params=True,  # 新增：是否只保存训练参数
                 **kwargs):
        """
        初始化LoRA版本的LatentVisualDiffusion模型
        
        参数:
            use_lora (bool): 是否使用LoRA
            lora_config (dict): LoRA配置，包括rank, alpha, dropout和target_modules
            trained_params_path (str): 训练参数路径，用于加载预训练的参数
            save_only_trained_params (bool): 是否只保存训练参数
        """
        super().__init__(*args, **kwargs)
        
        # 输出基本配置信息
        print(f"LoRA配置: 使用={use_lora}, 参数路径={trained_params_path}")
        
        self.use_lora = use_lora
        self.save_only_trained_params = save_only_trained_params
        self.lora_config = lora_config or {
            "rank": 16,
            "alpha": 32,
            "dropout": 0.1,
            "target_modules": [],
            "exact_matches": [],  # 新增: 精确路径匹配列表
            "train_pdecoder": True  # 是否同时训练PDecoder
        }
        
        # 重新设置requires_grad
        self._freeze_all()  # 先冻结所有参数
        
        # 应用LoRA
        if self.use_lora:
            self._apply_lora()
            
            # 加载训练参数
            if trained_params_path is not None:
                if os.path.exists(trained_params_path):
                    print(f"加载训练参数: {trained_params_path}")
                    self._load_trained_params(trained_params_path)
                else:
                    print(f"警告: 训练参数文件不存在: {trained_params_path}")
            else:
                print("未指定训练参数路径")
        
        # 如果需要训练PDecoder
        if self.lora_config.get("train_pdecoder", True):
            self._enable_pdecoder_training()
            
    def _apply_lora(self):
        """
        将UNet模型中的目标模块替换为LoRA版本
        """
        # 获取LoRA配置
        rank = self.lora_config.get("rank", 16)
        alpha = self.lora_config.get("alpha", 32)
        dropout = self.lora_config.get("dropout", 0.1)
        target_modules = self.lora_config.get("target_modules", None)
        exact_matches = self.lora_config.get("exact_matches", [])  # 新增: 获取精确匹配列表
        
        # 应用LoRA到UNet
        self.model.diffusion_model = replace_unet_modules_with_lora(
            self.model.diffusion_model, 
            rank=rank, 
            alpha=alpha, 
            dropout=dropout,
            target_replace_module=target_modules,
            exact_matches=exact_matches  # 新增: 传递精确匹配列表
        )
    
    def _load_trained_params(self, checkpoint_path):
        """
        加载训练参数
        
        参数:
            checkpoint_path (str): 训练参数路径
        """
        print(f"正在加载训练参数: {checkpoint_path}")
        # 显示当前PDecoder加载设置
        train_pdecoder = self.lora_config.get("train_pdecoder", True)
        print(f"PDecoder加载设置: train_pdecoder={train_pdecoder}")
        
        # 加载LoRA和PDecoder参数
        load_trained_parameters(
            model=self, 
            path=checkpoint_path, 
            load_lora=True, 
            load_pdecoder=train_pdecoder
        )
        print("训练参数加载完成!")
    
    def _enable_pdecoder_training(self):
        """
        启用PDecoder的训练
        """
        print("启用PDecoder训练...")
        # 遍历所有参数，启用PDecoder的梯度
        for name, param in self.named_parameters():
            if "p_decoder" in name:
                param.requires_grad = True
                
        print(f"PDecoder参数已启用训练.")
    
    def validation_step(self, batch, batch_idx):
        """
        验证步骤，添加random_uncond参数
        """
        # 在验证时使用False作为random_uncond参数
        _, loss_dict_no_ema = self.shared_step(batch, random_uncond=False)
        
        self.log_dict(
            {f"val/{key}": val for key, val in loss_dict_no_ema.items()},
            prog_bar=True,
            logger=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        return loss_dict_no_ema
        
    def configure_optimizers(self):
        """
        配置优化器，为LoRA参数和PDecoder参数分别设置不同的学习率
        """
        lr = self.learning_rate
        
        # 获取所有LoRA参数
        lora_params = get_lora_parameters(self.model.diffusion_model)
        
        # 获取所有PDecoder参数
        pdecoder_params = []
        if self.lora_config.get("train_pdecoder", True):
            pdecoder_params = get_pdecoder_parameters(self)
        
        # 设置不同的参数组
        param_groups = [
            {"params": lora_params, "lr": lr * 5.0},  # LoRA参数使用更大的学习率
        ]
        
        if pdecoder_params:
            param_groups.append({"params": pdecoder_params, "lr": lr})
        
        # 创建优化器
        optimizer = torch.optim.AdamW(param_groups)
        
        if self.use_scheduler:
            lr_scheduler = self.configure_schedulers(optimizer)
            return [optimizer], [lr_scheduler]
        return optimizer
    
    def on_save_checkpoint(self, checkpoint):
        """
        保存检查点时，保存训练参数
        
        如果save_only_trained_params=True，则仅保存训练参数
        否则，同时保存完整检查点和训练参数
        """
        if not self.save_only_trained_params:
            # 使用默认的保存机制保存完整检查点
            super().on_save_checkpoint(checkpoint)
        
        # 将LoRA配置保存到检查点
        checkpoint["lora_config"] = self.lora_config
        
        # 创建训练配置
        training_config = {
            "lora_config": self.lora_config,
            "learning_rate": self.learning_rate,
            "global_step": self.global_step,
            "current_epoch": self.current_epoch
        }
        
        # 创建一个单独的训练参数保存路径
        try:
            # 尝试获取当前checkpoint callback的dirpath
            if hasattr(self.trainer, "checkpoint_callback") and hasattr(self.trainer.checkpoint_callback, "dirpath"):
                dirpath = self.trainer.checkpoint_callback.dirpath
                global_step = self.global_step
                
                # 如果只保存训练参数，则完全替换检查点
                if self.save_only_trained_params:
                    filename = f"{dirpath}/trained_params_step_{global_step}"
                    full_path = save_trained_parameters(self, filename, training_config)
                    # 用训练参数文件路径替换检查点文件路径
                    checkpoint["_target_module_file"] = full_path
                    checkpoint["state_dict"] = {}  # 清空state_dict
                else:
                    # 同时保存训练参数和完整检查点
                    filename = f"{dirpath}/trained_params_step_{global_step}"
                    save_trained_parameters(self, filename, training_config)
        except Exception as e:
            print(f"保存训练参数错误: {e}")
    
    def on_train_epoch_end(self):
        """
        在每个训练epoch结束时保存训练参数
        """
        super().on_train_epoch_end()
        
        # 创建训练配置
        training_config = {
            "lora_config": self.lora_config,
            "learning_rate": self.learning_rate,
            "current_epoch": self.current_epoch
        }
        
        # 创建一个单独的训练参数保存路径
        try:
            # 尝试获取当前checkpoint callback的dirpath
            if hasattr(self.trainer, "checkpoint_callback") and hasattr(self.trainer.checkpoint_callback, "dirpath"):
                dirpath = self.trainer.checkpoint_callback.dirpath
                epoch = self.current_epoch
                filename = f"{dirpath}/trained_params_epoch_{epoch}"
                
                # 保存训练参数
                save_trained_parameters(self, filename, training_config)
        except Exception as e:
            print(f"保存训练参数错误: {e}") 


    def forward(self, x, c, gt, mask, **kwargs):
        # t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=self.device).long()
        t = torch.randint(50, 51, (x.shape[0],), device=self.device).long()
        if self.use_dynamic_rescale:
            x = x * extract_into_tensor(self.scale_arr, t, x.shape)
        return self.p_losses(x, c, t, gt, mask, **kwargs)


class LatentVisualDiffusionv14LoraTest1(LatentVisualDiffusionv14Lora):
    """
    对LatentVisualDiffusionv14模型添加LoRA支持
    
    该模型支持：
    1. 使用LoRA微调UNet
    2. 同时训练PDecoder
    3. 保存和加载LoRA权重
    """
    def __init__(self, 
                 *args, 
                 save_result_dir=None,
                 use_lora=True, 
                 lora_config=None,
                 lora_checkpoint=None,  # 新增：LoRA权重路径
                 **kwargs):
        """
        初始化LoRA版本的LatentVisualDiffusion模型
        
        参数:
            use_lora (bool): 是否使用LoRA
            lora_config (dict): LoRA配置，包括rank, alpha, dropout和target_modules
            lora_checkpoint (str): LoRA权重路径，用于加载预训练的LoRA权重
        """
        super().__init__(*args, **kwargs)

        self.save_result_dir = save_result_dir
        if self.save_result_dir is not None and not os.path.exists(self.save_result_dir):
            os.makedirs(self.save_result_dir, exist_ok=True)
        
        # self.use_lora = use_lora
        # self.lora_config = lora_config or {
        #     "rank": 16,
        #     "alpha": 32,
        #     "dropout": 0.1,
        #     "target_modules": [
        #         "to_q", "to_k", "to_v", "to_out.0",
        #         "proj_in", "proj_out",
        #         "temopral_conv.conv1.2", "temopral_conv.conv2.3", 
        #         "temopral_conv.conv3.3", "temopral_conv.conv4.3"
        #     ],
        #     'exact_matches': [],
        #     "train_pdecoder": True  # 是否同时训练PDecoder
        # }
        
        # # 重新设置requires_grad
        # self._freeze_all()  # 先冻结所有参数
        
        # # 应用LoRA
        # if self.use_lora:
        #     self._apply_lora()
            
        #     # 加载LoRA权重
        #     if lora_checkpoint is not None and os.path.exists(lora_checkpoint):
        #         self._load_lora_weights(lora_checkpoint)
        
        # # 如果需要训练PDecoder
        # if self.lora_config.get("train_pdecoder", True):
        #     self._enable_pdecoder_training()
            

    def shared_step(self, batch, random_uncond, **kwargs):
        x, c, fs, gt, mask, fns = self.get_batch_input(batch, random_uncond=random_uncond, return_fs=True)
        kwargs.update({"fs": fs.long()})
        loss, loss_dict = self(x, c, gt, mask, fns, **kwargs)
        return loss, loss_dict

    def forward(self, x, c, gt, mask, fns, **kwargs):
        # t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=self.device).long()
        t = torch.randint(50, 51, (x.shape[0],), device=self.device).long()
        if self.use_dynamic_rescale:
            x = x * extract_into_tensor(self.scale_arr, t, x.shape)
        return self.p_losses(x, c, t, gt, mask, fns, **kwargs)
    
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
        gt = torch.tensor(gt).to(out[0].device)
        out.append(gt)

        out.append(batch['mask'])

        fn_list = [p.split('/')[-1] for p in batch['path']]
        out.append(fn_list)

        return out

    def p_losses(self, x_start, cond, t, gt, mask, fns, noise=None, **kwargs):
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

        gt_retitusion = self.p_decoder(feat_dict, mask, full_resolution=0)
        l1_loss = nn.L1Loss()
        # ipdb.set_trace()
        loss = l1_loss(gt_retitusion[:,0], gt)
        loss_dict = {}
        loss_dict.update({'p_l1_loss': loss})

        fn_2_pred = {}
        for fn_i in range(gt_retitusion.shape[0]):
            fn_2_pred[fns[fn_i]] = {}
            fn_2_pred[fns[fn_i]]['pred'] =  gt_retitusion[fn_i][0].detach().cpu().numpy().tolist()
            fn_2_pred[fns[fn_i]]['gt'] = gt[fn_i].detach().cpu().numpy().tolist()

        
        with open(os.path.join(self.save_result_dir, fns[0] + '.json'), 'w') as dump_f:
            json.dump(fn_2_pred, dump_f)
    

        return loss, loss_dict 


class LatentVisualDiffusionv14LoraTest2(LatentVisualDiffusionv14LoraTest1):
    """
    对LatentVisualDiffusionv14模型添加LoRA支持
    
    该模型支持：
    1. 使用LoRA微调UNet
    2. 同时训练PDecoder
    3. 保存和加载LoRA权重
    """
    def __init__(self, 
                 *args, 
                 save_result_dir=None,
                 use_lora=True, 
                 lora_config=None,
                 lora_checkpoint=None,  # 新增：LoRA权重路径
                 **kwargs):
        """
        初始化LoRA版本的LatentVisualDiffusion模型
        
        参数:
            use_lora (bool): 是否使用LoRA
            lora_config (dict): LoRA配置，包括rank, alpha, dropout和target_modules
            lora_checkpoint (str): LoRA权重路径，用于加载预训练的LoRA权重
        """
        super().__init__(*args, **kwargs)

        self.save_result_dir = save_result_dir
        if not os.path.exists(self.save_result_dir):
            os.makedirs(self.save_result_dir, exist_ok=True)
        
        # self.use_lora = use_lora
        # self.lora_config = lora_config or {
        #     "rank": 16,
        #     "alpha": 32,
        #     "dropout": 0.1,
        #     "target_modules": [
        #         "to_q", "to_k", "to_v", "to_out.0",
        #         "proj_in", "proj_out",
        #         "temopral_conv.conv1.2", "temopral_conv.conv2.3", 
        #         "temopral_conv.conv3.3", "temopral_conv.conv4.3"
        #     ],
        #     'exact_matches': [],
        #     "train_pdecoder": True  # 是否同时训练PDecoder
        # }
        
        # # 重新设置requires_grad
        # self._freeze_all()  # 先冻结所有参数
        
        # # 应用LoRA
        # if self.use_lora:
        #     self._apply_lora()
            
        #     # 加载LoRA权重
        #     if lora_checkpoint is not None and os.path.exists(lora_checkpoint):
        #         self._load_lora_weights(lora_checkpoint)
        
        # # 如果需要训练PDecoder
        # if self.lora_config.get("train_pdecoder", True):
        #     self._enable_pdecoder_training()





# v1: concat mask on feat + no extra supervision
class LatentVisualDiffusionv14Lorav1(LatentVisualDiffusionv14Lora):
    """
    对LatentVisualDiffusionv14模型添加LoRA支持
    
    该模型支持：
    1. 使用LoRA微调UNet
    2. 同时训练PDecoder
    3. 保存和加载LoRA权重
    """
    def __init__(self, 
                 *args, 
                 use_lora=True, 
                 lora_config=None,
                 lora_checkpoint=None,  # 新增：LoRA权重路径
                 **kwargs):
        """
        初始化LoRA版本的LatentVisualDiffusion模型
        
        参数:
            use_lora (bool): 是否使用LoRA
            lora_config (dict): LoRA配置，包括rank, alpha, dropout和target_modules
            lora_checkpoint (str): LoRA权重路径，用于加载预训练的LoRA权重
        """
        super().__init__(*args, **kwargs)
        
        self.use_lora = use_lora
        self.lora_config = lora_config or {
            "rank": 16,
            "alpha": 32,
            "dropout": 0.1,
            "target_modules": [
                "to_q", "to_k", "to_v", "to_out.0",
                "proj_in", "proj_out",
                "temopral_conv.conv1.2", "temopral_conv.conv2.3", 
                "temopral_conv.conv3.3", "temopral_conv.conv4.3"
            ],
            'exact_matches': [],
            "train_pdecoder": True  # 是否同时训练PDecoder
        }
        
        # 重新设置requires_grad
        self.p_decoder = PDecoderCondMask()
        self._freeze_all()  # 先冻结所有参数
        
        # 应用LoRA
        if self.use_lora:
            self._apply_lora()
            
            # 加载LoRA权重
            if lora_checkpoint is not None and os.path.exists(lora_checkpoint):
                self._load_lora_weights(lora_checkpoint)
        
        # 如果需要训练PDecoder
        if self.lora_config.get("train_pdecoder", True):
            self._enable_pdecoder_training()


class LatentVisualDiffusionv14Lorav1Test1(LatentVisualDiffusionv14Lora):
    """
    对LatentVisualDiffusionv14模型添加LoRA支持
    
    该模型支持：
    1. 使用LoRA微调UNet
    2. 同时训练PDecoder
    3. 保存和加载LoRA权重
    """
    def __init__(self, 
                 *args, 
                 use_lora=True, 
                 lora_config=None,
                 lora_checkpoint=None,  # 新增：LoRA权重路径
                 **kwargs):
        """
        初始化LoRA版本的LatentVisualDiffusion模型
        
        参数:
            use_lora (bool): 是否使用LoRA
            lora_config (dict): LoRA配置，包括rank, alpha, dropout和target_modules
            lora_checkpoint (str): LoRA权重路径，用于加载预训练的LoRA权重
        """
        super().__init__(*args, **kwargs)

        self.save_result_dir = '/work/guanqi/DynamiCrafter/2.19_result_v23test1_1800itr'
        if not os.path.exists(self.save_result_dir):
            os.makedirs(self.save_result_dir, exist_ok=True)
        
        self.use_lora = use_lora
        self.lora_config = lora_config or {
            "rank": 16,
            "alpha": 32,
            "dropout": 0.1,
            "target_modules": [
                "to_q", "to_k", "to_v", "to_out.0",
                "proj_in", "proj_out",
                "temopral_conv.conv1.2", "temopral_conv.conv2.3", 
                "temopral_conv.conv3.3", "temopral_conv.conv4.3"
            ],
            'exact_matches': [],
            "train_pdecoder": True  # 是否同时训练PDecoder
        }
        
        # 重新设置requires_grad
        self.p_decoder = PDecoderCondMask()
        self._freeze_all()  # 先冻结所有参数
        
        # 应用LoRA
        if self.use_lora:
            self._apply_lora()
            
            # 加载LoRA权重
            if lora_checkpoint is not None and os.path.exists(lora_checkpoint):
                self._load_lora_weights(lora_checkpoint)
        
        # 如果需要训练PDecoder
        if self.lora_config.get("train_pdecoder", True):
            self._enable_pdecoder_training()
            

    def shared_step(self, batch, random_uncond, **kwargs):
        x, c, fs, gt, mask, fns = self.get_batch_input(batch, random_uncond=random_uncond, return_fs=True)
        kwargs.update({"fs": fs.long()})
        loss, loss_dict = self(x, c, gt, mask, fns, **kwargs)
        return loss, loss_dict
    
    def forward(self, x, c, gt, mask, fns, **kwargs):
        # t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=self.device).long()
        t = torch.randint(50, 51, (x.shape[0],), device=self.device).long()
        if self.use_dynamic_rescale:
            x = x * extract_into_tensor(self.scale_arr, t, x.shape)
        return self.p_losses(x, c, t, gt, mask, fns, **kwargs)

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
        gt = torch.tensor(gt).to(out[0].device)
        out.append(gt)

        out.append(batch['mask'])

        fn_list = [p.split('/')[-1] for p in batch['path']]
        out.append(fn_list)

        return out

    def p_losses(self, x_start, cond, t, gt, mask, fns, noise=None, **kwargs):
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

        gt_retitusion = self.p_decoder(feat_dict, mask, full_resolution=0)
        l1_loss = nn.L1Loss()
        # ipdb.set_trace()
        loss = l1_loss(gt_retitusion[:,0], gt)
        loss_dict = {}
        loss_dict.update({'p_l1_loss': loss})

        fn_2_pred = {}
        for fn_i in range(gt_retitusion.shape[0]):
            fn_2_pred[fns[fn_i]] = {}
            fn_2_pred[fns[fn_i]]['pred'] =  gt_retitusion[fn_i][0].detach().cpu().numpy().tolist()
            fn_2_pred[fns[fn_i]]['gt'] = gt[fn_i].detach().cpu().numpy().tolist()

        
        with open(os.path.join(self.save_result_dir, fns[0] + '.json'), 'w') as dump_f:
            json.dump(fn_2_pred, dump_f)
    

        return loss, loss_dict 


class LatentVisualDiffusionv14Lorav1Test2(LatentVisualDiffusionv14Lorav1Test1):
    """
    对LatentVisualDiffusionv14模型添加LoRA支持
    
    该模型支持：
    1. 使用LoRA微调UNet
    2. 同时训练PDecoder
    3. 保存和加载LoRA权重
    """
    def __init__(self, 
                 *args, 
                 use_lora=True, 
                 lora_config=None,
                 lora_checkpoint=None,  # 新增：LoRA权重路径
                 **kwargs):
        """
        初始化LoRA版本的LatentVisualDiffusion模型
        
        参数:
            use_lora (bool): 是否使用LoRA
            lora_config (dict): LoRA配置，包括rank, alpha, dropout和target_modules
            lora_checkpoint (str): LoRA权重路径，用于加载预训练的LoRA权重
        """
        super().__init__(*args, **kwargs)

        self.save_result_dir = '/work/guanqi/DynamiCrafter/2.19_result_v23test2_1800itr'
        if not os.path.exists(self.save_result_dir):
            os.makedirs(self.save_result_dir, exist_ok=True)
        
        self.use_lora = use_lora
        self.lora_config = lora_config or {
            "rank": 16,
            "alpha": 32,
            "dropout": 0.1,
            "target_modules": [
                "to_q", "to_k", "to_v", "to_out.0",
                "proj_in", "proj_out",
                "temopral_conv.conv1.2", "temopral_conv.conv2.3", 
                "temopral_conv.conv3.3", "temopral_conv.conv4.3"
            ],
            'exact_matches': [],
            "train_pdecoder": True  # 是否同时训练PDecoder
        }
        
        # 重新设置requires_grad
        self.p_decoder = PDecoderCondMask()
        self._freeze_all()  # 先冻结所有参数
        
        # 应用LoRA
        if self.use_lora:
            self._apply_lora()
            
            # 加载LoRA权重
            if lora_checkpoint is not None and os.path.exists(lora_checkpoint):
                self._load_lora_weights(lora_checkpoint)
        
        # 如果需要训练PDecoder
        if self.lora_config.get("train_pdecoder", True):
            self._enable_pdecoder_training()




# v0: cond mask on feat + extra supervision
class LatentVisualDiffusionv14LoraElasticity(LatentVisualDiffusionv14):
    """
    对LatentVisualDiffusionv14模型添加LoRA支持
    
    该模型支持：
    1. 使用LoRA微调UNet
    2. 同时训练PDecoder
    3. 保存和加载LoRA权重
    """
    def __init__(self, 
                 *args, 
                 use_lora=True, 
                 lora_config=None,
                 trained_params_path=None,  # 新增：训练参数路径
                 save_only_trained_params=True,  # 新增：是否只保存训练参数
                 **kwargs):
        """
        初始化LoRA版本的LatentVisualDiffusion模型
        
        参数:
            use_lora (bool): 是否使用LoRA
            lora_config (dict): LoRA配置，包括rank, alpha, dropout和target_modules
            trained_params_path (str): 训练参数路径，用于加载预训练的参数
            save_only_trained_params (bool): 是否只保存训练参数
        """
        super().__init__(*args, **kwargs)
        
        # 输出基本配置信息
        print(f"LoRA配置: 使用={use_lora}, 参数路径={trained_params_path}")
        
        self.use_lora = use_lora
        self.save_only_trained_params = save_only_trained_params
        self.lora_config = lora_config or {
            "rank": 16,
            "alpha": 32,
            "dropout": 0.1,
            "target_modules": [
                "to_q", "to_k", "to_v", "to_out.0",
                "proj_in", "proj_out",
                "temopral_conv.conv1.2", "temopral_conv.conv2.3", 
                "temopral_conv.conv3.3", "temopral_conv.conv4.3"
            ],
            "exact_matches": [],  # 新增: 精确路径匹配列表
            "train_pdecoder": True  # 是否同时训练PDecoder
        }
        
        # 重新设置requires_grad
        self._freeze_all()  # 先冻结所有参数
        
        # 应用LoRA
        if self.use_lora:
            self._apply_lora()
            
            # 加载训练参数
            if trained_params_path is not None:
                if os.path.exists(trained_params_path):
                    print(f"加载训练参数: {trained_params_path}")
                    self._load_trained_params(trained_params_path)
                else:
                    print(f"警告: 训练参数文件不存在: {trained_params_path}")
            else:
                print("未指定训练参数路径")
        
        # 如果需要训练PDecoder
        if self.lora_config.get("train_pdecoder", True):
            self._enable_pdecoder_training()
            
    def _apply_lora(self):
        """
        将UNet模型中的目标模块替换为LoRA版本
        """
        # 获取LoRA配置
        rank = self.lora_config.get("rank", 16)
        alpha = self.lora_config.get("alpha", 32)
        dropout = self.lora_config.get("dropout", 0.1)
        target_modules = self.lora_config.get("target_modules", None)
        exact_matches = self.lora_config.get("exact_matches", [])  # 新增: 获取精确匹配列表
        
        # 应用LoRA到UNet
        self.model.diffusion_model = replace_unet_modules_with_lora(
            self.model.diffusion_model, 
            rank=rank, 
            alpha=alpha, 
            dropout=dropout,
            target_replace_module=target_modules,
            exact_matches=exact_matches  # 新增: 传递精确匹配列表
        )
    
    def _load_trained_params(self, checkpoint_path):
        """
        加载训练参数
        
        参数:
            checkpoint_path (str): 训练参数路径
        """
        print(f"正在加载训练参数: {checkpoint_path}")
        # 显示当前PDecoder加载设置
        train_pdecoder = self.lora_config.get("train_pdecoder", True)
        print(f"PDecoder加载设置: train_pdecoder={train_pdecoder}")
        
        # 加载LoRA和PDecoder参数
        load_trained_parameters(
            model=self, 
            path=checkpoint_path, 
            load_lora=True, 
            load_pdecoder=train_pdecoder
        )
        print("训练参数加载完成!")
    
    def _enable_pdecoder_training(self):
        """
        启用PDecoder的训练
        """
        print("启用PDecoder训练...")
        # 遍历所有参数，启用PDecoder的梯度
        for name, param in self.named_parameters():
            if "p_decoder" in name:
                param.requires_grad = True
                
        print(f"PDecoder参数已启用训练.")
    
    def validation_step(self, batch, batch_idx):
        """
        验证步骤，添加random_uncond参数
        """
        # 在验证时使用False作为random_uncond参数
        _, loss_dict_no_ema = self.shared_step(batch, random_uncond=False)
        
        self.log_dict(
            {f"val/{key}": val for key, val in loss_dict_no_ema.items()},
            prog_bar=True,
            logger=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        return loss_dict_no_ema
        
    def configure_optimizers(self):
        """
        配置优化器，为LoRA参数和PDecoder参数分别设置不同的学习率
        """
        lr = self.learning_rate
        
        # 获取所有LoRA参数
        lora_params = get_lora_parameters(self.model.diffusion_model)
        
        # 获取所有PDecoder参数
        pdecoder_params = []
        if self.lora_config.get("train_pdecoder", True):
            pdecoder_params = get_pdecoder_parameters(self)
        
        # 设置不同的参数组
        param_groups = [
            {"params": lora_params, "lr": lr * 5.0},  # LoRA参数使用更大的学习率
        ]
        
        if pdecoder_params:
            param_groups.append({"params": pdecoder_params, "lr": lr})
        
        # 创建优化器
        optimizer = torch.optim.AdamW(param_groups)
        
        if self.use_scheduler:
            lr_scheduler = self.configure_schedulers(optimizer)
            return [optimizer], [lr_scheduler]
        return optimizer
    
    def on_save_checkpoint(self, checkpoint):
        """
        保存检查点时，保存训练参数
        
        如果save_only_trained_params=True，则仅保存训练参数
        否则，同时保存完整检查点和训练参数
        """
        if not self.save_only_trained_params:
            # 使用默认的保存机制保存完整检查点
            super().on_save_checkpoint(checkpoint)
        
        # 将LoRA配置保存到检查点
        checkpoint["lora_config"] = self.lora_config
        
        # 创建训练配置
        training_config = {
            "lora_config": self.lora_config,
            "learning_rate": self.learning_rate,
            "global_step": self.global_step,
            "current_epoch": self.current_epoch
        }
        
        # 创建一个单独的训练参数保存路径
        try:
            # 尝试获取当前checkpoint callback的dirpath
            if hasattr(self.trainer, "checkpoint_callback") and hasattr(self.trainer.checkpoint_callback, "dirpath"):
                dirpath = self.trainer.checkpoint_callback.dirpath
                global_step = self.global_step
                
                # 如果只保存训练参数，则完全替换检查点
                if self.save_only_trained_params:
                    filename = f"{dirpath}/trained_params_step_{global_step}"
                    full_path = save_trained_parameters(self, filename, training_config)
                    # 用训练参数文件路径替换检查点文件路径
                    checkpoint["_target_module_file"] = full_path
                    checkpoint["state_dict"] = {}  # 清空state_dict
                else:
                    # 同时保存训练参数和完整检查点
                    filename = f"{dirpath}/trained_params_step_{global_step}"
                    save_trained_parameters(self, filename, training_config)
        except Exception as e:
            print(f"保存训练参数错误: {e}")
    
    def on_train_epoch_end(self):
        """
        在每个训练epoch结束时保存训练参数
        """
        super().on_train_epoch_end()
        
        # 创建训练配置
        training_config = {
            "lora_config": self.lora_config,
            "learning_rate": self.learning_rate,
            "current_epoch": self.current_epoch
        }
        
        # 创建一个单独的训练参数保存路径
        try:
            # 尝试获取当前checkpoint callback的dirpath
            if hasattr(self.trainer, "checkpoint_callback") and hasattr(self.trainer.checkpoint_callback, "dirpath"):
                dirpath = self.trainer.checkpoint_callback.dirpath
                epoch = self.current_epoch
                filename = f"{dirpath}/trained_params_epoch_{epoch}"
                
                # 保存训练参数
                save_trained_parameters(self, filename, training_config)
        except Exception as e:
            print(f"保存训练参数错误: {e}") 

    def forward(self, x, c, gt, mask, keytime_mask, **kwargs):
        # t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=self.device).long()
        t = torch.randint(50, 51, (x.shape[0],), device=self.device).long()
        if self.use_dynamic_rescale:
            x = x * extract_into_tensor(self.scale_arr, t, x.shape)
        return self.p_losses(x, c, t, gt, mask, keytime_mask, **kwargs)

    def shared_step(self, batch, random_uncond, **kwargs):
        x, c, fs, gt, mask, keytime_mask = self.get_batch_input(batch, random_uncond=random_uncond, return_fs=True)
        kwargs.update({"fs": fs.long()})
        loss, loss_dict = self(x, c, gt, mask, keytime_mask, **kwargs)
        return loss, loss_dict

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


class LatentVisualDiffusionv14LoraElasticityTest1(LatentVisualDiffusionv14Lora):
    """
    对LatentVisualDiffusionv14模型添加LoRA支持
    
    该模型支持：
    1. 使用LoRA微调UNet
    2. 同时训练PDecoder
    3. 保存和加载LoRA权重
    """
    def __init__(self, 
                 *args, 
                 use_lora=True, 
                 lora_config=None,
                 lora_checkpoint=None,  # 新增：LoRA权重路径
                 **kwargs):
        """
        初始化LoRA版本的LatentVisualDiffusion模型
        
        参数:
            use_lora (bool): 是否使用LoRA
            lora_config (dict): LoRA配置，包括rank, alpha, dropout和target_modules
            lora_checkpoint (str): LoRA权重路径，用于加载预训练的LoRA权重
        """
        super().__init__(*args, **kwargs)

        self.save_result_dir = '/work/guanqi/DynamiCrafter/2.19_result_v21test1_5800itr_debug'
        if not os.path.exists(self.save_result_dir):
            os.makedirs(self.save_result_dir, exist_ok=True)
        
        self.use_lora = use_lora
        self.lora_config = lora_config or {
            "rank": 16,
            "alpha": 32,
            "dropout": 0.1,
            "target_modules": [
                "to_q", "to_k", "to_v", "to_out.0",
                "proj_in", "proj_out",
                "temopral_conv.conv1.2", "temopral_conv.conv2.3", 
                "temopral_conv.conv3.3", "temopral_conv.conv4.3"
            ],
            'exact_matches': [],
            "train_pdecoder": True  # 是否同时训练PDecoder
        }
        
        # 重新设置requires_grad
        self.p_decoder = PDecoderSupElasticity()
        self._freeze_all()  # 先冻结所有参数
        
        # 应用LoRA
        if self.use_lora:
            self._apply_lora()
            
            # 加载LoRA权重
            if lora_checkpoint is not None and os.path.exists(lora_checkpoint):
                self._load_lora_weights(lora_checkpoint)
        
        # 如果需要训练PDecoder
        if self.lora_config.get("train_pdecoder", True):
            self._enable_pdecoder_training()
            

    def forward(self, x, c, gt, mask, fns, **kwargs):
        # t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=self.device).long()
        t = torch.randint(50, 51, (x.shape[0],), device=self.device).long()
        if self.use_dynamic_rescale:
            x = x * extract_into_tensor(self.scale_arr, t, x.shape)
        return self.p_losses(x, c, t, gt, mask, fns, **kwargs)

    def shared_step(self, batch, random_uncond, **kwargs):
        x, c, fs, gt, mask, fns = self.get_batch_input(batch, random_uncond=random_uncond, return_fs=True)
        kwargs.update({"fs": fs.long()})
        loss, loss_dict = self(x, c, gt, mask, fns, **kwargs)
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
        gt = torch.tensor(gt).to(out[0].device)
        out.append(gt)

        out.append(batch['mask'])

        fn_list = [p.split('/')[-1] for p in batch['path']]
        out.append(fn_list)

        return out

    def p_losses(self, x_start, cond, t, gt, mask, fns, noise=None, **kwargs):
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
        # ipdb.set_trace()
        loss = l1_loss(gt_retitusion[:,0], gt)
        loss_dict = {}
        loss_dict.update({'p_l1_loss': loss})

        print(loss)

        fn_2_pred = {}
        for fn_i in range(gt_retitusion.shape[0]):
            fn_2_pred[fns[fn_i]] = {}
            fn_2_pred[fns[fn_i]]['pred'] =  gt_retitusion[fn_i][0].detach().cpu().numpy().tolist()
            fn_2_pred[fns[fn_i]]['gt'] = gt[fn_i].detach().cpu().numpy().tolist()

        
        with open(os.path.join(self.save_result_dir, fns[0] + '.json'), 'w') as dump_f:
            json.dump(fn_2_pred, dump_f)
    

        return loss, loss_dict 

    
class LatentVisualDiffusionv14LoraElasticityTest2(LatentVisualDiffusionv14LoraElasticityTest1):
    """
    对LatentVisualDiffusionv14模型添加LoRA支持
    
    该模型支持：
    1. 使用LoRA微调UNet
    2. 同时训练PDecoder
    3. 保存和加载LoRA权重
    """
    def __init__(self, 
                 *args, 
                 use_lora=True, 
                 lora_config=None,
                 lora_checkpoint=None,  # 新增：LoRA权重路径
                 **kwargs):
        """
        初始化LoRA版本的LatentVisualDiffusion模型
        
        参数:
            use_lora (bool): 是否使用LoRA
            lora_config (dict): LoRA配置，包括rank, alpha, dropout和target_modules
            lora_checkpoint (str): LoRA权重路径，用于加载预训练的LoRA权重
        """
        super().__init__(*args, **kwargs)

        self.save_result_dir = '/work/guanqi/DynamiCrafter/2.19_result_v21test2_1ep'
        if not os.path.exists(self.save_result_dir):
            os.makedirs(self.save_result_dir, exist_ok=True)
        
        self.use_lora = use_lora
        self.lora_config = lora_config or {
            "rank": 16,
            "alpha": 32,
            "dropout": 0.1,
            "target_modules": [
                "to_q", "to_k", "to_v", "to_out.0",
                "proj_in", "proj_out",
                "temopral_conv.conv1.2", "temopral_conv.conv2.3", 
                "temopral_conv.conv3.3", "temopral_conv.conv4.3"
            ],
            'exact_matches': [],
            "train_pdecoder": True  # 是否同时训练PDecoder
        }
        
        # 重新设置requires_grad
        self.p_decoder = PDecoderSupElasticity()
        self._freeze_all()  # 先冻结所有参数
        
        # 应用LoRA
        if self.use_lora:
            self._apply_lora()
            
            # 加载LoRA权重
            if lora_checkpoint is not None and os.path.exists(lora_checkpoint):
                self._load_lora_weights(lora_checkpoint)
        
        # 如果需要训练PDecoder
        if self.lora_config.get("train_pdecoder", True):
            self._enable_pdecoder_training()



# v1: multiply mask on feat + extra supervision
class LatentVisualDiffusionv14LoraElasticityv1(LatentVisualDiffusionv14LoraElasticity):
    """
    对LatentVisualDiffusionv14模型添加LoRA支持
    
    该模型支持：
    1. 使用LoRA微调UNet
    2. 同时训练PDecoder
    3. 保存和加载LoRA权重
    """
    def __init__(self, 
                 *args, 
                 use_lora=True, 
                 lora_config=None,
                 lora_checkpoint=None,  # 新增：LoRA权重路径
                 **kwargs):
        """
        初始化LoRA版本的LatentVisualDiffusion模型
        
        参数:
            use_lora (bool): 是否使用LoRA
            lora_config (dict): LoRA配置，包括rank, alpha, dropout和target_modules
            lora_checkpoint (str): LoRA权重路径，用于加载预训练的LoRA权重
        """
        super().__init__(*args, **kwargs)
        
        self.use_lora = use_lora
        self.lora_config = lora_config or {
            "rank": 16,
            "alpha": 32,
            "dropout": 0.1,
            "target_modules": [
                "to_q", "to_k", "to_v", "to_out.0",
                "proj_in", "proj_out",
                "temopral_conv.conv1.2", "temopral_conv.conv2.3", 
                "temopral_conv.conv3.3", "temopral_conv.conv4.3"
            ],
            'exact_matches': [],
            "train_pdecoder": True  # 是否同时训练PDecoder
        }
        
        # 重新设置requires_grad
        self.p_decoder = PDecoderSupElasticityv1()
        self._freeze_all()  # 先冻结所有参数
        
        # 应用LoRA
        if self.use_lora:
            self._apply_lora()
            
            # 加载LoRA权重
            if lora_checkpoint is not None and os.path.exists(lora_checkpoint):
                self._load_lora_weights(lora_checkpoint)
        
        # 如果需要训练PDecoder
        if self.lora_config.get("train_pdecoder", True):
            self._enable_pdecoder_training()




class LatentVisualDiffusionv14LoraElasticityv1Test1(LatentVisualDiffusionv14LoraElasticityTest1):
    """
    对LatentVisualDiffusionv14模型添加LoRA支持
    
    该模型支持：
    1. 使用LoRA微调UNet
    2. 同时训练PDecoder
    3. 保存和加载LoRA权重
    """
    def __init__(self, 
                 *args, 
                 save_result_dir=None,
                 use_lora=True, 
                 lora_config=None,
                 lora_checkpoint=None,  # 新增：LoRA权重路径
                 **kwargs):
        """
        初始化LoRA版本的LatentVisualDiffusion模型
        
        参数:
            use_lora (bool): 是否使用LoRA
            lora_config (dict): LoRA配置，包括rank, alpha, dropout和target_modules
            lora_checkpoint (str): LoRA权重路径，用于加载预训练的LoRA权重
        """
        super().__init__(*args, **kwargs)

        self.save_result_dir = save_result_dir
        if self.save_result_dir is not None and not os.path.exists(self.save_result_dir):
            os.makedirs(self.save_result_dir, exist_ok=True)
        
        self.use_lora = use_lora
        self.lora_config = lora_config or {
            "rank": 16,
            "alpha": 32,
            "dropout": 0.1,
            "target_modules": [
                "to_q", "to_k", "to_v", "to_out.0",
                "proj_in", "proj_out",
                "temopral_conv.conv1.2", "temopral_conv.conv2.3", 
                "temopral_conv.conv3.3", "temopral_conv.conv4.3"
            ],
            'exact_matches': [],
            "train_pdecoder": True  # 是否同时训练PDecoder
        }
        
        # 重新设置requires_grad
        self.p_decoder = PDecoderSupElasticityv1()
        self._freeze_all()  # 先冻结所有参数
        
        # 应用LoRA
        if self.use_lora:
            self._apply_lora()
            
            # 加载LoRA权重
            if lora_checkpoint is not None and os.path.exists(lora_checkpoint):
                self._load_lora_weights(lora_checkpoint)
        
        # 如果需要训练PDecoder
        if self.lora_config.get("train_pdecoder", True):
            self._enable_pdecoder_training()


class LatentVisualDiffusionv14LoraElasticityv1Test2(LatentVisualDiffusionv14LoraElasticityTest1):
    """
    对LatentVisualDiffusionv14模型添加LoRA支持
    
    该模型支持：
    1. 使用LoRA微调UNet
    2. 同时训练PDecoder
    3. 保存和加载LoRA权重
    """
    def __init__(self, 
                 *args, 
                 save_result_dir=None,
                 use_lora=True, 
                 lora_config=None,
                 lora_checkpoint=None,  # 新增：LoRA权重路径
                 **kwargs):
        """
        初始化LoRA版本的LatentVisualDiffusion模型
        
        参数:
            use_lora (bool): 是否使用LoRA
            lora_config (dict): LoRA配置，包括rank, alpha, dropout和target_modules
            lora_checkpoint (str): LoRA权重路径，用于加载预训练的LoRA权重
        """
        super().__init__(*args, **kwargs)

        self.save_result_dir = save_result_dir
        if not os.path.exists(self.save_result_dir):
            os.makedirs(self.save_result_dir, exist_ok=True)
        
        self.use_lora = use_lora
        self.lora_config = lora_config or {
            "rank": 16,
            "alpha": 32,
            "dropout": 0.1,
            "target_modules": [
                "to_q", "to_k", "to_v", "to_out.0",
                "proj_in", "proj_out",
                "temopral_conv.conv1.2", "temopral_conv.conv2.3", 
                "temopral_conv.conv3.3", "temopral_conv.conv4.3"
            ],
            'exact_matches': [],
            "train_pdecoder": True  # 是否同时训练PDecoder
        }
        
        # 重新设置requires_grad
        self.p_decoder = PDecoderSupElasticityv1()
        self._freeze_all()  # 先冻结所有参数
        
        # 应用LoRA
        if self.use_lora:
            self._apply_lora()
            
            # 加载LoRA权重
            if lora_checkpoint is not None and os.path.exists(lora_checkpoint):
                self._load_lora_weights(lora_checkpoint)
        
        # 如果需要训练PDecoder
        if self.lora_config.get("train_pdecoder", True):
            self._enable_pdecoder_training()



# v1: multiply mask on feat + extra supervision + extra supervision loss coef to be 0.01
class LatentVisualDiffusionv14LoraElasticityv2(LatentVisualDiffusionv14LoraElasticity):
    """
    对LatentVisualDiffusionv14模型添加LoRA支持
    
    该模型支持：
    1. 使用LoRA微调UNet
    2. 同时训练PDecoder
    3. 保存和加载LoRA权重
    """
    def __init__(self, 
                 *args, 
                 use_lora=True, 
                 lora_config=None,
                 lora_checkpoint=None,  # 新增：LoRA权重路径
                 **kwargs):
        """
        初始化LoRA版本的LatentVisualDiffusion模型
        
        参数:
            use_lora (bool): 是否使用LoRA
            lora_config (dict): LoRA配置，包括rank, alpha, dropout和target_modules
            lora_checkpoint (str): LoRA权重路径，用于加载预训练的LoRA权重
        """
        super().__init__(*args, **kwargs)
        
        self.use_lora = use_lora
        self.lora_config = lora_config or {
            "rank": 16,
            "alpha": 32,
            "dropout": 0.1,
            "target_modules": [
                "to_q", "to_k", "to_v", "to_out.0",
                "proj_in", "proj_out",
                "temopral_conv.conv1.2", "temopral_conv.conv2.3", 
                "temopral_conv.conv3.3", "temopral_conv.conv4.3"
            ],
            'exact_matches': [],
            "train_pdecoder": True  # 是否同时训练PDecoder
        }
        
        # 重新设置requires_grad
        self.p_decoder = PDecoderSupElasticityv1()
        self._freeze_all()  # 先冻结所有参数
        
        # 应用LoRA
        if self.use_lora:
            self._apply_lora()
            
            # 加载LoRA权重
            if lora_checkpoint is not None and os.path.exists(lora_checkpoint):
                self._load_lora_weights(lora_checkpoint)
        
        # 如果需要训练PDecoder
        if self.lora_config.get("train_pdecoder", True):
            self._enable_pdecoder_training()


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

        extra_loss_coef = 0.01
        
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

    


