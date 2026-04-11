import argparse, os, sys, datetime
from omegaconf import OmegaConf
from transformers import logging as transf_logging
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.trainer import Trainer
import torch
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from utils.utils import instantiate_from_config
from utils_train import get_trainer_callbacks, get_trainer_logger, get_trainer_strategy
from utils_train import set_logger, init_workspace, load_checkpoints
from pytorch_lightning.callbacks import RichProgressBar

def get_parser(**parser_kwargs):
    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument('--seed', '-s', type=int, default=20230211, help='seed for seed_everything')
    parser.add_argument('--name', '-n', type=str, default='', help='experiment name, as saving folder')
    parser.add_argument('--base', '-b', nargs='*', metavar='base_config.yaml', help='paths to base configs. Loaded from left-to-right. Parameters can be overwritten or added with command-line options of the form `--key value`.', default=list())
    parser.add_argument('--train', '-t', action='store_true', default=False, help='train')
    parser.add_argument('--val', '-v', action='store_true', default=False, help='val')
    parser.add_argument('--test', action='store_true', default=False, help='test')
    parser.add_argument('--logdir', '-l', type=str, default='logs', help='directory for logging dat shit')
    parser.add_argument('--auto_resume', action='store_true', default=False, help='resume from full-info checkpoint')
    parser.add_argument('--auto_resume_weight_only', action='store_true', default=False, help='resume from weight-only checkpoint')
    parser.add_argument('--debug', '-d', action='store_true', default=False, help='enable post-mortem debugging')
    return parser

def get_nondefault_trainer_args(args):
    parser = argparse.ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    default_trainer_args = parser.parse_args([])
    return sorted((k for k in vars(default_trainer_args) if getattr(args, k) != getattr(default_trainer_args, k)))
if __name__ == '__main__':
    now = datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    global_rank = int(os.environ.get('RANK', '0'))
    num_rank = int(os.environ.get('WORLD_SIZE', '1'))
    parser = get_parser()
    parser = Trainer.add_argparse_args(parser)
    args, unknown = parser.parse_known_args()
    transf_logging.set_verbosity_error()
    seed_everything(args.seed)
    configs = [OmegaConf.load(cfg) for cfg in args.base]
    cli = OmegaConf.from_dotlist(unknown)
    config = OmegaConf.merge(*configs, cli)
    lightning_config = config.pop('lightning', OmegaConf.create())
    trainer_config = lightning_config.get('trainer', OmegaConf.create())
    workdir, ckptdir, cfgdir, loginfo = init_workspace(args.name, args.logdir, config, lightning_config, global_rank)
    logger = set_logger(logfile=os.path.join(loginfo, 'log_%d:%s.txt' % (global_rank, now)))
    logger.info('@lightning version: %s [>=1.8 required]' % pl.__version__)
    logger.info('***** Configing Data *****')
    data = instantiate_from_config(config.data)
    data.setup()
    for k in data.datasets:
        logger.info(f'{k}, {data.datasets[k].__class__.__name__}, {len(data.datasets[k])}')
    logger.info('***** Configing Model *****')
    config.model.params.logdir = workdir
    model = instantiate_from_config(config.model)
    model = load_checkpoints(model, config.model)
    if model.rescale_betas_zero_snr:
        model.register_schedule(given_betas=model.given_betas, beta_schedule=model.beta_schedule, timesteps=model.timesteps, linear_start=model.linear_start, linear_end=model.linear_end, cosine_s=model.cosine_s)
    for k in get_nondefault_trainer_args(args):
        trainer_config[k] = getattr(args, k)
    num_nodes = trainer_config.num_nodes
    if 'devices' in trainer_config:
        ngpu_per_node = trainer_config.devices
    else:
        ngpu_per_node = num_rank // num_nodes
    logger.info(f'Running on {num_rank}={num_nodes}x{ngpu_per_node} GPUs')
    base_lr = config.model.base_learning_rate
    bs = config.data.params.batch_size
    if getattr(config.model, 'scale_lr', True):
        model.learning_rate = num_rank * bs * base_lr
    else:
        model.learning_rate = base_lr
    logger.info('***** Configing Trainer *****')
    if 'accelerator' not in trainer_config:
        trainer_config['accelerator'] = 'gpu'
    trainer_kwargs = dict()
    trainer_kwargs['num_sanity_val_steps'] = 0
    logger_cfg = get_trainer_logger(lightning_config, workdir, args.debug)
    trainer_kwargs['logger'] = instantiate_from_config(logger_cfg)
    callbacks_cfg = get_trainer_callbacks(lightning_config, config, workdir, ckptdir, logger)
    trainer_kwargs['callbacks'] = [instantiate_from_config(callbacks_cfg[k]) for k in callbacks_cfg]
    trainer_kwargs['callbacks'].append(RichProgressBar())
    strategy_cfg = get_trainer_strategy(lightning_config)
    trainer_kwargs['strategy'] = strategy_cfg if type(strategy_cfg) == str else instantiate_from_config(strategy_cfg)
    trainer_kwargs['precision'] = lightning_config.get('precision', 32)
    trainer_kwargs['sync_batchnorm'] = False
    trainer_args = argparse.Namespace(**trainer_config)
    trainer = Trainer.from_argparse_args(trainer_args, **trainer_kwargs)

    def melk(*args, **kwargs):
        if trainer.global_rank == 0:
            print('Summoning checkpoint.')
            ckpt_path = os.path.join(ckptdir, 'last_summoning.ckpt')
            trainer.save_checkpoint(ckpt_path)

    def divein(*args, **kwargs):
        if trainer.global_rank == 0:
            import pudb
            pudb.set_trace()
    import signal
    signal.signal(signal.SIGUSR1, melk)
    signal.signal(signal.SIGUSR2, divein)
    logger.info('***** Running the Loop *****')
    resume_ckpt_path = None
    if args.auto_resume:
        logger.info('***** Checking for checkpoint to resume *****')
        last_ckpt = os.path.join(ckptdir, 'last.ckpt')
        if os.path.exists(last_ckpt):
            resume_ckpt_path = last_ckpt
            logger.info(f'Found last checkpoint: {resume_ckpt_path}')
        else:
            ckpt_files = []
            for f in os.listdir(ckptdir):
                if f.endswith('.ckpt') and 'epoch=' in f:
                    ckpt_files.append(f)
            if ckpt_files:
                ckpt_files.sort(key=lambda x: os.path.getmtime(os.path.join(ckptdir, x)), reverse=True)
                resume_ckpt_path = os.path.join(ckptdir, ckpt_files[0])
                logger.info(f'Found latest checkpoint: {resume_ckpt_path}')
            else:
                logger.info('No checkpoint found for resume, starting from scratch')
    if args.auto_resume and resume_ckpt_path:
        try:
            _resume_blob = torch.load(resume_ckpt_path, map_location='cpu')
        except Exception as exc:
            logger.warning('Could not read resume checkpoint %s: %s', resume_ckpt_path, exc)
            _resume_blob = None
        if _resume_blob is not None:
            _has_optimizer = bool(_resume_blob.get('optimizer_states'))
            if _has_optimizer:
                del _resume_blob
            else:
                logger.info('Resume checkpoint is weights-only; loaded weights into model, optimizer and global_step will restart (see ModelCheckpoint.save_weights_only).')
                if 'state_dict' in _resume_blob:
                    model.load_state_dict(_resume_blob['state_dict'], strict=False)
                else:
                    model.load_state_dict(_resume_blob, strict=False)
                resume_ckpt_path = None
                del _resume_blob
    if args.train:
        try:
            if 'strategy' in lightning_config and lightning_config['strategy'].startswith('deepspeed'):
                logger.info('<Training in DeepSpeed Mode>')
                if trainer_kwargs['precision'] == 16:
                    with torch.cuda.amp.autocast():
                        trainer.fit(model, data, ckpt_path=resume_ckpt_path)
                else:
                    trainer.fit(model, data, ckpt_path=resume_ckpt_path)
            else:
                logger.info('<Training in DDPSharded Mode>')
                trainer.fit(model, data, ckpt_path=resume_ckpt_path)
        except Exception:
            raise
    if args.val:
        trainer.validate(model, data)
