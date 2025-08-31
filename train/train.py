import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)
from ddpm import Unet3D, Trainer, GaussianDiffusion_Nolatent
import hydra
from omegaconf import DictConfig
from Dataset.TS_Dataset import get_TS_dataloader
from Dataset.MMWHS_Dataset import get_MMWHS_dataloader
import torch
from ddpm.unet import UNet
import torch.nn as nn
import sys
import os
from datetime import datetime
import atexit
import random
import numpy as np
from omegaconf import OmegaConf

def set_seed(seed):
    random.seed(seed)     
    np.random.seed(seed) 
    torch.manual_seed(seed) 
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.deterministic = True  

@hydra.main(config_path='config', config_name='base_cfg', version_base=None)
def run(cfg: DictConfig):
    set_seed(1)
    print(OmegaConf.to_container(cfg, resolve=True))
    # torch.cuda.set_device(cfg.model.gpus)
    # set_seed(cfg.model.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if cfg.model.denoising_fn == 'Unet3D':
        model = Unet3D(
            dim=cfg.model.diffusion_img_size,
            dim_mults=cfg.model.dim_mults,
            channels=cfg.model.diffusion_num_channels,
            cond_dim=16,
        )
    elif cfg.model.denoising_fn == 'UNet':
        model = UNet(
            in_ch=cfg.model.diffusion_num_channels,
            out_ch=cfg.model.diffusion_num_channels,
            spatial_dims=3
        )
    else:
        raise ValueError(f"Model {cfg.model.denoising_fn} doesn't exist")

    model = nn.DataParallel(model)

    diffusion = GaussianDiffusion_Nolatent(
        model,
        image_size=cfg.model.diffusion_img_size,
        num_frames=cfg.model.diffusion_depth_size,
        channels=cfg.model.diffusion_num_channels,
        timesteps=cfg.model.timesteps,
        loss_type=cfg.model.loss_type,
        device=device
    ).to(device)

    if cfg.dataset.name == 'MMWHS':
        train_dataset = get_MMWHS_dataloader(root_dir=cfg.dataset.root_dir, mode=cfg.dataset.mode, data_type=cfg.dataset.data_type) 
    elif cfg.dataset.name == 'TS' :
        train_dataset = get_TS_dataloader(root_dir=cfg.dataset.root_dir, mode=cfg.dataset.mode)
    else :
        raise ValueError ("No Such Dataset")

    trainer = Trainer(
        diffusion,
        cfg=cfg,
        dataset=train_dataset,
        train_batch_size=cfg.model.batch_size,
        save_and_sample_every=cfg.model.save_and_sample_every,
        train_lr=cfg.model.train_lr,
        train_num_steps=cfg.model.train_num_steps,
        gradient_accumulate_every=cfg.model.gradient_accumulate_every,
        ema_decay=cfg.model.ema_decay,
        amp=cfg.model.amp,
        results_folder=cfg.model.results_folder,
        num_workers=cfg.model.num_workers,
        device=device,
    )

    if cfg.model.load_milestone:
        trainer.load(cfg.model.load_milestone)

    trainer.train()

class Tee:
    def __init__(self, *files):
        self.files = files
    
    def write(self, obj):
        for f in self.files:
            if not f.closed:
                f.write(obj)
                f.flush()  
    
    def flush(self):
        for f in self.files:
            if not f.closed:
                f.flush()


if __name__ == '__main__':

    log_dir = "./log_train"
    os.makedirs(log_dir, exist_ok=True)
    filename = os.path.join(log_dir, datetime.now().strftime("%Y%m%d_%H%M%S") + ".log")
    log_file = open(filename, 'w', encoding='utf-8')
    sys.stdout = Tee(sys.stdout, log_file)
    atexit.register(lambda: log_file.close())
    run()
