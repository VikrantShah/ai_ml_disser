import argparse
import shutil
from pathlib import Path
import yaml

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from torchvision.models import mobilenet_v2
import torch.nn as nn

def read_yaml(fpath):
    with open(fpath, "r") as stream:
        return yaml.safe_load(stream)

def make_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--kfold", required=True, type=int)
    parser.add_argument("--gpus", default="0", type=str)
    parser.add_argument("--debug", action="store_true")
    return parser

def seed_torch(seed=42):
    import random
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def make_output_path(output_path, debug):
    output_path.mkdir(parents=True, exist_ok=True)
    if debug:
        output_path = output_path / "debug"
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path

def src_backup(input_dir, output_dir):
    for item in input_dir.iterdir():
        if item.is_dir():
            shutil.copytree(item, output_dir / item.name, dirs_exist_ok=True)
        else:
            shutil.copy2(item, output_dir / item.name)

class MyLogger:
    # Placeholder for logger implementation
    pass

class LightningModuleReg(nn.Module):
    def __init__(self, cfg):
        super(LightningModuleReg, self).__init__()
        self.cfg = cfg
        self.model = mobilenet_v2(pretrained=cfg['Model']['pretrained'])
        self.model.classifier[1] = nn.Linear(self.model.last_channel, cfg['Model']['out_channel'])

    def forward(self, x):
        return self.model(x)

def train_a_kfold(cfg, cfg_name, output_path):
    # Model checkpoints
    checkpoint_callback = ModelCheckpoint(
        dirpath=output_path,
        filename='{epoch}-{val_loss:.2f}',
        save_top_k=3,
        monitor='val_loss',
        mode='min',
        verbose=True,
    )

    # Logger
    logger_name = f"kfold_{str(cfg['Data']['dataset']['kfold']).zfill(2)}.csv"
    mylogger = MyLogger()  # Implement your logger as needed

    # Trainer
    seed_torch(cfg['General']['seed'])
    seed_everything(cfg['General']['seed'])
    debug = cfg['General']['debug']
    trainer = Trainer(
        logger=mylogger,
        max_epochs=5 if debug else cfg['General']['epoch'],
        checkpoint_callback=checkpoint_callback,
        early_stop_callback=False,
        train_percent_check=0.02 if debug else 1.0,
        val_percent_check=0.06 if debug else 1.0,
        gpus=cfg['General']['gpus'],
        use_amp=cfg['General']['fp16'],
        amp_level=cfg['General']['amp_level'],
        distributed_backend=cfg['General']['multi_gpu_mode'],
        log_save_interval=5 if debug else 200,
        accumulate_grad_batches=cfg['General']['grad_acc'],
        deterministic=True,
    )

    # Lightning module and start training
    model = LightningModuleReg(cfg)
    trainer.fit(model)

def main():
    args = make_parse().parse_args()

    # Read Config
    cfg = read_yaml(fpath=args.config)
    cfg['Data']['dataset']['kfold'] = args.kfold
    cfg['General']['debug'] = args.debug
    for key, value in cfg.items():
        print(f"    {key.ljust(30)}: {value}")

    # Set gpu
    cfg['General']['gpus'] = list(map(int, args.gpus.split(",")))

    # Make output path
    output_path = Path("../output/model") / Path(args.config).stem
    output_path = make_output_path(output_path, args.debug)

    # Source code backup
    shutil.copy2(args.config, str(output_path / Path(args.config).name))
    src_backup_path = output_path / "src_backup"
    src_backup_path.mkdir(exist_ok=True)
    src_backup(input_dir=Path("./"), output_dir=src_backup_path)

    # Train start
    # seed_torch(seed=cfg['General']['seed'])
    train_a_kfold(cfg, Path(args.config).stem, output_path)

if __name__ == "__main__":
    main()
