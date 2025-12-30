from generation_models import TimeVAECGATS
from Trainers import CGATPretrain
from dataset_utils import ECGDataset
import argparse
import torch
import json
import time
from datetime import datetime
from zoneinfo import ZoneInfo
import os

import numpy as np
from torch.utils.data import Subset


def save_args_to_jsonl(args, output_path):
    args_dict = vars(args)
    with open(output_path, "w") as f:
        json.dump(args_dict, f)
        f.write("\n")  # JSONL 一行一个 JSON


def get_pretrain_args():
    parser = argparse.ArgumentParser(description="parameters for TimeVAE-CGATS pretraining")

    """time series general parameters"""
    parser.add_argument("--seq_len", type=int, required=True)
    parser.add_argument("--feature_size", type=int, required=True)
    parser.add_argument("--one_channel", type=int, required=True)

    """model parameters"""
    parser.add_argument("--latent_dim", type=int, required=True)
    parser.add_argument("--trend_poly", type=int, required=True)
    parser.add_argument("--kl_wt", type=float, required=True)
    parser.add_argument("--hidden_layer_sizes", type=json.loads, required=True)
    parser.add_argument("--custom_seas", type=json.loads, required=True)


    """data parameters"""
    parser.add_argument("--max_anomaly_length", type=int, required=True)
    parser.add_argument("--min_anomaly_length", type=int, required=True)
    parser.add_argument("--raw_data_paths_train", type=str, required=True)
    parser.add_argument("--indices_paths_train", type=str, required=True)

    """training parameters"""
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--grad_clip_norm", type=float, required=True)
    parser.add_argument("--early_stop", type=str, required=True)
    parser.add_argument("--patience", type=int, required=True)

    """wandb parameters"""
    parser.add_argument("--wandb_project", type=str,required=True)
    parser.add_argument("--wandb_run", type=str, required=True)

    """save and load parameters"""
    parser.add_argument("--ckpt_dir", type=str, required=True)

    """gpu parameters"""
    parser.add_argument("--gpu_id", type=int, required=True)

    return parser.parse_args()


def pretrain():
    args = get_pretrain_args()
    # timestamp = datetime.now(ZoneInfo("America/Chicago")).strftime("%Y-%m-%d-%H:%M:%S")
    # args.ckpt_dir = f"{args.ckpt_dir}/{timestamp}"
    os.makedirs(args.ckpt_dir, exist_ok=True)
    save_args_to_jsonl(args, f"{args.ckpt_dir}/config.jsonl")

    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    model = TimeVAECGATS(
        hidden_layer_sizes=args.hidden_layer_sizes,
        trend_poly=args.trend_poly,
        custom_seas=args.custom_seas,
        use_residual_conn=True,
        seq_len=args.seq_len,
        feat_dim=args.feature_size,
        latent_dim=args.latent_dim,
        kl_wt = args.kl_wt,
    )


    full_set = ECGDataset(
        raw_data_paths=args.raw_data_paths_train,
        indices_paths=args.indices_paths_train,
        seq_len=args.seq_len,
        max_anomaly_length=args.max_anomaly_length,
        min_anomaly_length=args.min_anomaly_length,
        one_channel=args.one_channel,
    )
    N = len(full_set)
    indices = np.arange(N)

    split = int(0.8 * N)
    train_idx = indices[:split]
    val_idx = indices[split:]

    train_set = Subset(full_set, train_idx)
    val_set = Subset(full_set, val_idx)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=args.batch_size, shuffle=False, drop_last=False)


    optimizer= torch.optim.Adam(model.parameters(), lr=args.lr)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.8,  # multiply LR by 0.5
        patience=4,  # wait 3 epochs with no improvement
        threshold=1e-4,  # improvement threshold
        min_lr=1e-6,  # min LR clamp
    )

    trainer = CGATPretrain(
        optimizer=optimizer,
        scheduler=scheduler,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        max_epochs=args.epochs,
        device=device,
        save_dir=args.ckpt_dir,
        wandb_run_name=args.wandb_run,
        wandb_project_name=args.wandb_project,
        grad_clip_norm=args.grad_clip_norm,
        early_stop=args.early_stop,
        patience=args.patience,
    )
    trainer.pretrain(config=vars(args))

if __name__ == "__main__":
    pretrain()