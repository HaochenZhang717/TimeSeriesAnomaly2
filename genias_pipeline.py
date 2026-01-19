from Trainers import GenIAS_Trainer
from generation_models import GenIASModel
from dataset_utils import ImputationNormalECGDataset, ImputationNormalECGDatasetForSample
from dataset_utils import ImputationECGDataset
import argparse
import torch
import json
import os
import numpy as np
from torch.utils.data import Subset



def dict_collate_fn(batch):
    out = {}
    for key in batch[0].keys():
        out[key] = torch.stack([sample[key] for sample in batch], dim=0)
    return out


def save_args_to_jsonl(args, output_path):
    args_dict = vars(args)
    with open(output_path, "w") as f:
        json.dump(args_dict, f)
        f.write("\n")  # JSONL 一行一个 JSON


def get_args():
    parser = argparse.ArgumentParser(description="parameters for flow-ts pretraining")
    """what to do"""
    parser.add_argument(
        "--what_to_do", type=str, required=True,
        choices=[
            "genias_trian",
            "impute_sample",
            "impute_normal_sample",
            "principle_impute_sample",
            "impute_sample_non_downstream",
        ],
        help="what to do"
    )

    """time series general parameters"""
    parser.add_argument("--data_type", type=str, required=True)
    parser.add_argument("--seq_len", type=int, required=True)
    parser.add_argument("--feature_size", type=int, required=True)
    parser.add_argument("--one_channel", type=int, required=True)
    parser.add_argument("--event_labels_paths_train", type=json.loads, required=True)

    """model parameters"""
    parser.add_argument("--hidden_layer_sizes", type=json.loads, required=True)
    parser.add_argument("--trend_poly", type=json.loads, required=True)
    parser.add_argument("--custom_seas", type=json.loads, required=True)
    parser.add_argument("--delta_min", type=float, required=True)
    parser.add_argument("--delta_max", type=float, required=True)
    parser.add_argument("--perturbation_weight", type=float, required=True)
    parser.add_argument("--kl_wt", type=float, required=True)
    parser.add_argument("--latent_dim", type=int, required=True)
    parser.add_argument("--sigma_prior", type=float, required=True)

    """data parameters"""
    parser.add_argument("--raw_data_paths_train", type=json.loads, required=True)
    parser.add_argument("--raw_data_paths_test", type=json.loads, required=True)
    parser.add_argument("--indices_paths_train", type=json.loads, required=True)
    parser.add_argument("--indices_paths_test", type=json.loads, required=True)
    parser.add_argument("--indices_paths_anomaly_for_sample", type=json.loads, default="none")
    parser.add_argument("--min_infill_length", type=int, required=True)
    parser.add_argument("--max_infill_length", type=int, required=True)

    """training parameters"""
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--max_epochs", type=int, required=True)
    parser.add_argument("--grad_clip_norm", type=float, required=True)
    parser.add_argument("--grad_accum_steps", type=int, required=True)
    parser.add_argument("--early_stop", type=str, required=True)
    parser.add_argument("--patience", type=int, required=True)

    """wandb parameters"""
    parser.add_argument("--wandb_project", type=str,required=True)
    parser.add_argument("--wandb_run", type=str, required=True)

    """save and load parameters"""
    parser.add_argument("--ckpt_dir", type=str, required=True)

    """save path """
    parser.add_argument("--generated_path", type=str, required=True)

    """gpu parameters"""
    parser.add_argument("--gpu_id", type=int, required=True)

    return parser.parse_args()




def impute_sample(args):

    model = GenIASModel(
        hidden_layer_sizes=args.hidden_layer_sizes,
        trend_poly=args.trend_poly,
        custom_seas=args.custom_seas,
        use_residual_conn=True,
        delta_min=args.delta_min,
        delta_max=args.delta_max,
        reconstruction_weight=1.0,
        perturbation_weight=args.perturbation_weight,
        kl_weight=args.kl_wt,
        seq_len=args.seq_len,
        feat_dim=args.feature_size,
        latent_dim=args.latent_dim,
        sigma_prior=args.sigma_prior,
    )


    model.load_state_dict(torch.load(f"{args.ckpt_dir}/ckpt.pth"))
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    model.to(device=device)
    model.eval()


    assert args.data_type == "ecg"
    normal_set = ImputationNormalECGDataset(
        raw_data_paths=args.raw_data_paths_train,
        indices_paths=args.indices_paths_train,
        seq_len=args.seq_len,
        one_channel=args.one_channel,
        min_infill_length=args.min_infill_length,
        max_infill_length=args.max_infill_length,
    )

    normal_loader = torch.utils.data.DataLoader(
        normal_set, batch_size=args.batch_size,
        shuffle=True, drop_last=True,
        collate_fn=dict_collate_fn,
    )



    num_generate = 10000
    all_samples = []
    all_labels = []
    all_reals = []
    num_samples = 0
    while num_samples < num_generate:
        for normal_batch in normal_loader:
            signals = normal_batch['signals'].to(device=device, dtype=torch.float32) #(batch_size, seq_len, ts_dim)
            attn_mask = normal_batch['attn_mask'].to(device=device, dtype=torch.bool) # (batch_size, seq_len)
            noise_mask = normal_batch['noise_mask'].to(device=device, dtype=torch.long)

            x_occluded = signals * attn_mask.unsqueeze(-1)
            with torch.no_grad():
                samples = model.get_anomaly_samples(
                    x_occluded,
                    noise_mask
                )

            all_samples.append(samples)
            all_labels.append(noise_mask)
            all_reals.append(signals)

            num_samples += samples.shape[0]
            print(f"Generated {num_samples}/{num_generate} ")
            if num_samples >= num_generate:
                break

    all_samples = torch.cat(all_samples, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    all_reals = torch.cat(all_reals, dim=0)

    all_results = {
        'all_samples': all_samples,
        'all_labels': all_labels,
        'all_reals': all_reals,
    }
    torch.save(all_results, f"{args.ckpt_dir}/no_code_impute_samples.pth")



def principle_impute_sample(args):


    model = GenIASModel(
        hidden_layer_sizes=args.hidden_layer_sizes,
        trend_poly=args.trend_poly,
        custom_seas=args.custom_seas,
        use_residual_conn=True,
        delta_min=args.delta_min,
        delta_max=args.delta_max,
        reconstruction_weight=1.0,
        perturbation_weight=args.perturbation_weight,
        kl_weight=args.kl_wt,
        seq_len=args.seq_len,
        feat_dim=args.feature_size,
        latent_dim=args.latent_dim,
        sigma_prior=args.sigma_prior,
    )

    model.load_state_dict(torch.load(f"{args.ckpt_dir}/ckpt.pth"))
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    model.to(device=device)
    model.eval()


    assert args.data_type == "ecg"

    normal_set = ImputationNormalECGDatasetForSample(
        raw_data_paths=args.raw_data_paths_train,
        indices_paths=args.indices_paths_train,
        event_labels_paths=args.event_labels_paths_train,
        seq_len=args.seq_len,
        one_channel=args.one_channel,
        min_infill_length=args.min_infill_length,
        max_infill_length=args.max_infill_length,
    )


    normal_loader = torch.utils.data.DataLoader(
        normal_set, batch_size=args.batch_size,
        shuffle=True, drop_last=True,
        collate_fn=dict_collate_fn,
    )



    num_generate = 10000
    all_samples = []
    all_labels = []
    all_reals = []
    num_samples = 0
    while num_samples < num_generate:
        for normal_batch in normal_loader:
            signals = normal_batch['signals'].to(device=device, dtype=torch.float32) #(batch_size, seq_len, ts_dim)
            attn_mask = normal_batch['attn_mask'].to(device=device, dtype=torch.bool) # (batch_size, seq_len)
            noise_mask = normal_batch['noise_mask'].to(device=device, dtype=torch.long)

            x_occluded = signals * attn_mask.unsqueeze(-1)
            with torch.no_grad():
                samples = model.get_anomaly_samples(
                    x_occluded,
                    noise_mask
                )

            all_samples.append(samples)
            all_labels.append(noise_mask)
            all_reals.append(signals)

            num_samples += samples.shape[0]
            print(f"Generated {num_samples}/{num_generate} ")
            if num_samples >= num_generate:
                break

    all_samples = torch.cat(all_samples, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    all_reals = torch.cat(all_reals, dim=0)

    all_results = {
        'all_samples': all_samples,
        'all_labels': all_labels,
        'all_reals': all_reals,
    }
    torch.save(all_results, f"{args.ckpt_dir}/principle_no_code_impute_samples.pth")


def impute_normal_sample(args):
    model = GenIASModel(
        hidden_layer_sizes=args.hidden_layer_sizes,
        trend_poly=args.trend_poly,
        custom_seas=args.custom_seas,
        use_residual_conn=True,
        delta_min=args.delta_min,
        delta_max=args.delta_max,
        reconstruction_weight=1.0,
        perturbation_weight=args.perturbation_weight,
        kl_weight=args.kl_wt,
        seq_len=args.seq_len,
        feat_dim=args.feature_size,
        latent_dim=args.latent_dim,
        sigma_prior=args.sigma_prior,
    )

    model.load_state_dict(torch.load(f"{args.ckpt_dir}/ckpt.pth"))
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    model.to(device=device)
    model.eval()


    assert args.data_type == "ecg"
    normal_set = ImputationNormalECGDataset(
        raw_data_paths=args.raw_data_paths_train,
        indices_paths=args.indices_paths_train,
        seq_len=args.seq_len,
        one_channel=args.one_channel,
        min_infill_length=args.min_infill_length,
        max_infill_length=args.max_infill_length,
    )

    normal_loader = torch.utils.data.DataLoader(
        normal_set, batch_size=args.batch_size,
        shuffle=True, drop_last=True,
        collate_fn=dict_collate_fn,
    )

    num_generate = 10000
    all_samples = []
    all_labels = []
    all_reals = []
    num_samples = 0
    while num_samples < num_generate:
        for normal_batch in normal_loader:
            signals = normal_batch['signals'].to(device=device, dtype=torch.float32) #(batch_size, seq_len, ts_dim)
            attn_mask = normal_batch['attn_mask'].to(device=device, dtype=torch.bool) # (batch_size, seq_len)
            noise_mask = normal_batch['noise_mask'].to(device=device, dtype=torch.long)

            x_occluded = signals * attn_mask.unsqueeze(-1)
            with torch.no_grad():
                samples = model.get_normal_samples(
                    x_occluded,
                    noise_mask
                )

            all_samples.append(samples)
            all_labels.append(noise_mask)
            all_reals.append(signals)

            num_samples += samples.shape[0]
            print(f"Generated {num_samples}/{num_generate} ")
            if num_samples >= num_generate:
                break

    all_samples = torch.cat(all_samples, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    all_reals = torch.cat(all_reals, dim=0)

    all_results = {
        'all_samples': all_samples,
        'all_labels': all_labels,
        'all_reals': all_reals,
    }
    torch.save(all_results, f"{args.ckpt_dir}/normal_samples.pth")


def impute_sample_non_downstream(args):


    model = GenIASModel(
        hidden_layer_sizes=args.hidden_layer_sizes,
        trend_poly=args.trend_poly,
        custom_seas=args.custom_seas,
        use_residual_conn=True,
        delta_min=args.delta_min,
        delta_max=args.delta_max,
        reconstruction_weight=1.0,
        perturbation_weight=args.perturbation_weight,
        kl_weight=args.kl_wt,
        seq_len=args.seq_len,
        feat_dim=args.feature_size,
        latent_dim=args.latent_dim,
        sigma_prior=args.sigma_prior,
    )

    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(f"{args.ckpt_dir}/ckpt.pth", map_location=device))
    model.to(device=device)
    model.eval()

    anomaly_set = ImputationECGDataset(
        raw_data_paths=args.raw_data_paths_train,
        indices_paths=args.indices_paths_train,
        seq_len=args.seq_len,
        one_channel=args.one_channel,
        max_infill_length=args.max_infill_length,
    )

    anomaly_loader = torch.utils.data.DataLoader(
        anomaly_set,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=dict_collate_fn,
    )

    num_samples_per_input = 5  # ⭐ 每个样本采 5 次

    all_samples = []
    all_labels = []
    all_reals = []

    for batch in anomaly_loader:
        signals = batch['signals'].to(device=device, dtype=torch.float32)  # (batch_size, seq_len, ts_dim)
        attn_mask = batch['attn_mask'].to(device=device, dtype=torch.bool)  # (batch_size, seq_len)
        noise_mask = batch['noise_mask'].to(device=device, dtype=torch.long)

        x_occluded = signals * attn_mask.unsqueeze(-1)
        # -------- 多次 stochastic sampling --------
        batch_samples = []
        with torch.no_grad():
            for _ in range(num_samples_per_input):
                samples = model.get_anomaly_samples(
                    x_occluded,
                    noise_mask
                )
                batch_samples.append(samples.unsqueeze(1))  # (B, 1, T, C)

        # (B, K, T, C)
        batch_samples = torch.cat(batch_samples, dim=1)

        all_samples.append(batch_samples)
        all_labels.append(noise_mask)
        all_reals.append(signals)

    all_samples = torch.cat(all_samples, dim=0)  # (N, K T, C)
    all_labels = torch.cat(all_labels, dim=0)  # (N, T)
    all_reals = torch.cat(all_reals, dim=0)  # (N, T, C)

    all_results = {
        "all_samples": all_samples,
        "all_labels": all_labels,
        "all_reals": all_reals,
    }

    # save_path = f"{args.ckpt_dir}/no_code_impute_samples_non_downstream.pth"
    save_path = f"{args.ckpt_dir}/no_code_impute_samples_non_downstream_train.pth"
    torch.save(all_results, save_path)
    print(f"[Saved] {save_path} | samples shape = {all_samples.shape}")



def genias_train(args):
    os.makedirs(args.ckpt_dir, exist_ok=True)
    save_args_to_jsonl(args, f"{args.ckpt_dir}/config.jsonl")

    model = GenIASModel(
        hidden_layer_sizes=args.hidden_layer_sizes,
        trend_poly=args.trend_poly,
        custom_seas=args.custom_seas,
        use_residual_conn=True,
        delta_min=args.delta_min,
        delta_max=args.delta_max,
        reconstruction_weight=1.0,
        perturbation_weight=args.perturbation_weight,
        kl_weight=args.kl_wt,
        seq_len=args.seq_len,
        feat_dim=args.feature_size,
        latent_dim=args.latent_dim,
        sigma_prior=args.sigma_prior,
    )

    assert args.data_type == "ecg"

    full_set = ImputationNormalECGDataset(
        raw_data_paths=args.raw_data_paths_train,
        indices_paths=args.indices_paths_train,
        seq_len=args.seq_len,
        one_channel=args.one_channel,
        min_infill_length=args.min_infill_length,
        max_infill_length=args.max_infill_length,
    )
    N = len(full_set)
    indices = np.arange(N)

    split = int(0.8 * N)
    train_idx = indices[:split]
    val_idx = indices[split:]

    train_set = Subset(full_set, train_idx)
    val_set = Subset(full_set, val_idx)


    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size,
        shuffle=True, drop_last=True,
        collate_fn = dict_collate_fn,
    )
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=args.batch_size,
        shuffle=False, drop_last=False,
        collate_fn=dict_collate_fn,
    )

    optimizer= torch.optim.Adam(model.parameters(), lr=args.lr)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.8,  # multiply LR by 0.5
        patience=1,  # wait 3 epochs with no improvement
        threshold=1e-4,  # improvement threshold
        min_lr=1e-5,  # min LR clamp
    )

    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    trainer = GenIAS_Trainer(
        optimizer=optimizer,
        scheduler=scheduler,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        max_epochs=args.max_epochs,
        device=device,
        save_dir=args.ckpt_dir,
        wandb_run_name=args.wandb_run,
        wandb_project_name=args.wandb_project,
        grad_clip_norm=args.grad_clip_norm,
        early_stop=args.early_stop,
        patience=args.patience,
    )

    trainer.train(config=vars(args))


def main():
    args = get_args()

    if args.what_to_do == "genias_trian":
        genias_train(args)
    elif args.what_to_do == "impute_sample":
        impute_sample(args)
    elif args.what_to_do == "impute_normal_sample":
        impute_normal_sample(args)
    elif args.what_to_do == "principle_impute_sample":
        principle_impute_sample(args)
    elif args.what_to_do == "impute_sample_non_downstream":
        impute_sample_non_downstream(args)
    else:
        raise NotImplementedError


if __name__ == "__main__":
    main()
