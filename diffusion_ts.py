from Trainers import DiffusionTSTrainer
from generation_models import Diffusion_TS
from dataset_utils import ImputationNormalECGDataset, ImputationNormalECGDatasetForSample
from dataset_utils import ImputationECGDataset
import argparse
import torch
import json
import os
import numpy as np
from evaluation_utils import calculate_robustTAD
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
            "no_code_imputation_train",
            "impute_sample",
            "principle_impute_sample",
            "impute_sample_non_downstream",
            "anomaly_evaluate"
        ],
        help="what to do"
    )

    """time series general parameters"""
    parser.add_argument("--data_type", type=str, required=True)
    parser.add_argument("--seq_len", type=int, required=True)
    parser.add_argument("--feature_size", type=int, required=True)
    parser.add_argument("--one_channel", type=int, required=True)

    """model parameters"""
    parser.add_argument("--n_layer_enc", type=int, required=True)
    parser.add_argument("--n_layer_dec", type=int, required=True)
    parser.add_argument("--d_model", type=int, required=True)
    parser.add_argument("--n_heads", type=int, required=True)

    """data parameters"""
    parser.add_argument("--raw_data_paths_train", type=json.loads, required=True)
    parser.add_argument("--raw_data_paths_test", type=json.loads, required=True)
    parser.add_argument("--event_labels_paths_train", type=json.loads, required=True)
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



def no_code_imputation_train(args):
    os.makedirs(args.ckpt_dir, exist_ok=True)
    save_args_to_jsonl(args, f"{args.ckpt_dir}/config.jsonl")

    model = Diffusion_TS(
        seq_length=args.seq_len,
        feature_size=args.feature_size,
        n_layer_enc=args.n_layer_enc,
        n_layer_dec=args.n_layer_dec,
        d_model=args.d_model,
        n_heads=args.n_heads,
        mlp_hidden_times=4,
    )

    assert args.data_type == "ecg"

    train_set = ImputationECGDataset(
        raw_data_paths=args.raw_data_paths_train,
        indices_paths=args.indices_paths_train,
        seq_len=args.seq_len,
        one_channel=args.one_channel,
        max_infill_length=args.max_infill_length,
    )

    val_set = ImputationECGDataset(
        raw_data_paths=args.raw_data_paths_test,
        indices_paths=args.indices_paths_test,
        seq_len=args.seq_len,
        one_channel=args.one_channel,
        max_infill_length=args.max_infill_length,
    )

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
    trainer = DiffusionTSTrainer(
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
        grad_accum_steps=args.grad_accum_steps,
        early_stop=args.early_stop,
        patience=args.patience,
    )

    trainer.no_code_imputation_train(config=vars(args))


def no_code_impute_sample(args):
    model = Diffusion_TS(
        seq_length=args.seq_len,
        feature_size=args.feature_size,
        n_layer_enc=args.n_layer_enc,
        n_layer_dec=args.n_layer_dec,
        d_model=args.d_model,
        n_heads=args.n_heads,
        mlp_hidden_times=4,
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

    # normal_set = ImputationNormalECGDatasetForSample(
    #     raw_data_paths=args.raw_data_paths_train,
    #     indices_paths=args.indices_paths_train,
    #     event_labels_paths=args.event_labels_paths_train,
    #     seq_len=args.seq_len,
    #     one_channel=args.one_channel,
    #     min_infill_length=args.min_infill_length,
    #     max_infill_length=args.max_infill_length,
    # )

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

            with torch.no_grad():
                samples = model.no_code_impute(
                    signals,
                    attn_mask=attn_mask,
                    noise_mask=noise_mask
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


def principle_no_code_impute_sample(args):
    model = Diffusion_TS(
        seq_length=args.seq_len,
        feature_size=args.feature_size,
        n_layer_enc=args.n_layer_enc,
        n_layer_dec=args.n_layer_dec,
        d_model=args.d_model,
        n_heads=args.n_heads,
        mlp_hidden_times=4,
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

            with torch.no_grad():
                samples = model.no_code_impute(
                    signals,
                    attn_mask=attn_mask,
                    noise_mask=noise_mask
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


def no_code_impute_sample_non_downstream(args):
    model = Diffusion_TS(
        seq_length=args.seq_len,
        feature_size=args.feature_size,
        n_layer_enc=args.n_layer_enc,
        n_layer_dec=args.n_layer_dec,
        d_model=args.d_model,
        n_heads=args.n_heads,
        mlp_hidden_times=4,
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
        signals = batch['signals'].to(device, dtype=torch.float32)  # (B, T, C)
        attn_mask = batch['attn_mask'].to(device, dtype=torch.bool)  # (B, T)
        noise_mask = batch['noise_mask'].to(device, dtype=torch.long)  # (B, T)

        # -------- 多次 stochastic sampling --------
        batch_samples = []
        with torch.no_grad():
            for _ in range(num_samples_per_input):
                samples = model.no_code_impute(
                    signals,
                    attn_mask=attn_mask,
                    noise_mask=noise_mask
                )  # (B, T, C)
                batch_samples.append(samples.unsqueeze(1))  # (B, 1, T, C)

        # (B, K, T, C)
        batch_samples = torch.cat(batch_samples, dim=1)

        all_samples.append(batch_samples)
        all_labels.append(noise_mask)
        all_reals.append(signals)

    # -------- 拼接所有 batch --------
    all_samples = torch.cat(all_samples, dim=0)  # (N, K T, C)
    all_labels = torch.cat(all_labels, dim=0)  # (N, T)
    all_reals = torch.cat(all_reals, dim=0)  # (N, T, C)

    all_results = {
        "all_samples": all_samples,
        "all_labels": all_labels,
        "all_reals": all_reals,
    }

    save_path = f"{args.ckpt_dir}/no_code_impute_samples_non_downstream.pth"
    torch.save(all_results, save_path)
    print(f"[Saved] {save_path} | samples shape = {all_samples.shape}")


def anomaly_evaluate(args):
    device = torch.device(f"cuda:{args.gpu_id}")

    if args.data_type == "ecg":
        real_set = ImputationECGDataset(
            raw_data_paths=args.raw_data_paths_train,
            indices_paths=args.indices_paths_test,
            seq_len=args.seq_len,
            one_channel=args.one_channel,
            max_infill_length=args.max_infill_length,
        )
    else:
        raise ValueError(f"Unknown data_type {args.data_type}")

    real_data = []
    real_labels = []
    for which_list, which_index in real_set.global_index:
        ts_start = real_set.index_lines_list[which_list][which_index]["ts_start"]
        ts_end = real_set.index_lines_list[which_list][which_index]["ts_end"]
        anomaly_start = real_set.index_lines_list[which_list][which_index]["anomaly_start"]
        anomaly_end = real_set.index_lines_list[which_list][which_index]["anomaly_end"]

        relative_anomaly_start = anomaly_start - ts_start
        relative_anomaly_end = anomaly_end - ts_start
        real_datum = torch.from_numpy(real_set.normed_signal_list[which_list][ts_start:ts_end])
        real_label = torch.zeros(len(real_datum)).to(device=device)
        real_label[relative_anomaly_start:relative_anomaly_end] = 1

        real_data.append(real_datum.unsqueeze(0))
        real_labels.append(real_label.unsqueeze(0))

    real_data = torch.cat(real_data, dim=0).to(device=device)
    if args.one_channel:
        real_data = real_data[:,:,:1]
    real_labels = torch.cat(real_labels, dim=0).to(device=device)


    all_anomalies = torch.load(args.generated_path, map_location=device)


    gen_data = all_anomalies['all_samples']
    gen_labels = all_anomalies['all_labels']

    # ---- Step 1: 找出含 NaN 的样本 ----
    nan_mask = torch.isnan(gen_data).any(dim=(1, 2))  # True 表示该样本含 NaN

    print("Samples containing NaN:", nan_mask.sum().item(), "/", gen_data.size(0))
    # ---- Step 2: 删除这些样本 ----
    gen_data = gen_data[~nan_mask]
    gen_labels = gen_labels[~nan_mask]


    # initialize a dict for result

    output_record = {
        "args": vars(args),
    }

    # ------------------
    # calculate_robustTAD
    # ------------------
    precisions = []
    recalls = []
    f1s = []
    normal_accuracies = []
    anomaly_accuracies = []
    for _ in range(5):
        random_indices = torch.randperm(len(gen_data))[:10000]
        sampled_gen_data = gen_data[random_indices]
        sampled_gen_labels = gen_labels[random_indices]

        print("real_data.shape:", real_data.shape)
        print("real_labels.shape:", real_labels.shape)
        print("gen_data.shape:", gen_data.shape)
        print("gen_labels.shape:", gen_labels.shape)

        normal_accuracy, anomaly_accuracy, precision, recall, f1 = calculate_robustTAD(
            anomaly_weight=1.0,
            feature_size=args.feature_size,
            ori_data=real_data,
            ori_labels=real_labels,
            gen_data=sampled_gen_data,
            gen_labels=sampled_gen_labels,
            device=device,
            lr=1e-5,
            max_epochs=2000,
            batch_size=64,
            patience=20)
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        normal_accuracies.append(normal_accuracy)
        anomaly_accuracies.append(anomaly_accuracy)

    mean_precision = np.mean(precisions)
    mean_recall = np.mean(recalls)
    mean_f1 = np.mean(f1s)
    mean_normal_accuracy = np.mean(normal_accuracies)
    mean_anomaly_accuracy = np.mean(anomaly_accuracies)

    std_precision = np.std(precisions)
    std_recall = np.std(recalls)
    std_f1 = np.std(f1s)
    std_normal_accuracy = np.std(normal_accuracies)
    std_anomaly_accuracy = np.std(anomaly_accuracies)

    print(f"precision: {mean_precision}+-{std_precision}")
    print(f"recall: {mean_recall}+-{std_recall}")
    print(f"f1: {mean_f1}+-{std_f1}")
    print(f"normal_accuracy: {mean_normal_accuracy}+-{std_normal_accuracy}")
    print(f"anomaly_accuracy: {mean_anomaly_accuracy}+-{std_anomaly_accuracy}")


    result = {
        "precision_mean": float(mean_precision),
        "precision_std": float(std_precision),
        "recall_mean": float(mean_recall),
        "recall_std": float(std_recall),
        "f1_mean": float(mean_f1),
        "f1_std": float(std_f1),
        "normal_accuracy_mean": float(mean_normal_accuracy),
        "normal_accuracy_std": float(std_normal_accuracy),
        "anomaly_accuracy_mean": float(mean_anomaly_accuracy),
        "anomaly_accuracy_std": float(std_anomaly_accuracy),
    }
    output_record.update({"result_robustTAD": result})

    # ------------------
    # calculate MLSTM-FCN
    # ------------------
    # precisions = []
    # recalls = []
    # f1s = []
    # normal_accuracies = []
    # anomaly_accuracies = []
    # for _ in range(5):
    #     random_indices = torch.randperm(len(gen_data))[:10000]
    #     sampled_gen_data = gen_data[random_indices]
    #     sampled_gen_labels = gen_labels[random_indices]
    #
    #     print("real_data.shape:", real_data.shape)
    #     print("real_labels.shape:", real_labels.shape)
    #     print("gen_data.shape:", gen_data.shape)
    #     print("gen_labels.shape:", gen_labels.shape)
    #
    #     normal_accuracy, anomaly_accuracy, precision, recall, f1 = calculate_MLSTM_FCN(
    #         anomaly_weight=1.0,
    #         feature_size=args.feature_size,
    #         ori_data=real_data,
    #         ori_labels=real_labels,
    #         gen_data=sampled_gen_data,
    #         gen_labels=sampled_gen_labels,
    #         device=device,
    #         lr=1e-5,
    #         max_epochs=2000,
    #         batch_size=64,
    #         patience=20)
    #     precisions.append(precision)
    #     recalls.append(recall)
    #     f1s.append(f1)
    #     normal_accuracies.append(normal_accuracy)
    #     anomaly_accuracies.append(anomaly_accuracy)
    #
    # mean_precision = np.mean(precisions)
    # mean_recall = np.mean(recalls)
    # mean_f1 = np.mean(f1s)
    # mean_normal_accuracy = np.mean(normal_accuracies)
    # mean_anomaly_accuracy = np.mean(anomaly_accuracies)
    #
    # std_precision = np.std(precisions)
    # std_recall = np.std(recalls)
    # std_f1 = np.std(f1s)
    # std_normal_accuracy = np.std(normal_accuracies)
    # std_anomaly_accuracy = np.std(anomaly_accuracies)
    #
    # print(f"precision: {mean_precision}+-{std_precision}")
    # print(f"recall: {mean_recall}+-{std_recall}")
    # print(f"f1: {mean_f1}+-{std_f1}")
    # print(f"normal_accuracy: {mean_normal_accuracy}+-{std_normal_accuracy}")
    # print(f"anomaly_accuracy: {mean_anomaly_accuracy}+-{std_anomaly_accuracy}")
    #
    #
    # result = {
    #     "precision_mean": float(mean_precision),
    #     "precision_std": float(std_precision),
    #     "recall_mean": float(mean_recall),
    #     "recall_std": float(std_recall),
    #     "f1_mean": float(mean_f1),
    #     "f1_std": float(std_f1),
    #     "normal_accuracy_mean": float(mean_normal_accuracy),
    #     "normal_accuracy_std": float(std_normal_accuracy),
    #     "anomaly_accuracy_mean": float(mean_anomaly_accuracy),
    #     "anomaly_accuracy_std": float(std_anomaly_accuracy),
    # }
    # output_record.update({"result_MLSTM_FCN": result})




    save_path = os.path.join(args.ckpt_dir, f"evaluation_results.jsonl")
    # os.makedirs(args.generated_path, exist_ok=True)

    with open(save_path, "a") as f:
        f.write(json.dumps(output_record) + "\n")



def main():
    args = get_args()
    if args.what_to_do == "no_code_imputation_train":
        no_code_imputation_train(args)
    elif args.what_to_do == "impute_sample":
        no_code_impute_sample(args)
    elif args.what_to_do == "principle_impute_sample":
        principle_no_code_impute_sample(args)
    elif args.what_to_do == "impute_sample_non_downstream":
        no_code_impute_sample_non_downstream(args)
    elif args.what_to_do == "anomaly_evaluate":
        anomaly_evaluate(args)
    else:
        raise NotImplementedError


if __name__ == "__main__":
    main()
