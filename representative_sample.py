import torch
import math
import matplotlib.pyplot as plt
from generation_models import DSPFlow
from dataset_utils import ImputationNormalECGDataset, NoContextAnomalyECGDataset
from dataset_utils import ImputationECGDataset, NoContextNormalECGDataset
from dataset_utils import ImputationNormalECGDatasetForSample

from dataset_utils import ImputationNormalERCOTDataset, NoContextAnomalyERCOTDataset
from dataset_utils import ImputationERCOTDataset, NoContextNormalERCOTDataset
from dataset_utils import PredictionECGDataset, PredictionNormalECGDataset
import numpy as np


def dict_collate_fn(batch):
    out = {}
    for key in batch[0].keys():
        out[key] = torch.stack([sample[key] for sample in batch], dim=0)
    return out


def plot_with_confidence_interval(samples, title="Posterior Samples"):
    # 1. 转换为 Numpy 并提取对应通道
    if isinstance(samples, torch.Tensor):
        data = samples.detach().cpu().numpy()[:, :, 0]  # (K, T)
    else:
        data = samples[:, :, 0]

    # 2. 计算统计量
    mu = np.mean(data, axis=0)  # 均值 (T,)
    std = np.std(data, axis=0)  # 标准差 (T,)

    # 定义置信区间（1.96倍标准差对应约95%置信区间）
    upper_95 = mu + 1.96 * std
    lower_95 = mu - 1.96 * std

    # 定义较窄的区间（1倍标准差对应约68%置信区间）
    upper_68 = mu + std
    lower_68 = mu - std

    t = np.arange(len(mu))

    plt.figure(figsize=(4, 2))

    # 画出 95% 置信区间 (浅色)
    plt.fill_between(t, lower_95, upper_95, color='#8C2F39', alpha=0.15, label='95% Confidence Interval')

    # 画出 68% 置信区间 (深色)
    plt.fill_between(t, lower_68, upper_68, color='#8C2F39', alpha=0.3, label='68% Confidence Interval')

    # 画出均值线
    plt.plot(t, mu, color='#8C2F39', lw=2, label='Mean Prediction')

    # 如果你想保留一些原始采样的影子（可选，画3-5条作为参考）
    # for i in range(min(3, data.shape[0])):
    for i in range(data.shape[0]):
        plt.plot(t, data[i], color='#8C2F39', alpha=0.3, lw=0.5, linestyle='--')

    plt.title(title)
    plt.xlabel("Time Steps")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def sample_550_600():
    data_path = "./dataset_utils/real_anomaly_data_test/mitdb_train.pth"
    all_data = torch.load(data_path, map_location='cpu')
    real_data = all_data['all_samples']
    anomaly_labels = all_data['all_labels']

    all_representatives = []
    anomaly_indices = [25, 91, 82,114,118]
    for i in anomaly_indices:
        anomaly_index = torch.where(anomaly_labels[i] != 0)[0]
        anomaly_segment = real_data[i,anomaly_index]
        plt.figure(figsize=(3.6, 1.8))
        plt.plot(anomaly_segment[:,0], linewidth=3, color="#8C2F39")
        plt.title(f"Given Anomaly Segment", fontsize=8)
        plt.xlabel("Time Steps", fontsize=8)
        plt.ylabel("Normalized Value", fontsize=8)
        plt.tight_layout()
        plt.grid(True, alpha=0.3)

        plt.savefig(f"anomaly-segment-{i}.pdf")
        plt.close()


        pad_to_800 = torch.zeros(800, 2)
        attn_mask = torch.zeros(800)
        pad_to_800[:len(anomaly_segment)] = anomaly_segment
        attn_mask[:len(anomaly_segment)] = 1
        print(len(anomaly_segment))
        all_representatives.append(
            {"signal": pad_to_800,  "attn_mask": attn_mask}
        )

    model = DSPFlow(
            seq_length=1000,
            vqvae_seq_len=800,
            num_codes=500,
            feature_size=2,
            n_layer_enc=4,
            n_layer_dec=4,
            d_model=64,
            n_heads=4,
            mlp_hidden_times=4,
            vqvae_ckpt="/Users/zhc/Documents/formal_experiment/mitdb_two_channels/dsp_flow_mixed_K500/vqvae_save_path/vqvae.pt"
        )

    ckpt = torch.load(
        f"/Users/zhc/Documents/formal_experiment/mitdb_two_channels/dsp_flow_mixed_K500/impute_finetune_ckpt_lr1e-4/ckpt.pth",
        map_location="cpu"
    )
    model.load_state_dict(ckpt)
    model.eval()

    posterior_list = []

    for anomaly_representative in all_representatives:
        with torch.no_grad():
            embed = model.vqvae.encode(
                anomaly_representative["signal"].unsqueeze(0),
                anomaly_representative["attn_mask"].unsqueeze(0),
            )
        posterior_list.append(embed)




    DATA_PATHS = ["./dataset_utils/ECG_datasets/raw_data/106.npz"]
    NORMAL_INDICES_FOR_SAMPLE = ["./dataset_utils/ECG_datasets/indices/slide_windows_106npz/train/normal_1000.jsonl"]
    EVENT_LABELS_PATHS = ["./dataset_utils/ECG_datasets/indices/slide_windows_106npz/train/event_label.npy"]
    LEN_WHOLE = 1000
    MAX_LEN_ANOMALY = 600
    MIN_LEN_ANOMALY = 550

    normal_set = ImputationNormalECGDatasetForSample(
        raw_data_paths=DATA_PATHS,
        indices_paths=NORMAL_INDICES_FOR_SAMPLE,
        event_labels_paths=EVENT_LABELS_PATHS,
        seq_len=LEN_WHOLE,
        one_channel=0,
        max_infill_length=MAX_LEN_ANOMALY,
        min_infill_length=MIN_LEN_ANOMALY,
    )

    normal_loader = torch.utils.data.DataLoader(
            normal_set,
            batch_size=1,
            shuffle=True,
            drop_last=True,
            collate_fn=dict_collate_fn,
        )

    # for batch_id, normal_batch in enumerate(normal_loader):
    #
    #     signals = normal_batch["signals"].to(dtype=torch.float32)  # (B, T, C)
    #     attn_mask = normal_batch["attn_mask"].to(dtype=torch.bool)
    #     noise_mask = normal_batch["noise_mask"].to(dtype=torch.long)
    #
    #     B, T, C = signals.shape
    #     K = 40
    #
    #     signals_big = signals.expand(K, -1, -1)
    #     attn_mask_big = attn_mask.expand(K, -1)
    #     noise_mask_big = noise_mask.expand(K, -1)
    #
    #     for i_posterior, posterior in enumerate(posterior_list):
    #         posterior_big = posterior.expand(K, -1, -1)
    #
    #         samples_big = model.posterior_impute(
    #             signals_big,
    #             posterior_big,
    #             attn_mask=attn_mask_big,
    #             noise_mask=noise_mask_big,
    #         )
    #         torch.save(samples_big, f"550_600_anomaly_pattern{anomaly_indices[i_posterior]}.pt" )
    #
    #         # plot_with_confidence_interval(samples_big, title=f"{anomaly_indices[i_posterior]}")
    #         # plt.figure(figsize=(4, 2))
    #         # for sample in samples_big:
    #         #     plt.plot(sample[:,0], color="red")
    #         # plt.title(f"{anomaly_indices[i_posterior]}")
    #         # plt.show()
    #
    #     break
    #     print("123")


def sample_200_250():
    data_path = "./dataset_utils/real_anomaly_data_test/mitdb_train.pth"
    all_data = torch.load(data_path, map_location='cpu')
    real_data = all_data['all_samples']
    anomaly_labels = all_data['all_labels']

    all_representatives = []
    anomaly_indices = [22, 41]
    for i in anomaly_indices:
        anomaly_index = torch.where(anomaly_labels[i] != 0)[0]
        anomaly_segment = real_data[i, anomaly_index]
        plt.figure(figsize=(3.6, 1.8))
        plt.plot(anomaly_segment[:, 0], linewidth=3, color="#8C2F39")
        plt.title(f"Given Anomaly Segment", fontsize=8)
        plt.xlabel("Time Steps", fontsize=8)
        plt.ylabel("Normalized Value", fontsize=8)
        plt.tight_layout()
        plt.grid(True, alpha=0.3)

        plt.savefig(f"anomaly-segment-{i}.pdf")
        plt.close()

        pad_to_800 = torch.zeros(800, 2)
        attn_mask = torch.zeros(800)
        pad_to_800[:len(anomaly_segment)] = anomaly_segment
        attn_mask[:len(anomaly_segment)] = 1
        print(len(anomaly_segment))
        all_representatives.append(
            {"signal": pad_to_800, "attn_mask": attn_mask}
        )

    model = DSPFlow(
        seq_length=1000,
        vqvae_seq_len=800,
        num_codes=500,
        feature_size=2,
        n_layer_enc=4,
        n_layer_dec=4,
        d_model=64,
        n_heads=4,
        mlp_hidden_times=4,
        vqvae_ckpt="/Users/zhc/Documents/formal_experiment/mitdb_two_channels/dsp_flow_mixed_K500/vqvae_save_path/vqvae.pt"
    )

    ckpt = torch.load(
        f"/Users/zhc/Documents/formal_experiment/mitdb_two_channels/dsp_flow_mixed_K500/impute_finetune_ckpt_lr1e-4/ckpt.pth",
        map_location="cpu"
    )
    model.load_state_dict(ckpt)
    model.eval()

    posterior_list = []

    for anomaly_representative in all_representatives:
        with torch.no_grad():
            embed = model.vqvae.encode(
                anomaly_representative["signal"].unsqueeze(0),
                anomaly_representative["attn_mask"].unsqueeze(0),
            )
        posterior_list.append(embed)

    DATA_PATHS = ["./dataset_utils/ECG_datasets/raw_data/106.npz"]
    NORMAL_INDICES_FOR_SAMPLE = ["./dataset_utils/ECG_datasets/indices/slide_windows_106npz/train/normal_1000.jsonl"]
    EVENT_LABELS_PATHS = ["./dataset_utils/ECG_datasets/indices/slide_windows_106npz/train/event_label.npy"]
    LEN_WHOLE = 1000
    MAX_LEN_ANOMALY = 300
    MIN_LEN_ANOMALY = 250

    normal_set = ImputationNormalECGDatasetForSample(
        raw_data_paths=DATA_PATHS,
        indices_paths=NORMAL_INDICES_FOR_SAMPLE,
        event_labels_paths=EVENT_LABELS_PATHS,
        seq_len=LEN_WHOLE,
        one_channel=0,
        max_infill_length=MAX_LEN_ANOMALY,
        min_infill_length=MIN_LEN_ANOMALY,
    )

    normal_loader = torch.utils.data.DataLoader(
        normal_set,
        batch_size=1,
        shuffle=True,
        drop_last=True,
        collate_fn=dict_collate_fn,
    )

    # for batch_id, normal_batch in enumerate(normal_loader):
    #
    #     signals = normal_batch["signals"].to(dtype=torch.float32)  # (B, T, C)
    #     attn_mask = normal_batch["attn_mask"].to(dtype=torch.bool)
    #     noise_mask = normal_batch["noise_mask"].to(dtype=torch.long)
    #
    #     B, T, C = signals.shape
    #     K = 40
    #
    #     signals_big = signals.expand(K, -1, -1)
    #     attn_mask_big = attn_mask.expand(K, -1)
    #     noise_mask_big = noise_mask.expand(K, -1)
    #
    #     for i_posterior, posterior in enumerate(posterior_list):
    #         posterior_big = posterior.expand(K, -1, -1)
    #
    #         samples_big = model.posterior_impute(
    #             signals_big,
    #             posterior_big,
    #             attn_mask=attn_mask_big,
    #             noise_mask=noise_mask_big,
    #         )
    #         torch.save(samples_big, f"200_250_anomaly_pattern{anomaly_indices[i_posterior]}.pt" )
    #         # plot_with_confidence_interval(samples_big, title=f"{anomaly_indices[i_posterior]}")
    #         # plt.figure(figsize=(4, 2))
    #         # for sample in samples_big:
    #         #     plt.plot(sample[:,0], color="red")
    #         # plt.title(f"{anomaly_indices[i_posterior]}")
    #         # plt.show()
    #
    #         # print("123")
    #     break


if __name__ == "__main__":
    sample_550_600()
    sample_200_250()