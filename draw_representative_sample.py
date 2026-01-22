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




# def plot_with_confidence_interval(samples, title="Posterior Samples"):
#     # 1. 转换为 Numpy 并提取对应通道
#     if isinstance(samples, torch.Tensor):
#         data = samples.detach().cpu().numpy()[:, :, 0]  # (K, T)
#     else:
#         data = samples[:, :, 0]
#
#     # 2. 计算统计量
#     mu = np.mean(data, axis=0)  # 均值 (T,)
#     std = np.std(data, axis=0)  # 标准差 (T,)
#
#     # 定义置信区间（1.96倍标准差对应约95%置信区间）
#     upper_95 = mu + 1.96 * std
#     lower_95 = mu - 1.96 * std
#
#     # 定义较窄的区间（1倍标准差对应约68%置信区间）
#     upper_68 = mu + std
#     lower_68 = mu - std
#
#     t = np.arange(len(mu))
#
#     plt.figure(figsize=(3.6, 1.8))
#
#     # 画出 95% 置信区间 (浅色)
#     # plt.fill_between(t, lower_95, upper_95, color='#8C2F39', alpha=0.2, label='95% Confidence Interval')
#
#     # 画出 68% 置信区间 (深色)
#     # plt.fill_between(t, lower_68, upper_68, color='#8C2F39', alpha=0.4, label='68% Confidence Interval')
#
#     # 画出均值线
#     plt.plot(t, mu, color='#8C2F39', lw=2, label='Mean Prediction')
#
#     # 如果你想保留一些原始采样的影子（可选，画3-5条作为参考）
#     # for i in range(min(3, data.shape[0])):
#     for i in range(data.shape[0]):
#         plt.plot(t, data[i], color='#8C2F39', alpha=0.1, lw=0.5, )
#
#     plt.title("Generated Posterior Anomaly", fontsize=8)
#     plt.xlabel("Time Steps", fontsize=8)
#     plt.ylabel("Normalized Value", fontsize=8)
#     plt.legend(fontsize=8)
#     plt.grid(True, alpha=0.3)
#     plt.tight_layout()
#     # plt.show()
#     plt.savefig(title + ".pdf")




def plot_with_confidence_interval(samples, title="Posterior Samples"):
    if isinstance(samples, torch.Tensor):
        data = samples.detach().cpu().numpy()[:, :, 0]
    else:
        data = samples[:, :, 0]

    mu = np.mean(data, axis=0)
    std = np.std(data, axis=0)

    t = np.arange(len(mu))
    normal_mask = std < 1e-4

    plt.figure(figsize=(3.6, 1.8))

    # -------- helper: 找连续区间 --------
    def find_segments(mask):
        segments = []
        start = None
        for i, v in enumerate(mask):
            if v and start is None:
                start = i
            if not v and start is not None:
                segments.append((start, i))
                start = None
        if start is not None:
            segments.append((start, len(mask)))
        return segments

    normal_segments = find_segments(normal_mask)
    anomaly_segments = find_segments(~normal_mask)

    # -------- 画均值线（按区间） --------
    for s, e in normal_segments:
        plt.plot(
            t[s:e+1],
            mu[s:e+1],
            color="#3B2F2F",
            lw=2,
            label="Normal Context" if s == normal_segments[0][0] else None,
        )

    for s, e in anomaly_segments:
        plt.plot(
            t[s:e+1],
            mu[s:e+1],
            color="#8C2F39",
            lw=2,
            label="Anomaly Segment" if s == anomaly_segments[0][0] else None,
        )

    # -------- 原始采样（shadow，同样按区间） --------
    for i in range(data.shape[0]):
        for s, e in normal_segments:
            plt.plot(
                t[s:e+1],
                data[i][s:e+1],
                color="#3B2F2F",
                alpha=0.08,
                lw=0.5,
            )
        for s, e in anomaly_segments:
            plt.plot(
                t[s:e+1],
                data[i][s:e+1],
                color="#8C2F39",
                alpha=0.10,
                lw=0.5,
            )

    plt.title("Generated Anomaly", fontsize=8)
    plt.xlabel("Time Steps", fontsize=8)
    plt.ylabel("Normalized Value", fontsize=8)
    plt.legend(fontsize=7, frameon=False, loc="lower right", bbox_to_anchor=(0.98, -0.05))
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(title + ".pdf", bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    names = [
        "550_600_anomaly_pattern25.pt",
        "550_600_anomaly_pattern91.pt",
        "550_600_anomaly_pattern82.pt",
        "550_600_anomaly_pattern114.pt",
        "200_250_anomaly_pattern22.pt",
        "200_250_anomaly_pattern41.pt",
    ]
    for name in names:
        samples = torch.load(name, map_location='cpu')
        plot_with_confidence_interval(samples, title=name)