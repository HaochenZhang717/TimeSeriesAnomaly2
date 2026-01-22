import matplotlib.pyplot as plt
import torch
import numpy as np
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from momentfm import MOMENTPipeline


def show_ours():
    model = MOMENTPipeline.from_pretrained(
        "AutonLab/MOMENT-1-small",
        model_kwargs={"task_name": "embedding"},
    )
    model.init()
    model.eval()

    our_results = torch.load(
        "/Users/zhc/Documents/formal_experiment/mitdb_two_channels/dsp_flow_mixed_K500/impute_finetune_ckpt_lr1e-4/posterior_impute_samples_non_downstream_test.pth",
        map_location="cpu"
    )

    samples = our_results["all_samples"]
    reals = our_results["all_reals"]
    labels = our_results["all_labels"]
    N = len(samples)
    sample_embeds = []
    real_embeds = []
    for i in range(N):
        sample_anomaly_segments = samples[i]
        print(sample_anomaly_segments.shape)
        anomaly_idx = torch.where(labels[i] == 1)[0]
        sample_anomaly_segments = samples[i][:, anomaly_idx, :].permute(0, 2, 1)
        real_anomaly_segments = reals[i].unsqueeze(0)[:, anomaly_idx, :].permute(0, 2, 1)
        # print(sample_anomaly_segments.shape)
        with torch.no_grad():
            emb = model(x_enc=torch.cat([sample_anomaly_segments, real_anomaly_segments], dim=0), reduction="mean").embeddings
        emb_samples = emb[:-1]
        emb_reals = emb[-1].unsqueeze(0)
        # print(i)
        sample_embeds.append(emb_samples)
        real_embeds.append(emb_reals)
    return torch.cat(sample_embeds, dim=0), torch.cat(real_embeds, dim=0)

def show_no_code():
    model = MOMENTPipeline.from_pretrained(
        "AutonLab/MOMENT-1-small",
        model_kwargs={"task_name": "embedding"},
    )
    model.init()
    model.eval()


    our_results = torch.load(
        "/Users/zhc/Documents/formal_experiment/mitdb_two_channels/dsp_flow_no_code/no_code_impute_finetune_ckpt_lr1e-4/no_code_impute_samples_non_downstream.pth",
        map_location="cpu"
    )


    samples = our_results["all_samples"]
    reals = our_results["all_reals"]
    labels = our_results["all_labels"]
    N = len(samples)
    sample_embeds = []
    real_embeds = []
    for i in range(N):
        sample_anomaly_segments = samples[i]
        # print(sample_anomaly_segments.shape)
        anomaly_idx = torch.where(labels[i] == 1)[0]
        sample_anomaly_segments = samples[i][:, anomaly_idx, :].permute(0, 2, 1)
        real_anomaly_segments = reals[i].unsqueeze(0)[:, anomaly_idx, :].permute(0, 2, 1)
        # print(sample_anomaly_segments.shape)
        with torch.no_grad():
            emb = model(x_enc=torch.cat([sample_anomaly_segments, real_anomaly_segments], dim=0), reduction="mean").embeddings
        emb_samples = emb[:-1]
        emb_reals = emb[-1].unsqueeze(0)
        # print(i)
        sample_embeds.append(emb_samples)
        real_embeds.append(emb_reals)
    return torch.cat(sample_embeds, dim=0), torch.cat(real_embeds, dim=0)

def plot_three_way_diversity(real_tensor, ours_tensor, nocode_tensor, title="Data Diversity Comparison"):
    """
    输入:
        real_tensor: 真实数据的 Tensor/Array (N, d)
        ours_tensor: 我们方法生成的 Tensor/Array (N, d)
        nocode_tensor: 对照组生成的 Tensor/Array (N, d)
    """

    # 1. 辅助函数：统一转换为 Numpy
    def to_numpy(t):
        if isinstance(t, torch.Tensor):
            return t.detach().cpu().numpy()
        return np.array(t)

    real_np = to_numpy(real_tensor)
    ours_np = to_numpy(ours_tensor)
    nocode_np = to_numpy(nocode_tensor)

    # 2. 合并数据并记录索引
    # 顺序：0: Real, 1: Ours, 2: No Code
    combined_data = np.concatenate([real_np, ours_np, nocode_np], axis=0)

    n_real = len(real_np)
    n_ours = len(ours_np)
    n_nocode = len(nocode_np)

    labels = np.array([0] * n_real + [1] * n_ours + [2] * n_nocode)

    # 3. 标准化
    scaler = StandardScaler()
    combined_scaled = scaler.fit_transform(combined_data)

    # 4. 降维计算
    # PCA
    pca_res = PCA(n_components=2).fit_transform(combined_scaled)

    # t-SNE (perplexity 调整为总样本量相关的数值)
    total_n = len(combined_data)
    tsne_res = TSNE(n_components=2, perplexity=min(30, total_n // 4), random_state=42).fit_transform(combined_scaled)

    # 5. 绘图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle(title, fontsize=18, fontweight='bold')

    # 定义颜色方案 (深蓝、鲜红、淡橙)
    # 真实数据通常用冷色调，生成数据用对比鲜明的色调
    colors = ['#2c3e50', '#e74c3c', '#f39c12']
    target_names = ['Real Data', 'Ours (Proposed)', 'No Code (Baseline)']
    alphas = [0.4, 0.7, 0.6]  # 降低 Real 的透明度，方便看生成数据覆盖情况
    sizes = [40, 60, 50]

    for i in range(3):
        mask = (labels == i)
        # PCA
        ax1.scatter(pca_res[mask, 0], pca_res[mask, 1],
                    c=colors[i], label=target_names[i],
                    alpha=alphas[i], edgecolors='w', s=sizes[i], linewidth=0.5)
        # t-SNE
        ax2.scatter(tsne_res[mask, 0], tsne_res[mask, 1],
                    c=colors[i], label=target_names[i],
                    alpha=alphas[i], edgecolors='w', s=sizes[i], linewidth=0.5)

    ax1.set_title('PCA: Global Distribution Coverage', fontsize=14)
    ax1.grid(True, linestyle='--', alpha=0.3)
    ax1.legend()

    ax2.set_title('t-SNE: Local Structure & Diversity', fontsize=14)
    ax2.grid(True, linestyle='--', alpha=0.3)
    ax2.legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def calculate_v_score(tensor):
    """
    计算样本在时间维度上的平均方差。
    输入形状: (N, d) -> N个样本, d个特征点
    """
    # 计算每个维度(时间点)在N个样本上的方差
    variances = torch.var(tensor, dim=0)
    # 取所有维度的平均值作为最终得分
    v_score = torch.mean(variances).item()
    return v_score


if __name__ == '__main__':
    # our_results = torch.load(
    #     "/Users/zhc/Documents/formal_experiment/mitdb_two_channels/dsp_flow_mixed_K500/impute_finetune_ckpt_lr1e-4/posterior_impute_samples_non_downstream_test.pth",
    #     map_location="cpu"
    # )
    # no_code_results = torch.load(
    #     "/Users/zhc/Documents/formal_experiment/mitdb_two_channels/dsp_flow_no_code/no_code_impute_finetune_ckpt_lr1e-4/no_code_impute_samples_non_downstream.pth",
    #     map_location="cpu"
    # )
    # ours_samples = our_results["all_samples"]
    # no_code_samples = no_code_results["all_samples"]
    # real = our_results["all_reals"]
    # plot_three_way_diversity(real[:,:,0], ours_samples[:,:,:,0].reshape(-1, 1000), no_code_samples[:,:,:,0].reshape(-1, 1000))
    # load_real_data()
    all_ours_embeds, all_real_embeds = show_ours()
    all_no_code_embeds, _ = show_no_code()
    # all_no_code_signal, all_no_code_embeds = show_no_code()
    # all_real_embeds = load_real_data()
    #
    # print(calculate_v_score(all_ours_signal[0][:,0]))
    # print(calculate_v_score(all_no_code_signal[0][:,0]))
    plot_three_way_diversity(all_real_embeds, all_ours_embeds, all_no_code_embeds)