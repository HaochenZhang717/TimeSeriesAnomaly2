import torch
from generation_models import DSPFlow
# from dataset_utils import ImputationNormalECGDatasetForSample
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


def dict_collate_fn(batch):
    out = {}
    for key in batch[0].keys():
        out[key] = torch.stack([sample[key] for sample in batch], dim=0)
    return out


import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def plot_pca_analysis(data, title="PCA Analysis of Feature Matrix"):
    """
    输入 data 形状为 (251, 32)
    """
    # 1. 标准化 (PCA 对量纲极其敏感，必须标准化)
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    # 2. 执行 PCA
    # n_components=2 表示降到2维用于画图
    pca = PCA(n_components=2)
    pca_res = pca.fit_transform(data_scaled)

    # 获取解释方差比
    var_exp = pca.explained_variance_ratio_

    # 3. 绘图
    plt.figure(figsize=(10, 7))

    # 画出散点图
    plt.scatter(pca_res[:, 0], pca_res[:, 1],
                c='mediumseagreen', alpha=0.7, edgecolors='w', s=60)

    # 在坐标轴上标注该维度解释了多少方差
    plt.xlabel(f"Principal Component 1 ({var_exp[0] * 100:.1f}%)", fontsize=12)
    plt.ylabel(f"Principal Component 2 ({var_exp[1] * 100:.1f}%)", fontsize=12)

    plt.title(f"{title}\nTotal Variance Explained: {(var_exp[0] + var_exp[1]) * 100:.1f}%", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.axhline(0, color='black', lw=1, alpha=0.2)
    plt.axvline(0, color='black', lw=1, alpha=0.2)

    plt.show()





def get_latent_and_plot():
    data_path = "dataset_utils/real_anomaly_data_test/mitdb106_train_anomaly_segments.pt"
    all_data = torch.load(data_path, map_location='cpu', weights_only=False)


    all_representatives = []
    for i in range(len(all_data)):
        # anomaly_index = torch.where(anomaly_labels[i] != 0)[0]
        # anomaly_segment = real_data[i,anomaly_index]
        anomaly_segment = all_data[i]

        pad_to_800 = torch.zeros(800, 2)
        attn_mask = torch.zeros(800)
        pad_to_800[:len(anomaly_segment)] = torch.from_numpy(anomaly_segment)
        attn_mask[:len(anomaly_segment)] = 1
        # print(len(anomaly_segment))
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
            vqvae_ckpt="/Users/zhc/Documents/formal_experiment_0122/mitdb_two_channels/dsp_flow_mixed_K500/vqvae_save_path/vqvae.pt"
        )

    ckpt = torch.load(
        f"/Users/zhc/Documents/formal_experiment_0122/mitdb_two_channels/dsp_flow_mixed_K500/impute_finetune_ckpt_lr1e-4/ckpt.pth",
        map_location="cpu"
    )
    model.load_state_dict(ckpt)
    model.eval()

    latent_list = []

    for anomaly_representative in all_representatives:
        with torch.no_grad():
            embed = model.vqvae.encode(
                anomaly_representative["signal"].unsqueeze(0),
                anomaly_representative["attn_mask"].unsqueeze(0),
            )
        latent_list.append(embed)

    latents_all = torch.cat(latent_list).reshape(-1, 32)
    print(latents_all.shape)

    X = latents_all.numpy() # 模拟数据
    plot_pca_analysis(X)
    # # 2. 预处理：标准化 (非常重要！)
    # # t-SNE 受到特征量纲影响很大，标准化能确保每个特征权重一致
    # scaler = StandardScaler()
    # X_scaled = scaler.fit_transform(X)
    #
    # # 3. 运行 t-SNE
    # # perplexity: 建议设为样本量的 5% 到 20% 之间。对于 251 个样本，取 30 比较合适
    # tsne = TSNE(
    #     n_components=2,
    #     perplexity=30,
    #     learning_rate='auto',
    #     init='pca',
    #     random_state=42
    # )
    # X_embedded = tsne.fit_transform(X_scaled)
    #
    # # 4. 可视化
    # plt.figure(figsize=(10, 7))
    # plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c='steelblue', edgecolors='w', alpha=0.7, s=60)
    #
    # plt.title("t-SNE Visualization of (251, 32) Matrix", fontsize=14)
    # plt.xlabel("t-SNE dimension 1")
    # plt.ylabel("t-SNE dimension 2")
    # plt.grid(True, linestyle='--', alpha=0.5)
    # plt.show()



if __name__ == "__main__":
    get_latent_and_plot()