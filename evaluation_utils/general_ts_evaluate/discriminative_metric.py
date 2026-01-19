import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, TensorDataset, random_split
import matplotlib.pyplot as plt
import json


class GRUDiscriminator(nn.Module):
    """
    GRU-based binary classifier:
       input  (B, T, C)
       output (B, 1)
    """
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x, anomaly_lengths):
        out, _ = self.gru(x)
        idx = anomaly_lengths - 1  # (B,)
        idx = idx.clamp(min=0)  # 防止 0 或负数
        B = out.size(0)
        last_state = out[torch.arange(B), idx, :]  # (B, H)

        # last_state = out[:, -1, :]   # final hidden state
        logit = self.fc(last_state)
        prob = torch.sigmoid(logit)
        return prob, logit


def discriminative_score_metrics(
    ori_data,
    gen_data,
    anomaly_lengths,
    hidden_dim=64,
    max_epochs=2000,
    batch_size=64,
    patience=20,
    device="cuda"
):
    """
    PyTorch version of discriminative score in TimeGAN,
    rewritten to match the structure/style of predictive_score_metrics_torch.

    Train discriminator on synthetic+real data (with validation + early stopping),
    evaluate accuracy on a held-out test set.

    Returns:
        disc_score = |accuracy - 0.5|
        real_acc
        fake_acc
    """

    device = torch.device(device if torch.cuda.is_available() else "cpu")

    # ----------- 1. build full dataset -----------
    # label: real=1, fake=0
    X_real = torch.tensor(ori_data, dtype=torch.float32)
    X_fake = torch.tensor(gen_data, dtype=torch.float32)

    y_real = torch.ones((X_real.shape[0], 1), dtype=torch.float32, device=device)
    y_fake = torch.zeros((X_fake.shape[0], 1), dtype=torch.float32, device=device)

    X_all = torch.cat([X_real, X_fake], dim=0)
    y_all = torch.cat([y_real, y_fake], dim=0)

    # shuffle dataset
    n = X_all.shape[0]
    idx = torch.randperm(n)
    X_all = X_all[idx]
    y_all = y_all[idx]
    anomaly_lengths = anomaly_lengths[idx]

    # ----------- 2. train/val/test split -----------
    n_val  = int(0.2 * n)
    n_train = n - n_val

    train_ds, val_ds = random_split(
        TensorDataset(X_all, y_all, anomaly_lengths.repeat(2)),
        [n_train, n_val]
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)

    # ----------- 3. build model -----------
    input_dim = ori_data.shape[-1]
    model = GRUDiscriminator(input_dim=input_dim, hidden_dim=hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()

    # ----------- 4. training with early stopping -----------
    best_acc = 0.0
    patience_counter = 0

    for epoch in range(max_epochs):

        # ---- train ----
        model.train()
        for Xb, yb, lengths in train_loader:
            Xb, yb, lengths = Xb.to(device), yb.to(device), lengths.to(device)
            _, logit = model(Xb, lengths)
            loss = loss_fn(logit, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # ---- validation ----
        model.eval()
        val_losses = []
        num_seen = 0
        num_correct = 0
        with torch.no_grad():
            for Xb, yb, lengths in val_loader:
                Xb, yb, lengths = Xb.to(device), yb.to(device), lengths.to(device)
                prob, _ = model(Xb, lengths)
                pred = (prob.flatten() > 0.5).long()
                num_seen += pred.shape[0]
                num_correct += (pred == yb.flatten().long()).to(torch.float32).sum().item()

        val_acc = num_correct / num_seen
        print(f"Epoch {epoch:03d} | val_acc={val_acc:.6f}")

        # early stopping
        if best_acc < val_acc:
            best_acc = val_acc
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch}. Best val_acc = {best_acc:.6f}")
            break

    # ----------- 5. load best model -----------
    disc_score = max(best_acc, 1.0 - best_acc) - 0.5

    return disc_score

def calculate_discriminative_metrics(all_data_path, out_path):
    # all_data = torch.load(
    #         "/Users/zhc/Documents/mitdb_two_channels/dsp_flow_mixed_K500/impute_finetune_ckpt_lr1e-4/posterior_impute_samples_non_downstream.pth",
    #         map_location='cpu'
    #     )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_data = torch.load(
        all_data_path,
        map_location=device
    )
    full_gen_data = all_data["all_samples"]
    full_ori_data = all_data["all_reals"]
    anomaly_labels = all_data["all_labels"]
    max_anomaly_length = anomaly_labels.sum(-1).max().item()
    anomaly_lengths = anomaly_labels.sum(-1)
    scores = []
    for i in range(5):
        full_gen_data_tmp = full_gen_data[:, i]
        ts_dim = full_gen_data_tmp.shape[-1]
        N = full_gen_data_tmp.shape[0]
        gen_data = torch.zeros(N, max_anomaly_length, ts_dim).to(device=anomaly_labels.device)
        orig_data = torch.zeros(N, max_anomaly_length, ts_dim).to(device=anomaly_labels.device)

        for j in range(len(full_gen_data_tmp)):
            anomaly_indices = torch.where(anomaly_labels[j]==1)[0]
            gen_data[j, :len(anomaly_indices)] = full_gen_data_tmp[j][anomaly_indices]
            orig_data[j, :len(anomaly_indices)] = full_ori_data[j][anomaly_indices]

        score = discriminative_score_metrics(
            ori_data=orig_data,
            gen_data=gen_data,
            anomaly_lengths=anomaly_lengths,
            hidden_dim=64,
            max_epochs=2000,
            batch_size=16,
            patience=20,
            device=device
        )
        scores.append(score)

    # ===== 新增部分：写 jsonl =====
    scores_tensor = torch.tensor(scores, dtype=torch.float32)
    mean_score = scores_tensor.mean().item()
    std_score = scores_tensor.std(unbiased=False).item()

    with open(out_path, "w") as f:
        for i, s in enumerate(scores):
            record = {
                "trial": i,
                "score": float(s)
            }
            f.write(json.dumps(record) + "\n")

        summary = {
            "mean": mean_score,
            "std": std_score
        }
        f.write(json.dumps(summary) + "\n")

    return scores, mean_score, std_score




if __name__ == "__main__":
    pass


    all_data_paths = [
        "/work/vb21/haochen/code/formal_experiment/mitdb_two_channels/dsp_flow_mixed_K500/impute_finetune_ckpt_lr1e-4/posterior_impute_samples_non_downstream.pth",
        "/work/vb21/haochen/code/formal_experiment/mitdb_two_channels/dsp_flow_no_code/no_code_impute_finetune_ckpt_lr1e-4/no_code_impute_samples_non_downstream.pth",
        "/work/vb21/haochen/code/formal_experiment/mitdb_two_channels/flowts/no_code_impute_from_scratch_ckpt_lr1e-4/no_code_impute_samples_non_downstream.pth",
        "/work/vb21/haochen/code/formal_experiment/mitdb_two_channels/diffusion_ts/ckpt_lr1e-4/no_code_impute_samples_non_downstream.pth",
        "/work/vb21/haochen/code/formal_experiment/mitdb_two_channels/TimeVAE/no_code_impute_from_scratch_ckpt_lr1e-4/no_code_impute_samples_non_downstream.pth",
        "/work/vb21/haochen/code/formal_experiment/mitdb_two_channels/C-GATS/ckpt_lr1e-4/no_code_impute_samples_non_downstream.pth",
        "/work/vb21/haochen/code/formal_experiment/mitdb_two_channels/GENIAS/ckpt_lr1e-4/no_code_impute_samples_non_downstream.pth",

        "/work/vb21/haochen/code/formal_experiment/qtdb_two_channels/dsp_flow_mixed_K500/impute_finetune_ckpt_lr1e-4/posterior_impute_samples_non_downstream.pth",
        "/work/vb21/haochen/code/formal_experiment/qtdb_two_channels/dsp_flow_no_code/no_code_impute_finetune_ckpt_lr1e-4/no_code_impute_samples_non_downstream.pth",
        "/work/vb21/haochen/code/formal_experiment/qtdb_two_channels/flowts/no_code_impute_from_scratch_ckpt_lr1e-4/no_code_impute_samples_non_downstream.pth",
        "/work/vb21/haochen/code/formal_experiment/qtdb_two_channels/diffusion_ts/ckpt_lr1e-4/no_code_impute_samples_non_downstream.pth",
        "/work/vb21/haochen/code/formal_experiment/qtdb_two_channels/TimeVAE/no_code_impute_from_scratch_ckpt_lr1e-4/no_code_impute_samples_non_downstream.pth",
        "/work/vb21/haochen/code/formal_experiment/qtdb_two_channels/C-GATS/ckpt_lr1e-4/no_code_impute_samples_non_downstream.pth",
        "/work/vb21/haochen/code/formal_experiment/qtdb_two_channels/GENIAS/ckpt_lr1e-4/no_code_impute_samples_non_downstream.pth",

        "/work/vb21/haochen/code/formal_experiment/svdb_two_channels/dsp_flow_mixed_K500/impute_finetune_ckpt_lr1e-4/posterior_impute_samples_non_downstream.pth",
        "/work/vb21/haochen/code/formal_experiment/svdb_two_channels/dsp_flow_no_code/no_code_impute_finetune_ckpt_lr1e-4/no_code_impute_samples_non_downstream.pth",
        "/work/vb21/haochen/code/formal_experiment/svdb_two_channels/flowts/no_code_impute_from_scratch_ckpt_lr1e-4/no_code_impute_samples_non_downstream.pth",
        "/work/vb21/haochen/code/formal_experiment/svdb_two_channels/diffusion_ts/ckpt_lr1e-4/no_code_impute_samples_non_downstream.pth",
        "/work/vb21/haochen/code/formal_experiment/svdb_two_channels/TimeVAE/no_code_impute_from_scratch_ckpt_lr1e-4/no_code_impute_samples_non_downstream.pth",
        "/work/vb21/haochen/code/formal_experiment/svdb_two_channels/C-GATS/ckpt_lr1e-4/no_code_impute_samples_non_downstream.pth",
        "/work/vb21/haochen/code/formal_experiment/svdb_two_channels/GENIAS/ckpt_lr1e-4/no_code_impute_samples_non_downstream.pth",

        "/work/vb21/haochen/code/formal_experiment/PV/dsp_flow_mixed_K500/impute_finetune_ckpt_lr1e-4/posterior_impute_samples_non_downstream.pth",
        "/work/vb21/haochen/code/formal_experiment/PV/dsp_flow_no_code/no_code_impute_finetune_ckpt_lr1e-4/no_code_impute_samples_non_downstream.pth",
        "/work/vb21/haochen/code/formal_experiment/PV/flowts/no_code_impute_from_scratch_ckpt_lr1e-4/no_code_impute_samples_non_downstream.pth",
        "/work/vb21/haochen/code/formal_experiment/PV/diffusion_ts/ckpt_lr1e-4/no_code_impute_samples_non_downstream.pth",
        "/work/vb21/haochen/code/formal_experiment/PV/TimeVAE/no_code_impute_from_scratch_ckpt_lr1e-4/no_code_impute_samples_non_downstream.pth",
        "/work/vb21/haochen/code/formal_experiment/PV/C-GATS/ckpt_lr1e-4/no_code_impute_samples_non_downstream.pth",
        "/work/vb21/haochen/code/formal_experiment/PV/GENIAS/ckpt_lr1e-4/no_code_impute_samples_non_downstream.pth",

        "/work/vb21/haochen/code/formal_experiment/traffic/dsp_flow_mixed_K500/impute_finetune_ckpt_lr1e-4/posterior_impute_samples_non_downstream.pth",
        "/work/vb21/haochen/code/formal_experiment/traffic/dsp_flow_no_code/no_code_impute_finetune_ckpt_lr1e-4/no_code_impute_samples_non_downstream.pth",
        "/work/vb21/haochen/code/formal_experiment/traffic/flowts/no_code_impute_from_scratch_ckpt_lr1e-4/no_code_impute_samples_non_downstream.pth",
        "/work/vb21/haochen/code/formal_experiment/traffic/diffusion_ts/ckpt_lr1e-4/no_code_impute_samples_non_downstream.pth",
        "/work/vb21/haochen/code/formal_experiment/traffic/TimeVAE/no_code_impute_from_scratch_ckpt_lr1e-4/no_code_impute_samples_non_downstream.pth",
        "/work/vb21/haochen/code/formal_experiment/traffic/C-GATS/ckpt_lr1e-4/no_code_impute_samples_non_downstream.pth",
        "/work/vb21/haochen/code/formal_experiment/traffic/GENIAS/ckpt_lr1e-4/no_code_impute_samples_non_downstream.pth",
    ]

    all_save_paths = [
        "/work/vb21/haochen/code/discriminative_scores/mitdb_two_channels/dsp_flow_mixed_K500/scores.jsonl",
        "/work/vb21/haochen/code/discriminative_scores/mitdb_two_channels/dsp_flow_no_code/scores.jsonl",
        "/work/vb21/haochen/code/discriminative_scores/mitdb_two_channels/flowts/scores.jsonl",
        "/work/vb21/haochen/code/discriminative_scores/mitdb_two_channels/diffusion_ts/scores.jsonl",
        "/work/vb21/haochen/code/discriminative_scores/mitdb_two_channels/TimeVAE/scores.jsonl",
        "/work/vb21/haochen/code/discriminative_scores/mitdb_two_channels/C-GATS/scores.jsonl",
        "/work/vb21/haochen/code/discriminative_scores/mitdb_two_channels/GENIAS/scores.jsonl",

        "/work/vb21/haochen/code/discriminative_scores/qtdb_two_channels/dsp_flow_mixed_K500/scores.jsonl",
        "/work/vb21/haochen/code/discriminative_scores/qtdb_two_channels/dsp_flow_no_code/scores.jsonl",
        "/work/vb21/haochen/code/discriminative_scores/qtdb_two_channels/flowts/scores.jsonl",
        "/work/vb21/haochen/code/discriminative_scores/qtdb_two_channels/diffusion_ts/scores.jsonl",
        "/work/vb21/haochen/code/discriminative_scores/qtdb_two_channels/TimeVAE/scores.jsonl",
        "/work/vb21/haochen/code/discriminative_scores/qtdb_two_channels/C-GATS/scores.jsonl",
        "/work/vb21/haochen/code/discriminative_scores/qtdb_two_channels/GENIAS/scores.jsonl",

        "/work/vb21/haochen/code/discriminative_scores/svdb_two_channels/dsp_flow_mixed_K500/scores.jsonl",
        "/work/vb21/haochen/code/discriminative_scores/svdb_two_channels/dsp_flow_no_code/scores.jsonl",
        "/work/vb21/haochen/code/discriminative_scores/svdb_two_channels/flowts/scores.jsonl",
        "/work/vb21/haochen/code/discriminative_scores/svdb_two_channels/diffusion_ts/scores.jsonl",
        "/work/vb21/haochen/code/discriminative_scores/svdb_two_channels/TimeVAE/scores.jsonl",
        "/work/vb21/haochen/code/discriminative_scores/svdb_two_channels/C-GATS/scores.jsonl",
        "/work/vb21/haochen/code/discriminative_scores/svdb_two_channels/GENIAS/scores.jsonl",

        "/work/vb21/haochen/code/discriminative_scores/PV/dsp_flow_mixed_K500/scores.jsonl",
        "/work/vb21/haochen/code/discriminative_scores/PV/dsp_flow_no_code/scores.jsonl",
        "/work/vb21/haochen/code/discriminative_scores/PV/flowts/scores.jsonl",
        "/work/vb21/haochen/code/discriminative_scores/PV/diffusion_ts/scores.jsonl",
        "/work/vb21/haochen/code/discriminative_scores/PV/TimeVAE/scores.jsonl",
        "/work/vb21/haochen/code/discriminative_scores/PV/C-GATS/scores.jsonl",
        "/work/vb21/haochen/code/discriminative_scores/PV/GENIAS/scores.jsonl",

        "/work/vb21/haochen/code/discriminative_scores/traffic/dsp_flow_mixed_K500/scores.jsonl",
        "/work/vb21/haochen/code/discriminative_scores/traffic/dsp_flow_no_code/scores.jsonl",
        "/work/vb21/haochen/code/discriminative_scores/traffic/flowts/scores.jsonl",
        "/work/vb21/haochen/code/discriminative_scores/traffic/diffusion_ts/scores.jsonl",
        "/work/vb21/haochen/code/discriminative_scores/traffic/TimeVAE/scores.jsonl",
        "/work/vb21/haochen/code/discriminative_scores/traffic/C-GATS/scores.jsonl",
        "/work/vb21/haochen/code/discriminative_scores/traffic/GENIAS/scores.jsonl",
    ]

    for data_path, save_path in zip(all_data_paths, all_save_paths):
        calculate_discriminative_metrics(data_path, save_path)