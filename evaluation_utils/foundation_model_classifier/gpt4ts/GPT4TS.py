import numpy as np
import os
import json
import copy
from networkx.utils import configs
from transformers.models.gpt2.modeling_gpt2 import GPT2Model
from .Embed import DataEmbedding
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score
from einops import rearrange


class GPT4TSModel(nn.Module):
    
    def __init__(self, enc_in, seq_len):
        super(GPT4TSModel, self).__init__()
        freq = 'h'
        embed = 'timeF'
        dropout = 0.0
        d_ff = 2048
        d_model = 768

        self.is_ln = 1
        self.pred_len = seq_len
        self.seq_len = seq_len
        self.patch_size = 1
        self.stride = 1
        self.seq_len = seq_len
        self.d_ff = d_ff
        self.patch_num = (seq_len + self.pred_len - self.patch_size) // self.stride + 1

        self.padding_patch_layer = nn.ReplicationPad1d((0, self.stride)) 
        self.patch_num += 1
        self.enc_embedding = DataEmbedding(enc_in * self.patch_size, d_model, embed, freq, dropout)

        self.gpt2 = GPT2Model.from_pretrained('gpt2', output_attentions=True, output_hidden_states=True)
        self.gpt2.h = self.gpt2.h[:6] # use the frst six layers
        
        for i, (name, param) in enumerate(self.gpt2.named_parameters()):
            if 'ln' in name or 'wpe' in name: # or 'mlp' in name:
                param.requires_grad = True
            if 'mlp' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        self.act = nn.functional.gelu
        self.ln_proj = nn.LayerNorm(d_model)
        self.out_layer = nn.Linear(d_model, 1)


    def forward(self, x_enc):
        dec_out = self.classification(x_enc, x_mark_enc=None)
        return dec_out[:, -self.pred_len:, :].permute(0,2,1)  # [B, L, D]


    def classification(self, x_enc, x_mark_enc):
        # print(x_enc.shape)
        B, L, M = x_enc.shape
        input_x = rearrange(x_enc, 'b l m -> b m l')
        input_x = self.padding_patch_layer(input_x)
        input_x = input_x.unfold(dimension=-1, size=self.patch_size, step=self.stride)
        input_x = rearrange(input_x, 'b m n p -> b n (p m)')

        outputs = self.enc_embedding(input_x, None)

        outputs = self.gpt2(inputs_embeds=outputs).last_hidden_state
        outputs = self.act(outputs)
        outputs = self.ln_proj(outputs)
        outputs = self.out_layer(outputs)

        return outputs


class WeightedBCEWithLogitsLoss(nn.Module):
    def __init__(self, beta=1.0):
        """
        使用BCEWithLogitsLoss，它在内部处理logits，数值更稳定
        pos_weight: 正样本权重 = beta
        """
        super().__init__()
        self.beta = beta

    def forward(self, logits, labels):
        """
        logits: (B,1,T)
        labels: (B,T)
        """
        # 挤压掉通道维度，BCEWithLogitsLoss期望(B,T)或(B,C,T)
        if logits.dim() == 3:
            logits = logits.squeeze(1)  # (B,T)

        # pos_weight = beta 表示：正样本的损失乘以beta
        # 例如：beta=5，那么每个正样本的损失贡献是负样本的5倍
        loss_fn = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([self.beta], device=logits.device),
            reduction='mean'
        )
        return loss_fn(logits, labels.float())



class Wrapped_GPT4TSModel(nn.Module):
    def __init__(self, in_ch, seq_len, anomaly_weight):
        super().__init__()
        self.model = GPT4TSModel(enc_in=in_ch, seq_len=seq_len)
        self.criterion = WeightedBCEWithLogitsLoss(beta=anomaly_weight)

    def forward(self, inputs, labels):
        logits = self.model(inputs)
        loss = self.criterion(logits, labels)
        return loss

    def predict(self, inputs):
        logits = self.model(inputs)
        probs = torch.sigmoid(logits)
        return (probs > 0.5).to(torch.long).squeeze(1)


# def calculate_GPT4TS(
#         anomaly_weight, feature_size,
#         ori_data, ori_labels,
#         gen_data, gen_labels,
#         device, lr,
#         max_epochs=2000,
#         batch_size=64,
#         patience=20,
# ):
#     X_real = torch.tensor(ori_data, dtype=torch.float32)
#     X_fake = torch.tensor(gen_data, dtype=torch.float32)
#
#     y_real = torch.tensor(ori_labels, dtype=torch.float32)
#     y_fake = torch.tensor(gen_labels, dtype=torch.float32)
#
#     train_ds = TensorDataset(X_fake, y_fake)
#     test_ds = TensorDataset(X_real, y_real)
#
#     train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
#     test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, drop_last=False)
#
#     model = Wrapped_GPT4TSModel(
#         in_ch=feature_size, seq_len=X_real.shape[1], anomaly_weight=anomaly_weight,
#     ).to(device)
#
#     optimizer = torch.optim.Adam(model.parameters(), lr=lr)
#
#     scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#         optimizer,
#         mode='min',
#         factor=0.8,  # multiply LR by 0.5
#         patience=5,  # wait 3 epochs with no improvement
#         threshold=1e-4,  # improvement threshold
#         min_lr=1e-6,  # min LR clamp
#     )
#
#
#
#     best_val_loss = float("inf")
#     best_state = None
#     patience_counter = 0
#
#
#     for epoch in range(max_epochs):
#         model.train()
#         # for Xb, yb in tqdm(train_loader, desc=f"Epoch{epoch}"):
#         train_loss = 0.0
#         train_seen = 0
#         for Xb, yb in train_loader:
#             Xb, yb = Xb.to(device), yb.to(device)
#             loss = model(Xb, yb)
#             # breakpoint()
#             # print(loss.item())
#             train_loss += loss.item() * Xb.shape[0]
#             train_seen += Xb.shape[0]
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#
#         train_loss_avg = train_loss / train_seen
#         model.eval()
#         val_loss = 0.0
#         val_seen = 0
#         for Xb, yb in test_loader:
#             Xb, yb = Xb.to(device), yb.to(device)
#             with torch.no_grad():
#                 loss = model(Xb, yb)
#
#             val_loss += loss.item() * Xb.shape[0]
#             val_seen += Xb.shape[0]
#         val_loss_avg = val_loss / val_seen
#         print(f"Epoch{epoch}: train_loss: {train_loss_avg} | val_loss: {val_loss_avg} | lr: {optimizer.param_groups[0]['lr']} ||")
#
#
#
#
#
#         scheduler.step(val_loss_avg)
#
#         if best_val_loss > val_loss_avg:
#             best_val_loss = val_loss_avg
#             best_state = model.state_dict()
#             patience_counter = 0
#         else:
#             patience_counter += 1
#
#         if patience_counter >= patience:
#             print(f"\nEarly stopping at epoch {epoch}. Best val_loss = {best_val_loss:.6f}")
#             break
#
#     model.load_state_dict(best_state)
#     model.eval()
#
#     ### run evaluation on test set
#     normal_correct = 0
#     normal_num = 0
#     anomaly_correct = 0
#     anomaly_num = 0
#     all_preds = []
#     all_labels = []
#
#     for Xb, yb in test_loader:
#         Xb, yb = Xb.to(device), yb.to(device)
#         y_pred = model.predict(Xb)
#         normal_num += (yb == 0).sum().item()
#         anomaly_num += (yb == 1).sum().item()
#         normal_correct += ((y_pred==yb) * (yb==0)).sum().item()
#         anomaly_correct += ((y_pred==yb) * (yb==1)).sum().item()
#         all_preds.append(y_pred.detach().cpu())
#         all_labels.append(yb.detach().cpu())
#     normal_accuracy = normal_correct / normal_num
#     anomaly_accuracy = anomaly_correct / anomaly_num
#
#     all_preds = torch.cat(all_preds).flatten().numpy()
#     all_labels = torch.cat(all_labels).flatten().numpy()
#
#     precision = precision_score(all_labels, all_preds, zero_division=0)
#     recall = recall_score(all_labels, all_preds, zero_division=0)
#     f1 = f1_score(all_labels, all_preds, zero_division=0)
#
#     return normal_accuracy, anomaly_accuracy, precision, recall, f1



def calculate_GPT4TS_new(
        anomaly_weight, feature_size,
        ori_data, ori_labels,
        gen_data, gen_labels,
        device, lr,
        max_epochs=2000,
        batch_size=64,
        patience=20,
):
    X_real = torch.tensor(ori_data, dtype=torch.float32)
    X_fake = torch.tensor(gen_data, dtype=torch.float32)

    y_real = torch.tensor(ori_labels, dtype=torch.float32)
    y_fake = torch.tensor(gen_labels, dtype=torch.float32)

    train_ds = TensorDataset(X_fake, y_fake)
    test_ds = TensorDataset(X_real, y_real)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, drop_last=False)

    model = Wrapped_GPT4TSModel(
        in_ch=feature_size,
        seq_len=X_real.shape[1],
        anomaly_weight=anomaly_weight,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # =========================
    # 使用 F1 作为调度指标
    # =========================
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',      # ← 关键
        factor=0.8,
        patience=5,
        threshold=1e-4,
        min_lr=2e-5,
    )

    best_val_f1 = -float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(max_epochs):
        # -------------------
        # Train
        # -------------------
        model.train()
        train_loss = 0.0
        train_seen = 0

        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            loss = model(Xb, yb)

            train_loss += loss.item() * Xb.shape[0]
            train_seen += Xb.shape[0]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_loss_avg = train_loss / train_seen

        # -------------------
        # Validation (F1)
        # -------------------
        model.eval()
        val_loss = 0.0
        val_seen = 0

        all_preds = []
        all_labels = []

        for Xb, yb in test_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            with torch.no_grad():
                loss = model(Xb, yb)
                y_pred = model.predict(Xb)

            val_loss += loss.item() * Xb.shape[0]
            val_seen += Xb.shape[0]

            all_preds.append(y_pred.detach().cpu())
            all_labels.append(yb.detach().cpu())

        val_loss_avg = val_loss / val_seen

        all_preds = torch.cat(all_preds).flatten().numpy()
        all_labels = torch.cat(all_labels).flatten().numpy()

        precision = precision_score(all_labels, all_preds, zero_division=0)
        recall = recall_score(all_labels, all_preds, zero_division=0)
        val_f1 = f1_score(all_labels, all_preds, zero_division=0)

        print(
            f"Epoch {epoch}: "
            f"train_loss={train_loss_avg:.6f} | "
            f"val_loss={val_loss_avg:.6f} | "
            f"val_f1={val_f1:.4f} | "
            f"lr={optimizer.param_groups[0]['lr']:.2e}"
        )

        # -------------------
        # 用 F1 调度学习率
        # -------------------
        scheduler.step(val_f1)

        # -------------------
        # Early stopping (by F1)
        # -------------------
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(
                f"\nEarly stopping at epoch {epoch}. "
                f"Best val F1 = {best_val_f1:.4f}"
            )
            break

    # =========================
    # Load best model
    # =========================
    model.load_state_dict(best_state)
    model.eval()

    # =========================
    # Final evaluation
    # =========================
    normal_correct = 0
    normal_num = 0
    anomaly_correct = 0
    anomaly_num = 0

    all_preds = []
    all_labels = []

    for Xb, yb in test_loader:
        Xb, yb = Xb.to(device), yb.to(device)
        y_pred = model.predict(Xb)

        normal_num += (yb == 0).sum().item()
        anomaly_num += (yb == 1).sum().item()

        normal_correct += ((y_pred == yb) & (yb == 0)).sum().item()
        anomaly_correct += ((y_pred == yb) & (yb == 1)).sum().item()

        all_preds.append(y_pred.detach().cpu())
        all_labels.append(yb.detach().cpu())

    normal_accuracy = normal_correct / max(normal_num, 1)
    anomaly_accuracy = anomaly_correct / max(anomaly_num, 1)

    all_preds = torch.cat(all_preds).flatten().numpy()
    all_labels = torch.cat(all_labels).flatten().numpy()

    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)

    return normal_accuracy, anomaly_accuracy, precision, recall, f1


def run_GPT4TS_evaluate(args, real_data, real_labels, gen_data, gen_labels, device):
    output_record = {
        "args": vars(args),
    }
    # torch.save(real_data, "/root/tianyi/real_data.pt")
    # torch.save(real_labels, "/root/tianyi/real_labels.pt")
    # seq_len = real_data.shape[1]
    # real_data = real_data.reshape(-1, seq_len//2, 2)
    # real_labels = real_labels.reshape(-1, seq_len//2)
    # gen_data = gen_data.reshape(-1, seq_len//2, 2)
    # gen_labels = gen_labels.reshape(-1, seq_len//2)
    # breakpoint()

    # gen_data = real_data.clone()
    # gen_labels = real_labels.clone()

    precisions = []
    recalls = []
    f1s = []
    normal_accuracies = []
    anomaly_accuracies = []
    for _ in range(5):
        random_indices = torch.randperm(len(gen_data))[:1000]
        sampled_gen_data = gen_data[random_indices]
        sampled_gen_labels = gen_labels[random_indices]

        print("real_data.shape:", real_data.shape)
        print("real_labels.shape:", real_labels.shape)
        print("gen_data.shape:", gen_data.shape)
        print("gen_labels.shape:", gen_labels.shape)

        normal_accuracy, anomaly_accuracy, precision, recall, f1 = calculate_GPT4TS_new(
            anomaly_weight=1.0,
            feature_size=args.feature_size,
            ori_data=real_data,
            ori_labels=real_labels,
            gen_data=sampled_gen_data,
            gen_labels=sampled_gen_labels,
            device=device,
            lr=1e-4,
            max_epochs=1000,
            batch_size=16,
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
    output_record.update({"result_onefitsall": result})

    save_path = os.path.join(args.out_dir, f"onefitsall_evaluation_results.jsonl")

    # with open(save_path, "a") as f:
    #     f.write(json.dumps(output_record) + "\n")

    with open(save_path, "a") as f:
        json.dump(output_record, f, indent=2)
        f.write("\n")
