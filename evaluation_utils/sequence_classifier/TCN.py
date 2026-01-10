import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score
import copy

class TemporalBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        dilation,
        dropout=0.1,
    ):
        super().__init__()

        padding = (kernel_size - 1) * dilation

        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=padding,
            dilation=dilation,
        )
        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size,
            padding=padding,
            dilation=dilation,
        )

        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

        self.downsample = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else None
        )

    def forward(self, x):
        # x: [B, C, T]
        out = self.conv1(x)
        out = out[..., : x.size(-1)]  # remove extra padding
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = out[..., : x.size(-1)]
        out = self.relu(out)
        out = self.dropout(out)

        res = x if self.downsample is None else self.downsample(x)
        return out + res


class TCNBackbone(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        num_layers=6,
        kernel_size=3,
        dropout=0.1,
    ):
        super().__init__()

        layers = []
        for i in range(num_layers):
            dilation = 2 ** i
            in_ch = in_channels if i == 0 else hidden_channels
            layers.append(
                TemporalBlock(
                    in_ch,
                    hidden_channels,
                    kernel_size,
                    dilation,
                    dropout,
                )
            )

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        # x: [B, C, T]
        return self.network(x)  # [B, H, T]


class TCNPointClassifier(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels=128,
        num_layers=6,
        kernel_size=3,
        num_classes=2,
        dropout=0.1,
    ):
        super().__init__()

        self.backbone = TCNBackbone(
            in_channels,
            hidden_channels,
            num_layers,
            kernel_size,
            dropout,
        )

        self.head = nn.Conv1d(
            hidden_channels,
            1,
            kernel_size=1,
        )

    def forward(self, x):
        """
        x: [B, C, T]
        return logits: [B, 2, T]
        """
        feat = self.backbone(x.permute(0,2,1))
        feat = feat.mean(dim=1)
        logits = self.head(feat)
        return logits


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


class WrappedTCN(nn.Module):
    def __init__(self, in_ch, anomaly_weight):
        super().__init__()
        self.model = TCNPointClassifier(
            in_ch,
            hidden_channels=64,
            num_layers=3,
            kernel_size=3,
            num_classes=2,
            dropout=0.0,
        )

        self.criterion = WeightedBCEWithLogitsLoss(beta=anomaly_weight)

    def forward(self, inputs, labels):
        logits = self.model(inputs)
        loss = self.criterion(logits, labels)
        return loss

    def predict(self, inputs):
        logits = self.model(inputs)
        probs = torch.sigmoid(logits)
        return (probs > 0.5).to(torch.long).squeeze(1)


def calculate_TCN_sequence(
        anomaly_weight, feature_size,
        real_anomaly,
        fake_anomaly,
        real_normal_train,
        real_normal_test,
        device, lr,
        max_epochs=2000,
        batch_size=64,
        patience=20,
):
    real_anomaly = torch.from_numpy(real_anomaly).float()
    fake_anomaly = torch.from_numpy(fake_anomaly).float()
    real_normal_train = torch.from_numpy(real_normal_train).float()
    real_normal_test = torch.from_numpy(real_normal_test).float()

    X_train = torch.cat((fake_anomaly, real_normal_train), dim=0)
    y_train = torch.cat((torch.ones(fake_anomaly.shape[0]), torch.zeros(real_normal_train.shape[0])), dim=0)

    X_test = torch.cat((real_anomaly, real_normal_test), dim=0)
    y_test = torch.cat((torch.ones(real_anomaly.shape[0]), torch.zeros(real_normal_test.shape[0])), dim=0)

    train_ds = TensorDataset(X_train, y_train)
    test_ds = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, drop_last=False)

    model = WrappedTCN(in_ch=feature_size, anomaly_weight=anomaly_weight).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.8,  # multiply LR by 0.5
        patience=2,  # wait 3 epochs with no improvement
        threshold=1e-4,  # improvement threshold
        min_lr=1e-5,  # min LR clamp
    )


    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(max_epochs):
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
        model.eval()
        val_loss = 0.0
        val_seen = 0
        for Xb, yb in test_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            with torch.no_grad():
                loss = model(Xb, yb)
            val_loss += loss.item() * Xb.shape[0]
            val_seen += Xb.shape[0]
        val_loss_avg = val_loss / val_seen
        print(f"Epoch{epoch}: train_loss: {train_loss_avg} | val_loss: {val_loss_avg} | lr: {optimizer.param_groups[0]['lr']} ||")
        scheduler.step(val_loss_avg)

        if best_val_loss > val_loss_avg:
            best_val_loss = val_loss_avg
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch}. Best val_loss = {best_val_loss:.6f}")
            break

    model.load_state_dict(best_state)
    model.eval()

    ### run evaluation on test set
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
        normal_correct += ((y_pred==yb) * (yb==0)).sum().item()
        anomaly_correct += ((y_pred==yb) * (yb==1)).sum().item()
        all_preds.append(y_pred.detach().cpu())
        all_labels.append(yb.detach().cpu())
    normal_accuracy = normal_correct / normal_num
    anomaly_accuracy = anomaly_correct / anomaly_num

    all_preds = torch.cat(all_preds).flatten().numpy()
    all_labels = torch.cat(all_labels).flatten().numpy()

    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)

    return normal_accuracy, anomaly_accuracy, precision, recall, f1


def run_TCN_sequence_evaluate(args, real_anomaly, fake_anomaly, gen_data, gen_labels, device):
    output_record = {
        "args": vars(args),
    }

    precisions = []
    recalls = []
    f1s = []
    normal_accuracies = []
    anomaly_accuracies = []
    for _ in range(3):
        print("real_data.shape:", real_data.shape)
        print("real_labels.shape:", real_labels.shape)
        print("gen_data.shape:", gen_data.shape)
        print("gen_labels.shape:", gen_labels.shape)

        normal_accuracy, anomaly_accuracy, precision, recall, f1 = calculate_TCN_sequence(
            anomaly_weight=1.0,
            feature_size=args.feature_size,
            real_anomaly=real_anomaly,
            fake_anomaly=fake_anomaly,
            real_normal_train=real_normal_train,
            real_normal_test=real_normal_test,
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
