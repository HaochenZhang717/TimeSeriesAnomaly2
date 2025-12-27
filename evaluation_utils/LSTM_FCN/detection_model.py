import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score


class SE1D(nn.Module):
    """
    Temporal squeeze-and-excite for 1D feature maps.
    Input:  u [B, C, T]
    Output: u_recal [B, C, T]
    """
    def __init__(self, channels: int, r: int = 16):
        super().__init__()
        hidden = max(1, channels // r)
        self.fc1 = nn.Linear(channels, hidden, bias=True)
        self.fc2 = nn.Linear(hidden, channels, bias=True)

    def forward(self, u):
        z = u.mean(dim=-1)  # [B, C]
        s = torch.sigmoid(self.fc2(F.relu(self.fc1(z))))  # [B, C]
        return u * s.unsqueeze(-1)  # [B, C, T]


class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, k, use_se=False, se_r=16):
        super().__init__()
        pad = k // 2
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=k, padding=pad, bias=False)
        # match paper BN hyperparams: momentum=0.99, eps=0.001
        self.bn = nn.BatchNorm1d(out_ch, momentum=0.99, eps=1e-3)
        self.use_se = use_se
        self.se = SE1D(out_ch, r=se_r) if use_se else None

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x, inplace=True)
        if self.use_se:
            x = self.se(x)
        return x


class MLSTM_FCN_Model(nn.Module):
    def __init__(
        self,
        num_vars: int,
        num_classes: int,
        lstm_hidden: int = 128,
        lstm_layers: int = 1,
        dropout: float = 0.1,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.num_vars = num_vars
        self.num_classes = num_classes
        self.lstm_hidden = lstm_hidden
        self.lstm_layers = lstm_layers
        self.bidirectional = bidirectional
        self.dropout_p = dropout

        # FCN branch (keep time dimension!)
        self.fcn1 = ConvBNReLU(num_vars, 128, k=7, use_se=True, se_r=16)
        self.fcn2 = ConvBNReLU(128, 256, k=5, use_se=True, se_r=16)
        self.fcn3 = ConvBNReLU(256, 128, k=3, use_se=False)

        # LSTM branch: standard along time Q with input_size=M
        self.lstm = nn.LSTM(
            input_size=num_vars,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )

        lstm_out_dim = lstm_hidden * (2 if bidirectional else 1)
        self.dropout = nn.Dropout(dropout)

        # per-timestep head
        self.classifier = nn.Linear(128 + lstm_out_dim, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        x: [B, Q, M]
        mask: [B, Q] with 1 for valid, 0 for pad (optional)
        """
        if x.dim() != 3:
            raise ValueError(f"x must be [B,Q,M], got {tuple(x.shape)}")
        B, Q, M = x.shape
        if M != self.num_vars:
            raise ValueError(f"Expected num_vars={self.num_vars}, got M={M}")

        # ---- FCN branch ----
        xf = x.transpose(1, 2)         # [B, M, Q]
        xf = self.fcn1(xf)             # [B, 128, Q]
        xf = self.fcn2(xf)             # [B, 256, Q]
        xf = self.fcn3(xf)             # [B, 128, Q]
        xf = xf.transpose(1, 2)        # [B, Q, 128]

        # ---- LSTM branch ----
        # If you have padding and want LSTM to ignore it, you can pack_padded_sequence.
        # Here we keep it simple; you can still use mask in loss to ignore padded steps.
        hl, _ = self.lstm(x)           # [B, Q, H*(1 or 2)]
        hl = self.dropout(hl)          # [B, Q, H*dir]

        # ---- Fuse & classify per timestep ----
        feat = torch.cat([xf, hl], dim=-1)  # [B, Q, 128 + H*dir]
        logits = self.classifier(feat).permute(0, 2, 1)      # [B, Q, num_classes]
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



class MLSTM_FCN(nn.Module):
    def __init__(self, in_ch, anomaly_weight):
        super().__init__()
        self.model = MLSTM_FCN_Model(
            num_vars=in_ch,
            num_classes=2,
            lstm_hidden= 128,
            lstm_layers=1,
            dropout=0.1,
            bidirectional=True,
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




def calculate_MLSTM_FCN(
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

    model = MLSTM_FCN(in_ch=feature_size, anomaly_weight=anomaly_weight).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.8,  # multiply LR by 0.5
        patience=2,  # wait 3 epochs with no improvement
        threshold=1e-4,  # improvement threshold
        min_lr=1e-6,  # min LR clamp
    )



    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0


    for epoch in range(max_epochs):
        model.train()
        # for Xb, yb in tqdm(train_loader, desc=f"Epoch{epoch}"):
        train_loss = 0.0
        train_seen = 0
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            loss = model(Xb, yb)
            # breakpoint()
            # print(loss.item())
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
            best_state = model.state_dict()
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




def _test_shapes():
    torch.manual_seed(0)
    B, Q, M = 4, 450, 2
    num_classes = 2

    x = torch.randn(B, Q, M)
    # example labels (per timestep)
    y = torch.randint(0, num_classes, (B, Q))
    # example mask: last 50 steps padded for sample 0
    mask = torch.ones(B, Q, dtype=torch.long)
    mask[0, -50:] = 0
    y[0, -50:] = -1  # optional; compute_loss can also set via mask

    model = MLSTM_FCN(
        num_vars=M,
        num_classes=num_classes,
        lstm_hidden=128,
        lstm_layers=1,
        dropout=0.1,
        bidirectional=True,
    )

    logits = model(x, mask=mask)
    print("logits:", logits.shape)  # [B, Q, num_classes]

    # loss = model.compute_loss(logits, y, mask=mask, ignore_index=-1)
    # print("loss:", float(loss))


if __name__ == "__main__":
    _test_shapes()