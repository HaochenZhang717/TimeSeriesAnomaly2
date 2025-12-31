import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score




class LSTM_Model(nn.Module):
    def __init__(
        self,
        num_vars: int,
        lstm_hidden: int = 128,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.rnn = nn.LSTM(
            input_size=num_vars,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=bidirectional,
        )
        out_dim = lstm_hidden * (2 if bidirectional else 1)
        self.head = nn.Linear(out_dim, 1)

    def forward(self, x, mask=None):
        # x: [B, T, M]
        h, _ = self.rnn(x)               # [B, T, H]
        logits = self.head(h).squeeze(-1)  # [B, T]
        return logits


class GRU_Model(nn.Module):
    def __init__(
        self,
        num_vars: int,
        lstm_hidden: int = 128,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.rnn = nn.GRU(
            input_size=num_vars,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=bidirectional,
        )
        out_dim = lstm_hidden * (2 if bidirectional else 1)
        self.head = nn.Linear(out_dim, 1)

    def forward(self, x, mask=None):
        # x: [B, T, M]
        h, _ = self.rnn(x)               # [B, T, H]
        logits = self.head(h).squeeze(-1)  # [B, T]
        return logits


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
            num_classes,
            kernel_size=1,
        )

    def forward(self, x):
        """
        x: [B, C, T]
        return logits: [B, 2, T]
        """
        feat = self.backbone(x)
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



class WrappedLSTM(nn.Module):
    def __init__(self, in_ch, anomaly_weight):
        super().__init__()
        self.model = LSTM_Model(
            num_vars=in_ch,
            lstm_hidden= 128,
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


class WrappedGRU(nn.Module):
    def __init__(self, in_ch, anomaly_weight):
        super().__init__()
        self.model = GRU_Model(
            num_vars=in_ch,
            lstm_hidden= 128,
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


def calculate_LSTM(
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

    model = WrappedLSTM(in_ch=feature_size, anomaly_weight=anomaly_weight).to(device)
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



def calculate_GRU(
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

    model = WrappedGRU(in_ch=feature_size, anomaly_weight=anomaly_weight).to(device)
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


def calculate_TCN(
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

    model = WrappedTCN(in_ch=feature_size, anomaly_weight=anomaly_weight).to(device)
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


# def _test_shapes():
#     torch.manual_seed(0)
#     B, Q, M = 4, 450, 2
#     num_classes = 2
#
#     x = torch.randn(B, Q, M)
#     # y = torch.ceil(torch.rand(B, Q))
#     # example labels (per timestep)
#     y = torch.randint(0, num_classes, (B, Q))
#     # example mask: last 50 steps padded for sample 0
#     mask = torch.ones(B, Q, dtype=torch.long)
#     mask[0, -50:] = 0
#     y[0, -50:] = -1  # optional; compute_loss can also set via mask
#
#     model = MLSTM_FCN(
#         in_ch=M,
#         anomaly_weight=1.0
#     )
#
#     logits = model(x, y)
#     print("logits:", logits.shape)  # [B, Q, num_classes]
#
#     # loss = model.compute_loss(logits, y, mask=mask, ignore_index=-1)
#     # print("loss:", float(loss))


# if __name__ == "__main__":
#     _test_shapes()