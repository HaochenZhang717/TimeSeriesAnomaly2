import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader


# -----------------------------
# 基础模块：Double Conv (Conv → ReLU → Conv → ReLU)
# -----------------------------
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


# -----------------------------
# 上采样模块：UpConv + DoubleConv
# -----------------------------
class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose1d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)

        # 修正长度不一致问题（常见）
        if x.size(-1) != skip.size(-1):
            diff = skip.size(-1) - x.size(-1)
            x = F.pad(x, (0, diff))

        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


# class Discriminator(nn.Module):
#     def __init__(self, in_ch, pooling="mean"):
#         super().__init__()
#         self.pooling = pooling
#
#         # -------- Encoder --------
#         self.conv1 = DoubleConv(in_ch, 16)
#         self.conv2 = DoubleConv(16, 32)
#         self.conv3 = DoubleConv(32, 64)
#         self.conv4 = DoubleConv(64, 128)
#         self.conv5 = DoubleConv(128, 256)
#         self.pool = nn.MaxPool1d(2)
#
#         # -------- Decoder --------
#         self.up4 = UpBlock(256, 128)
#         self.up3 = UpBlock(128, 64)
#         self.up2 = UpBlock(64, 32)
#         self.up1 = UpBlock(32, 16)
#
#         # -------- Classification head --------
#         self.proj = nn.Conv1d(16, 128, kernel_size=1, bias=False)  # feature projection
#         self.classifier = nn.Linear(128, 1)
#
#     def forward(self, x, pad_mask):
#         """
#         x: (B, T, C)
#         pad_mask: (B, T)
#         """
#         x = x.permute(0, 2, 1)  # (B, C, T)
#
#         # ---- Encoder ----
#         c1 = self.conv1(x)
#         p1 = self.pool(c1)
#
#         c2 = self.conv2(p1)
#         p2 = self.pool(c2)
#
#         c3 = self.conv3(p2)
#         p3 = self.pool(c3)
#
#         c4 = self.conv4(p3)
#         p4 = self.pool(c4)
#
#         c5 = self.conv5(p4)
#
#         # ---- Decoder ----
#         u4 = self.up4(c5, c4)
#         u3 = self.up3(u4, c3)
#         u2 = self.up2(u3, c2)
#         u1 = self.up1(u2, c1)   # (B,16,T)
#
#         feat = self.proj(u1)    # (B,128,T)
#         feat = feat * pad_mask.unsqueeze(1)
#         # ---- Temporal pooling ----
#         if self.pooling == "mean":
#             feat = feat.mean(dim=-1)
#         elif self.pooling == "max":
#             feat = feat.max(dim=-1).values
#         elif self.pooling == "mean_max":
#             feat = torch.cat(
#                 [feat.mean(dim=-1), feat.max(dim=-1).values],
#                 dim=1
#             )
#         else:
#             raise ValueError("Unknown pooling type")
#
#         logits = self.classifier(feat)  # (B, num_classes)
#         return logits


# def calculate_discriminator_score(
#         feature_size,
#         ori_data, orig_pad_mask,
#         gen_data, gen_pad_mask,
#         device, lr,
#         max_epochs=2000,
#         batch_size=64,
# ):
#     signal_real = torch.tensor(ori_data, dtype=torch.float32)
#     signal_fake = torch.tensor(gen_data, dtype=torch.float32)
#     signal = torch.cat((signal_real, signal_fake), dim=0)
#
#     mask_real = torch.tensor(orig_pad_mask, dtype=torch.float32)
#     mask_fake = torch.tensor(gen_pad_mask, dtype=torch.float32)
#     mask = torch.cat((mask_real, mask_fake), dim=0)
#
#     ori_labels = torch.ones_like(mask_real).sum(1)
#     gen_labels = torch.zeros_like(mask_fake).sum(1)
#     label_real = torch.tensor(ori_labels, dtype=torch.float32)
#     label_fake = torch.tensor(gen_labels, dtype=torch.float32)
#     label = torch.cat((label_real, label_fake), dim=0)
#
#     train_ds = TensorDataset(signal, mask, label)
#
#     train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
#
#     model = Discriminator(in_ch=feature_size).to(device)
#
#     optimizer = torch.optim.Adam(model.parameters(), lr=lr)
#     scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#         optimizer,
#         mode='max',
#         factor=0.8,  # multiply LR by 0.5
#         patience=2,  # wait 3 epochs with no improvement
#         threshold=1e-4,  # improvement threshold
#         min_lr=1e-6,  # min LR clamp
#     )
#
#     best_val_acc = 0.0
#
#     for epoch in range(max_epochs):
#         model.train()
#         train_loss = 0.0
#         train_seen = 0
#         for signal_batch, mask_batch, label_batch in train_loader:
#             signal_batch = signal_batch.to(device)
#             mask_batch = mask_batch.to(device)
#             label_batch = label_batch.to(device)
#
#             logits = model(signal_batch, mask_batch)
#             loss = nn.BCEWithLogitsLoss()(logits, label_batch.float())
#
#             train_loss += loss.item() * signal_batch.shape[0]
#             train_seen += signal_batch.shape[0]
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#         train_loss_avg = train_loss / train_seen
#
#         # -----------------------
#         # Evaluate
#         # -----------------------
#         model.eval()
#         val_correct = 0
#         val_seen = 0
#
#         for signal_batch, mask_batch, label_batch in train_loader:
#             signal_batch = signal_batch.to(device)
#             mask_batch = mask_batch.to(device)
#             label_batch = label_batch.to(device)
#
#             with torch.no_grad():
#                 logits = model(signal_batch, mask_batch)  # [B] or [B,1]
#
#                 logits = logits.squeeze(-1)  # [B]
#                 probs = torch.sigmoid(logits)  # [B]
#                 preds = (probs > 0.5).long()  # {0,1}
#
#             val_correct += (preds == label_batch).sum().item()
#             val_seen += label_batch.numel()
#
#         val_accuracy = val_correct / val_seen
#
#         print(f"Epoch{epoch}: train_loss: {train_loss_avg} | val_acc: {val_accuracy} | lr: {optimizer.param_groups[0]['lr']} ||")
#         scheduler.step(val_accuracy)
#
#         if best_val_acc < val_accuracy:
#             best_val_acc = val_accuracy
#
#     return best_val_acc



class Discriminator(nn.Module):
    def __init__(self, in_ch, pooling="mean"):
        super().__init__()
        self.pooling = pooling

        # -------- Encoder --------
        self.conv1 = DoubleConv(in_ch, 16)
        self.conv2 = DoubleConv(16, 32)
        self.conv3 = DoubleConv(32, 64)
        self.conv4 = DoubleConv(64, 128)
        self.conv5 = DoubleConv(128, 256)
        self.pool = nn.MaxPool1d(2)

        # -------- Decoder --------
        self.up4 = UpBlock(256, 128)
        self.up3 = UpBlock(128, 64)
        self.up2 = UpBlock(64, 32)
        self.up1 = UpBlock(32, 16)

        # -------- Classification head --------
        self.proj = nn.Conv1d(16, 128, kernel_size=1, bias=False)  # feature projection
        self.classifier = nn.Linear(128, 1)

    def forward(self, x):
        """
        x: (B, T, C)
        """
        x = x.permute(0, 2, 1)  # (B, C, T)

        # ---- Encoder ----
        c1 = self.conv1(x)
        p1 = self.pool(c1)

        c2 = self.conv2(p1)
        p2 = self.pool(c2)

        c3 = self.conv3(p2)
        p3 = self.pool(c3)

        c4 = self.conv4(p3)
        p4 = self.pool(c4)

        c5 = self.conv5(p4)

        # ---- Decoder ----
        u4 = self.up4(c5, c4)
        u3 = self.up3(u4, c3)
        u2 = self.up2(u3, c2)
        u1 = self.up1(u2, c1)   # (B,16,T)

        feat = self.proj(u1)    # (B,128,T)

        # ---- Temporal pooling ----
        if self.pooling == "mean":
            feat = feat.mean(dim=-1)
        elif self.pooling == "max":
            feat = feat.max(dim=-1).values
        elif self.pooling == "mean_max":
            feat = torch.cat(
                [feat.mean(dim=-1), feat.max(dim=-1).values],
                dim=1
            )
        else:
            raise ValueError("Unknown pooling type")

        logits = self.classifier(feat)  # (B, num_classes)
        return logits



def calculate_discriminator_score(
        ori_data,
        gen_data,
        device, lr,
        max_epochs=2000,
        batch_size=64,
):
    feature_size = ori_data.shape[-1]
    signal_real = torch.tensor(ori_data, dtype=torch.float32)
    signal_fake = torch.tensor(gen_data, dtype=torch.float32)
    signal = torch.cat((signal_real, signal_fake), dim=0)

    ori_labels = torch.ones_like(signal_real).sum(dim=(1,2)).unsqueeze(-1)
    gen_labels = torch.zeros_like(signal_fake).sum(dim=(1,2)).unsqueeze(-1)
    label_real = torch.tensor(ori_labels, dtype=torch.float32)
    label_fake = torch.tensor(gen_labels, dtype=torch.float32)
    label = torch.cat((label_real, label_fake), dim=0)

    train_ds = TensorDataset(signal, label)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)

    model = Discriminator(in_ch=feature_size).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.8,  # multiply LR by 0.5
        patience=2,  # wait 3 epochs with no improvement
        threshold=1e-4,  # improvement threshold
        min_lr=1e-6,  # min LR clamp
    )

    best_val_acc = 0.0

    for epoch in range(max_epochs):
        model.train()
        train_loss = 0.0
        train_seen = 0
        for signal_batch, label_batch in train_loader:
            signal_batch = signal_batch.to(device)
            label_batch = label_batch.to(device)

            logits = model(signal_batch)
            loss = nn.BCEWithLogitsLoss()(logits, label_batch.float())

            train_loss += loss.item() * signal_batch.shape[0]
            train_seen += signal_batch.shape[0]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        train_loss_avg = train_loss / train_seen

        # -----------------------
        # Evaluate
        # -----------------------
        model.eval()
        val_correct = 0
        val_seen = 0

        for signal_batch, label_batch in train_loader:
            signal_batch = signal_batch.to(device)
            label_batch = label_batch.to(device)

            with torch.no_grad():
                logits = model(signal_batch)  # [B] or [B,1]

                logits = logits.squeeze(-1)  # [B]
                probs = torch.sigmoid(logits)  # [B]
                preds = (probs > 0.5).long()  # {0,1}

            val_correct += (preds == label_batch).sum().item()
            val_seen += label_batch.numel()

        val_accuracy = val_correct / val_seen

        print(f"Epoch{epoch}: train_loss: {train_loss_avg} | val_acc: {val_accuracy} | lr: {optimizer.param_groups[0]['lr']} ||")
        scheduler.step(val_accuracy)

        if best_val_acc < val_accuracy:
            best_val_acc = val_accuracy

    return best_val_acc


if __name__ == '__main__':
    inputs_ts = torch.randn(2, 512, 1)
    pad_mask = torch.ones(2, 512)
    model = Discriminator(in_ch=1)
    out = model(inputs_ts)
    print(out.shape)
