import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader



class GRUPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=1, pred_len=1):
        super().__init__()
        self.pred_len = pred_len
        self.gru = nn.GRU(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        """
        x: [B, T, C]
        return: [B, pred_len, C]
        """
        out, _ = self.gru(x)              # [B, T, H]
        h_last = out            # [B, H]
        pred = self.fc(h_last)            # [B, C]
        return pred       # [B, 1, C]




# def calculate_predictive_score(
#         feature_size,
#         ori_data, orig_pad_mask,
#         gen_data, gen_pad_mask,
#         device, lr,
#         max_epochs=2000,
#         batch_size=64,
# ):
#     signal_real = torch.tensor(ori_data, dtype=torch.float32)
#     signal_fake = torch.tensor(gen_data, dtype=torch.float32)
#
#     mask_real = torch.tensor(orig_pad_mask, dtype=torch.float32)
#     mask_fake = torch.tensor(gen_pad_mask, dtype=torch.float32)
#
#
#     train_ds = TensorDataset(signal_fake, mask_fake)
#     test_ds = TensorDataset(signal_real, mask_real)
#
#     train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
#     test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, drop_last=False)
#
#     model = GRUPredictor(input_dim=feature_size).to(device)
#
#     optimizer = torch.optim.Adam(model.parameters(), lr=lr)
#     scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#         optimizer,
#         mode='min',
#         factor=0.8,  # multiply LR by 0.5
#         patience=5,  # wait 3 epochs with no improvement
#         threshold=1e-4,  # improvement threshold
#         min_lr=1e-6,  # min LR clamp
#     )
#
#     best_val_loss = float('inf')
#
#     for epoch in range(max_epochs):
#         model.train()
#         train_loss = 0.0
#         train_seen = 0
#         for signal_batch, mask_batch in train_loader:
#             signal_batch = signal_batch.to(device)
#             mask_batch = mask_batch.to(device)[:, 1:].unsqueeze(-1)
#
#             model_out = model(signal_batch[:, :-1])
#             loss = nn.MSELoss(reduction="none")(model_out, signal_batch[:, 1:])
#             loss = (loss * mask_batch).sum() / mask_batch.sum()
#
#             train_loss += loss.item() * mask_batch.shape[0]
#             train_seen += mask_batch.shape[0]
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#         train_loss_avg = train_loss / train_seen
#
#         # -----------------------
#         # Evaluate
#         # -----------------------
#         model.eval()
#         val_loss = 0
#         val_seen = 0
#
#         for signal_batch, mask_batch in test_loader:
#             signal_batch = signal_batch.to(device)
#             mask_batch = mask_batch.to(device)[:, 1:].unsqueeze(-1)
#
#             with torch.no_grad():
#                 model_out = model(signal_batch[:, :-1])
#             loss = nn.MSELoss(reduction="none")(model_out, signal_batch[:, 1:])
#             loss = (loss * mask_batch).sum() / mask_batch.sum()
#
#             val_loss += loss.item() * mask_batch.shape[0]
#             val_seen += mask_batch.shape[0]
#
#         val_loss = val_loss / val_seen
#
#         print(f"Epoch{epoch}: train_loss: {train_loss_avg} | val_loss: {val_loss} | lr: {optimizer.param_groups[0]['lr']} ||")
#         scheduler.step(val_loss)
#
#         if best_val_loss > val_loss:
#             best_val_loss = val_loss
#
#     return best_val_loss

def calculate_predictive_score(
        ori_data,
        gen_data,
        device, lr,
        max_epochs=2000,
        batch_size=64,
):
    signal_real = torch.tensor(ori_data, dtype=torch.float32)
    signal_fake = torch.tensor(gen_data, dtype=torch.float32)
    feature_size = signal_real.shape[-1]
    # mask_real = torch.tensor(orig_pad_mask, dtype=torch.float32)
    # mask_fake = torch.tensor(gen_pad_mask, dtype=torch.float32)


    train_ds = TensorDataset(signal_fake)
    test_ds = TensorDataset(signal_real)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, drop_last=False)

    model = GRUPredictor(input_dim=feature_size).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.8,  # multiply LR by 0.5
        patience=5,  # wait 3 epochs with no improvement
        threshold=1e-4,  # improvement threshold
        min_lr=1e-6,  # min LR clamp
    )

    best_val_loss = float('inf')

    for epoch in range(max_epochs):
        model.train()
        train_loss = 0.0
        train_seen = 0
        for signal_batch in train_loader:
            signal_batch = signal_batch[0].to(device)

            model_out = model(signal_batch[:, :-1])
            loss = nn.MSELoss()(model_out, signal_batch[:, 1:])

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
        val_loss = 0
        val_seen = 0

        for signal_batch in test_loader:

            signal_batch = signal_batch[0].to(device)

            with torch.no_grad():
                model_out = model(signal_batch[:, :-1])
            loss = nn.MSELoss()(model_out, signal_batch[:, 1:])

            val_loss += loss.item() * signal_batch.shape[0]
            val_seen += signal_batch.shape[0]

        val_loss = val_loss / val_seen

        print(f"Epoch{epoch}: train_loss: {train_loss_avg} | val_loss: {val_loss} | lr: {optimizer.param_groups[0]['lr']} ||")
        scheduler.step(val_loss)

        if best_val_loss > val_loss:
            best_val_loss = val_loss

    return best_val_loss


if __name__ == '__main__':
    inputs_ts = torch.randn(2, 512, 1)
    pad_mask = torch.ones(2, 512)
    model = GRUPredictor(input_dim=1)
    out = model(inputs_ts)
    print(out.shape)
    print(inputs_ts.shape)
