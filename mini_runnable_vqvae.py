import os
import json
import math
import random
import numpy as np
from dataclasses import dataclass
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import MinMaxScaler
import wandb

import torch
import numpy as np
from collections import defaultdict
from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Subset

# --------------------------
# Dataset (copied from you)
# --------------------------

def load_jsonl(path):
    data = []
    with open(path, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data


class AnomalyDataset(Dataset):
    def __init__(
            self,
            raw_data_paths,
            indices_paths,
            one_channel,
            max_length,
            min_length,
            data_type,
    ):
        super().__init__()
        self.raw_data_paths = raw_data_paths
        self.indices_paths = indices_paths
        self.one_channel = one_channel
        self.max_length = max_length
        self.min_length = min_length
        self.data_type = data_type

        if data_type == "ecg":
            self.normed_signal_list = []
            self.index_lines_list = []

            for raw_data_path, indices_path in zip(self.raw_data_paths, self.indices_paths):
                raw_data = np.load(raw_data_path)
                raw_signal = raw_data["signal"]

                scaler = MinMaxScaler()
                normed_signal = scaler.fit_transform(raw_signal)
                index_lines = load_jsonl(indices_path)

                self.normed_signal_list.append(normed_signal)
                self.index_lines_list.append(index_lines)

        elif data_type == "ercot":
            self.normed_signal_list = []
            self.index_lines_list = []
            for raw_data_path, indices_path in zip(self.raw_data_paths, self.indices_paths):
                raw_data = np.load(raw_data_path)
                raw_signal = np.expand_dims(raw_data, axis=-1)
                scaler = MinMaxScaler()
                normed_signal = scaler.fit_transform(raw_signal)
                index_lines = load_jsonl(indices_path)
                self.normed_signal_list.append(normed_signal)
                self.index_lines_list.append(index_lines)

        else:
            raise NotImplementedError

        self.global_index = []
        for region_id, index_lines in enumerate(self.index_lines_list):
            for i in range(len(index_lines)):
                self.global_index.append((region_id, i))


    def __len__(self):
        return len(self.global_index)

    def __getitem__(self, index):
        if self.data_type == "ecg":
            which_list, which_index = self.global_index[index]

            # start, end = self.index_lines[index]
            start = self.index_lines_list[which_list][which_index]["start"]
            if "source_file" in self.index_lines_list[which_list][which_index].keys():
                # this is a normal data, we apply random length
                random_length = random.randint(self.min_length, self.max_length)
                end = start + random_length
            else: # this is anomaly data, we use fix length
                end = self.index_lines_list[which_list][which_index]["end"]

            assert end - start <= self.max_length

            data_in_use = self.normed_signal_list[which_list]
            if self.one_channel:
                data_in_use = data_in_use[:, :1]
            ts_dim = data_in_use.shape[1]
            signal = torch.zeros(self.max_length, ts_dim, dtype=torch.float32)
            signal[:end-start] = torch.from_numpy(data_in_use[start:end]).float()
            pad_mask = torch.zeros(self.max_length, 1)
            pad_mask[:end-start] = 1
            if self.one_channel:
                return signal[:, :1], pad_mask
            else:
                return signal, pad_mask

        elif self.data_type == "ercot":
            which_list, which_index = self.global_index[index]
            start = self.index_lines_list[which_list][which_index]["start"]
            random_length = random.randint(self.min_length, self.max_length)
            end = start + random_length

            assert end - start <= self.max_length

            data_in_use = self.normed_signal_list[which_list]
            if self.one_channel:
                data_in_use = data_in_use[:, :1]
            ts_dim = data_in_use.shape[1]
            signal = torch.zeros(self.max_length, ts_dim, dtype=torch.float32)
            signal[:end - start] = torch.from_numpy(data_in_use[start:end]).float()
            pad_mask = torch.zeros(self.max_length, 1)
            pad_mask[:end - start] = 1
            if self.one_channel:
                return signal[:, :1], pad_mask
            else:
                return signal, pad_mask

        else:
            raise NotImplementedError



class MixedAugmentedDataset(Dataset):
    def __init__(
            self,
            normal_data_paths,
            normal_indices_paths,
            anomaly_data_paths,
            anomaly_indices_paths,
            one_channel,
            max_length,
            min_length,
            data_type,
    ):
        super().__init__()
        self.normal_data_paths = normal_data_paths
        self.normal_indices_paths = normal_indices_paths

        self.anomaly_data_paths = anomaly_data_paths
        self.anomaly_indices_paths = anomaly_indices_paths

        self.one_channel = one_channel
        self.max_length = max_length
        self.min_length = min_length
        self.data_type = data_type

        self.normal_signal_list, self.normal_index_lines_list = \
            self.load_data(normal_data_paths, normal_indices_paths)

        self.anomaly_signal_list, self.anomaly_index_lines_list = \
            self.load_data(anomaly_data_paths, anomaly_indices_paths)

        self.global_index = []
        for region_id, index_lines in enumerate(self.normal_signal_list):
            for i in range(len(index_lines)):
                self.global_index.append((region_id, i))

    def load_data(self, raw_data_paths, indices_paths):
        if self.data_type == "ecg":
            signal_list = []
            index_lines_list = []

            for raw_data_path, indices_path in zip(raw_data_paths, indices_paths):
                raw_data = np.load(raw_data_path)
                raw_signal = raw_data["signal"]

                scaler = MinMaxScaler()
                normed_signal = scaler.fit_transform(raw_signal)
                index_lines = load_jsonl(indices_path)

                signal_list.append(normed_signal)
                index_lines_list.append(index_lines)
            return signal_list, index_lines_list

        elif self.data_type == "ercot":
            signal_list = []
            index_lines_list = []
            for raw_data_path, indices_path in zip(raw_data_paths, indices_paths):
                raw_data = np.load(raw_data_path)
                raw_signal = np.expand_dims(raw_data, axis=-1)
                scaler = MinMaxScaler()
                normed_signal = scaler.fit_transform(raw_signal)
                index_lines = load_jsonl(indices_path)
                signal_list.append(normed_signal)
                index_lines_list.append(index_lines)
            return signal_list, index_lines_list

        else:
            raise NotImplementedError(self.data_type)


    def __len__(self):
        return len(self.global_index)

    def __getitem__(self, index):
        # ------------------------------
        # sample a normal time series
        # ------------------------------
        which_list, which_index = self.global_index[index]

        start = self.normal_index_lines_list[which_list][which_index]["start"]
        random_length = random.randint(self.min_length, self.max_length)
        end = start + random_length

        assert end - start <= self.max_length

        data_in_use = self.normal_signal_list[which_list]
        if self.one_channel:
            data_in_use = data_in_use[:, :1]
        ts_dim = data_in_use.shape[1]
        signal = torch.zeros(self.max_length, ts_dim, dtype=torch.float32)
        signal[:end-start] = torch.from_numpy(data_in_use[start:end]).float()
        pad_mask = torch.zeros(self.max_length, 1)
        pad_mask[:end-start] = 1
        if self.one_channel:
            signal = signal[:, :1]

        # ------------------------------
        # sample an anomaly time series
        # ------------------------------
        anomaly_list_id = random.randint(0, len(self.anomaly_signal_list) - 1)
        anomaly_indices = self.anomaly_index_lines_list[anomaly_list_id]
        anomaly_signal_np = self.anomaly_signal_list[anomaly_list_id]

        anomaly_idx = random.randint(0, len(anomaly_indices) - 1)
        anomaly_info = anomaly_indices[anomaly_idx]

        a_start = anomaly_info["start"]
        a_end = a_start + random_length

        anomaly_signal = torch.zeros(self.max_length, ts_dim, dtype=torch.float32)
        anomaly_signal[:random_length] = torch.from_numpy(anomaly_signal_np[a_start:a_end]).float()

        if self.one_channel:
            anomaly_signal = anomaly_signal[:, :1]

        # add them together.
        lam = random.uniform(0.0, 1.0)

        return (1-lam)*anomaly_signal + lam*signal, pad_mask





# --------------------------
# Utils: pad mask
# --------------------------

def make_valid_mask(lengths: torch.Tensor, T: int) -> torch.Tensor:
    """
    lengths: [B]
    returns: [B, T] float mask, 1 for valid, 0 for pad
    """
    device = lengths.device
    return (torch.arange(T, device=device)[None, :] < lengths[:, None]).float()


# --------------------------
# Vector Quantizer (VQ-VAE)
# --------------------------

class VectorQuantizer(nn.Module):
    """
    Standard VQ-VAE (nearest neighbor) with straight-through gradient.

    z_e: [B, T', D]  (encoder outputs)
    z_q: [B, T', D]  (quantized)
    ids: [B, T']     (code indices)
    loss: vq loss (codebook + commitment)
    """
    def __init__(self, num_codes: int, code_dim: int, beta: float = 0.25):
        super().__init__()
        self.num_codes = num_codes
        self.code_dim = code_dim
        self.beta = beta

        self.codebook = nn.Embedding(num_codes, code_dim)
        nn.init.uniform_(self.codebook.weight, -1.0 / num_codes, 1.0 / num_codes)

    def forward(self, z_e: torch.Tensor):
        B, T, D = z_e.shape
        assert D == self.code_dim

        flat = z_e.reshape(B * T, D)  # [BT, D]
        e = self.codebook.weight      # [K, D]

        # squared euclidean distances: ||x||^2 - 2 x·e + ||e||^2
        dist = (
            flat.pow(2).sum(dim=1, keepdim=True)
            - 2.0 * flat @ e.t()
            + e.pow(2).sum(dim=1, keepdim=True).t()
        )  # [BT, K]

        ids = torch.argmin(dist, dim=1)             # [BT]
        z_q = self.codebook(ids).view(B, T, D)      # [B, T, D]

        # VQ losses
        codebook_loss = F.mse_loss(z_q, z_e.detach())
        commit_loss   = F.mse_loss(z_e, z_q.detach())
        vq_loss = codebook_loss + self.beta * commit_loss

        # Straight-through estimator
        z_q_st = z_e + (z_q - z_e).detach()

        return z_q_st, ids.view(B, T), vq_loss


# --------------------------
# 1D Conv VQ-VAE
# --------------------------

class ResBlock1D(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        downsample=False,
    ):
        super().__init__()
        padding = kernel_size // 2
        # self.downsample = downsample or (stride != 1) or (in_channels != out_channels)

        self.conv1 = nn.Conv1d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            bias=False,
        )
        self.conv2 = nn.Conv1d(
            out_channels, out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            bias=False,
        )

        self.norm1 = nn.BatchNorm1d(out_channels)
        self.norm2 = nn.BatchNorm1d(out_channels)
        self.act = nn.ReLU(inplace=True)

        # if self.downsample:
        #     self.skip = nn.Conv1d(
        #         in_channels, out_channels,
        #         kernel_size=1,
        #         stride=stride,
        #         bias=False,
        #     )
        # else:
        #     self.skip = nn.Identity()
        self.skip = nn.Identity()

    def forward(self, x):
        identity = self.skip(x)

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act(out)

        out = self.conv2(out)
        out = self.norm2(out)

        out = out + identity
        out = self.act(out)
        return out


class ResNetEncoder1D(nn.Module):
    """
    Input:  x [B, T, C]
    Output: z_e [B, T', D]
    """
    def __init__(
        self,
        seq_len,
        in_channels,
        channels=(64, 128, 256),
        blocks_per_stage=2,
        code_dim=128,
        kernel_size=3,
        down_ratio=2,
        code_len=4
    ):
        super().__init__()

        self.stem = nn.Conv1d(
            in_channels, channels[0],
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            bias=False,
        )

        stages = []
        in_ch = channels[0]

        for stage_idx, out_ch in enumerate(channels):
            for block_idx in range(blocks_per_stage):
                # downsample only at first block of each stage (except stage 0)
                stride = down_ratio if (block_idx == 0 and stage_idx > 0) else 1
                stages.append(
                    ResBlock1D(
                        in_ch, out_ch,
                        kernel_size=kernel_size,
                        stride=stride,
                    )
                )
                in_ch = out_ch

        self.stages = nn.Sequential(*stages)

        self.global_pooling = nn.Linear(seq_len, code_len)
        self.proj = nn.Conv1d(
            in_ch, code_dim,
            kernel_size=1,
            bias=False,
        )

    def forward(self, x, loss_mask):
        # x: [B, T, C] → [B, C, T]
        x = x.transpose(1, 2)
        h = self.stem(x)
        h = self.stages(h)
        h = h * loss_mask.permute(0, 2, 1)
        z = self.global_pooling(h)
        z = torch.relu(z)
        breakpoint()
        z = self.proj(z)          # [B, D, T']
        breakpoint()
        z = z.transpose(1, 2)     # [B, T', D]
        breakpoint()
        return z


class UpResBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, up_ratio):
        super().__init__()
        padding = kernel_size // 2

        self.upsample = nn.Upsample(scale_factor=up_ratio, mode="nearest")

        self.conv1 = nn.Conv1d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=False,
        )
        self.conv2 = nn.Conv1d(
            out_channels, out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=False,
        )

        self.norm1 = nn.BatchNorm1d(out_channels)
        self.norm2 = nn.BatchNorm1d(out_channels)
        self.act = nn.ReLU(inplace=True)

        self.skip = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.upsample(x)
        identity = self.skip(x)

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act(out)

        out = self.conv2(out)
        out = self.norm2(out)

        out = out + identity
        out = self.act(out)
        return out


class ResNetDecoder1D(nn.Module):
    """
    Input:  z_q [B, T', D]
    Output: x_hat [B, T, C]
    """
    def __init__(
        self,
        out_channels,
        channels=(256, 128, 64),
        blocks_per_stage=1,
        code_dim=128,
        kernel_size=3,
        up_ratio=2,
        code_len=123,
        seq_len=123
    ):
        super().__init__()

        stages = []
        in_ch = channels[0]

        for out_ch in channels[1:]:
            for _ in range(blocks_per_stage):
                stages.append(
                    UpResBlock1D(
                        in_ch, out_ch,
                        kernel_size=kernel_size,
                        up_ratio=up_ratio,
                    )
                )
                in_ch = out_ch

        self.stages = nn.Sequential(*stages)

        self.head = nn.Conv1d(
            in_ch, out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=False,
        )

        n_upsample = len(channels) - 1
        T0 = math.ceil(seq_len / (up_ratio ** n_upsample))

        self.input_proj_1 = nn.Linear(code_dim, channels[0])   # channel
        self.input_proj_2 = nn.Linear(code_len, T0)            # time


    def forward(self, z, target_len):
        '''
        z: [B, T, C]
        '''
        z = self.input_proj_1(z)

        z = z.transpose(1, 2)           # [B, C0, 4]
        z = self.input_proj_2(z)
        h = self.stages(z)
        x_hat = self.head(h)       # [B, C, T_recon]
        x_hat = x_hat.transpose(1, 2)

        # crop / pad to target_len
        if x_hat.size(1) > target_len:
            x_hat = x_hat[:, :target_len]
        elif x_hat.size(1) < target_len:
            x_hat = F.pad(x_hat, (0, 0, 0, target_len - x_hat.size(1)))

        return x_hat


class VQVAE1D(nn.Module):
    def __init__(
        self,
        in_channels, encoder_channels, decoder_channels, code_dim, num_codes,
        down_ratio, up_ratio, code_len, seq_len
    ):
        super().__init__()
        # self.encoder = ResNetEncoder1D(
        #     in_channels=in_channels,
        #     channels=(16, 16, 32, 32, 64, 64),
        #     blocks_per_stage=1,
        #     code_dim=code_dim,
        # )
        self.encoder = ResNetEncoder1D(
            in_channels=in_channels,
            channels=encoder_channels,
            blocks_per_stage=1,
            code_dim=code_dim,
            down_ratio=down_ratio,
            code_len=code_len
        )
        self.quantizer = VectorQuantizer(num_codes, code_dim)
        # self.decoder = ResNetDecoder1D(
        #     out_channels=in_channels,
        #     channels=(64, 64, 32, 32, 16, 16),
        #     blocks_per_stage=1,
        #     code_dim=code_dim,
        # )
        self.decoder = ResNetDecoder1D(
            out_channels=in_channels,
            channels=decoder_channels,
            blocks_per_stage=1,
            code_dim=code_dim,
            up_ratio=up_ratio,
            code_len=code_len,
            seq_len=seq_len
        )

    def forward(self, x, loss_mask):
        z_e = self.encoder(x, loss_mask)
        z_q, ids, vq_loss = self.quantizer(z_e)
        x_hat = self.decoder(z_q, x.size(1))
        return x_hat, ids, vq_loss

# --------------------------
# Train / Eval
# --------------------------

@dataclass
class TrainConfig:
    encoder_channels: tuple
    decoder_channels: tuple
    down_ratio: int
    up_ratio: int
    min_length: int
    max_length: int


    # normal_data_paths: str
    # normal_indices_paths: str
    # anomaly_data_paths: str
    # anomaly_indices_paths: str

    data_paths: str
    indices_paths: str


    one_channel: int
    feat_size: int
    data_type: str = 'blahblah'




    batch_size: int = 64
    epochs: int = 50
    lr: float = 1e-3
    weight_decay: float = 1e-2

    hidden: int = 64
    code_dim: int = 128
    code_len: int = 4
    num_codes: int = 1024
    beta: float = 0.25

    recon_loss: str = "mse"   # "mse" or "smoothl1"
    vq_loss_weight: float = 1.0

    device: str = "cuda"
    save_path: str = "vqvae_1d.pt"




@torch.no_grad()
def extract_token_ids(model: VQVAE1D, loader: DataLoader, device: torch.device):
    """
    Returns list of (ids, lengths_Tprime), where ids is [B, T'].
    Note: lengths in T' are approx lengths/8 (floor-ish), so we compute it.
    """
    model.eval()
    all_ids = []
    all_tprime_lens = []

    for x, lengths in loader:
        x = x.to(device)
        lengths = lengths.to(device)
        _, ids, _, z_e, _ = model(x)  # ids [B, T']

        # Approx downsample factor = 8 for 3 stride-2 layers.
        # length after convs depends on padding; this approximation is usually ok.
        tprime = ids.size(1)
        # We can infer valid token lengths by mapping original valid mask through encoder length
        # Simple heuristic:
        tprime_lens = torch.clamp((lengths.float() / 8.0).floor().long(), min=1, max=tprime)

        all_ids.append(ids)
        all_tprime_lens.append(tprime_lens)

    return all_ids, all_tprime_lens


@torch.no_grad()
def _compute_codebook_stats(epoch_ids, num_codes):
    """
    epoch_ids: list of Tensor [B, T']
    """
    flat_ids = torch.cat([ids.reshape(-1) for ids in epoch_ids], dim=0)

    counts = torch.bincount(flat_ids, minlength=num_codes).float()
    total = counts.sum()

    active_code_ratio = (counts > 0).float().mean().item()

    probs = counts / (total + 1e-9)
    perplexity = torch.exp(
        -(probs * torch.log(probs + 1e-9)).sum()
    ).item()

    dead_codes = (counts == 0).sum().item()

    return {
        "active_code_ratio": active_code_ratio,
        "perplexity": perplexity,
        "dead_codes": dead_codes,
        "counts": counts.cpu().numpy(),
    }


def train_vqvae(cfg: TrainConfig):
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    os.makedirs(cfg.save_path, exist_ok=True)
    # -------- wandb --------
    wandb.init(
        project="vqvae-ts",
        name=f"vqvae_K{cfg.num_codes}_D{cfg.code_dim}",
        config=cfg.__dict__,
    )

    # -------- dataset / loader --------
    full_set = AnomalyDataset(
        raw_data_paths=cfg.data_paths,
        indices_paths=cfg.indices_paths,
        one_channel=cfg.one_channel,
        min_length=cfg.min_length,
        max_length=cfg.max_length,
        data_type=cfg.data_type
    )

    # full_set = MixedAugmentedDataset(
    #     normal_data_paths=cfg.normal_data_paths,
    #     normal_indices_paths=cfg.normal_indices_paths,
    #     anomaly_data_paths=cfg.anomaly_data_paths,
    #     anomaly_indices_paths=cfg.anomaly_indices_paths,
    #     one_channel=cfg.one_channel,
    #     min_length=cfg.min_length,
    #     max_length=cfg.max_length,
    #     data_type=cfg.data_type
    # )

    N = len(full_set)
    indices = np.arange(N)

    split = int(0.8 * N)
    train_idx = indices[:split]
    # val_idx = indices[split:]

    dataset = Subset(full_set, train_idx)
    # val_set = Subset(full_set, val_idx)

    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=True,
    )

    # -------- model --------
    if cfg.one_channel:
        in_channels = 1
    else:
        in_channels = cfg.feat_size

    model = VQVAE1D(
        in_channels=in_channels,
        encoder_channels=cfg.encoder_channels,
        decoder_channels=cfg.decoder_channels,
        code_dim=cfg.code_dim,
        num_codes=cfg.num_codes,
        down_ratio=cfg.down_ratio,
        up_ratio=cfg.up_ratio,
        code_len=cfg.code_len,
        seq_len=cfg.max_length
    ).to(device)

    optim = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    def recon_criterion(x_hat, x, valid_mask):
        # valid_mask: [B, T, 1]
        per = (x_hat - x) ** 2
        per = per * valid_mask
        return per.sum() / (valid_mask.sum() + 1e-6)

    # -------- training --------
    model.train()
    for ep in range(cfg.epochs):
        total = 0.0
        total_rec = 0.0
        total_vq = 0.0

        epoch_ids = []   # 👈 收集整个 epoch 的 code ids

        for x, loss_mask in loader:
            x = x.to(device)                 # [B, T, C]
            loss_mask = loss_mask.to(device)
            B, T, C = x.shape

            # valid = make_valid_mask(lengths, T).unsqueeze(-1)  # [B, T, 1]

            x_hat, ids, vq_loss = model(x, loss_mask)
            rec_loss = recon_criterion(x_hat, x, loss_mask)

            loss = rec_loss + cfg.vq_loss_weight * vq_loss

            optim.zero_grad()
            loss.backward()
            optim.step()

            total += loss.item()
            total_rec += rec_loss.item()
            total_vq += vq_loss.item()

            epoch_ids.append(ids.detach())

        # -------- codebook stats --------
        stats = _compute_codebook_stats(epoch_ids, cfg.num_codes)

        # -------- logging --------
        avg_loss = total / len(loader)
        avg_rec = total_rec / len(loader)
        avg_vq = total_vq / len(loader)

        wandb.log({
            "epoch": ep + 1,
            "loss": avg_loss,
            "recon_loss": avg_rec,
            "vq_loss": avg_vq,
            "active_code_ratio": stats["active_code_ratio"],
            "perplexity": stats["perplexity"],
            "dead_codes": stats["dead_codes"],
        })

        # 每 5 个 epoch 画一次 usage histogram
        if (ep + 1) % 5 == 0:
            wandb.log({
                "codebook_usage_hist": wandb.Histogram(stats["counts"])
            })

        print(
            f"[Epoch {ep+1:03d}/{cfg.epochs}] "
            f"loss={avg_loss:.6f}  "
            f"rec={avg_rec:.6f}  "
            f"vq={avg_vq:.6f}  "
            f"active={stats['active_code_ratio']:.3f}  "
            f"ppl={stats['perplexity']:.1f}  "
            f"dead={stats['dead_codes']}"
        )

    # -------- save --------
    # os.makedirs(cfg.save_path, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": cfg.__dict__,
        },
        f"{cfg.save_path}/vqvae.pt",
    )
    print(f"Saved: {cfg.save_path}")

    wandb.finish()
    return model



@torch.no_grad()
def extract_code_segments(
    in_channels,
    code_dim,
    num_codes,
    model_path,
    raw_data_paths,
    indices_paths,
    data_type,
    one_channel,
    device,
    save_path,
    down_ratio,
    up_ratio,
    max_length,
    min_length,
    encoder_channels,
    decoder_channels,
    code_len,
):
    model = VQVAE1D(
        in_channels=in_channels,
        code_dim=code_dim,
        num_codes=num_codes,
        down_ratio=down_ratio,
        up_ratio=up_ratio,
        encoder_channels=encoder_channels,
        decoder_channels=decoder_channels,
        code_len=code_len,
        seq_len=max_length,
    ).to(device)
    model.eval()
    ckpt = torch.load(model_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])

    # model.load_state_dict(torch.load(model_path, map_location=device))

    code_segments = defaultdict(list)

    dataset = AnomalyDataset(
        raw_data_paths=raw_data_paths,
        indices_paths=indices_paths,
        one_channel=one_channel,
        max_length=max_length,
        min_length=min_length,
        data_type=data_type,
    )

    loader = DataLoader(
        dataset,
        batch_size=512,
        shuffle=False,
        drop_last=False,
    )

    all_codes = []
    for x, loss_mask in tqdm(loader, desc="Extracting code ids"):
        x = x.to(device)  # [B, T, C]
        lengths = loss_mask.sum((1,2)).to(device)

        z_e = model.encoder(x)
        _, ids, _ = model.quantizer(z_e)  # ids: [B, T']

        B, Tprime = ids.shape

        for b in range(B):
            entry = {
                "ids": ids[b].detach().cpu(),  # [T']
                "orig_len": int(lengths[b].item()),
                "tprime_len": int(Tprime),
            }
            all_codes.append(entry)

    torch.save(all_codes, save_path)
    print(f"[Saved] time-series codes -> {save_path}")



# -------------------------
# cluster analysis
# -------------------------
def find_nearest_neighbors(
    all_embeddings: torch.Tensor,
    anchor_idx: int,
    k: int = 5,
    metric: str = "l2",
):
    """
    all_embeddings: (N, D)
    anchor_idx: index in [0, N)
    k: number of nearest neighbors (excluding itself)
    metric: 'l2' or 'cosine'
    """
    anchor = all_embeddings[anchor_idx]  # (D,)

    if metric == "l2":
        dists = torch.norm(all_embeddings - anchor, dim=1)
    elif metric == "cosine":
        anchor_norm = anchor / anchor.norm()
        emb_norm = all_embeddings / all_embeddings.norm(dim=1, keepdim=True)
        dists = 1 - torch.matmul(emb_norm, anchor_norm)
    else:
        raise ValueError(f"Unknown metric {metric}")

    # 排序，去掉自己
    sorted_idx = torch.argsort(dists)
    neighbors = [i.item() for i in sorted_idx if i != anchor_idx][:k]

    return neighbors, dists[neighbors]



def plot_neighbor_time_series(
    code_segments,
    valid_indices,
    anchor_emb_idx,
    neighbor_emb_indices,
    signal_key="signal",
    max_len=None,
):
    """
    anchor_emb_idx: index in embedding space
    neighbor_emb_indices: list of indices in embedding space
    """
    all_idxs = [anchor_emb_idx] + neighbor_emb_indices
    n = len(all_idxs)

    plt.figure(figsize=(10, 2.5 * n))

    for row, emb_idx in enumerate(all_idxs):
        seg_idx = valid_indices[emb_idx]
        segment = code_segments[seg_idx]

        if signal_key not in segment:
            raise KeyError(f"segment {seg_idx} has no key '{signal_key}'")

        signal = segment[signal_key]
        signal = signal[:, 0].flatten()
        if isinstance(signal, torch.Tensor):
            signal = signal.cpu().numpy()

        if max_len is not None:
            signal = signal[:max_len]


        plt.subplot(n, 1, row + 1)
        plt.plot(signal.flatten(), lw=1.5)
        if row == 0:
            plt.title("Anchor Time Series", fontsize=12)
        else:
            plt.title(f"Neighbor {row}", fontsize=11)
        plt.grid(alpha=0.3)

    plt.tight_layout()
    # plt.show()
    plt.savefig("time_series.png")



def inspect_embedding_neighborhood(
    all_embeddings,
    code_segments,
    valid_indices,
    anchor_emb_idx: int,
    k: int = 5,
    metric: str = "l2",
    signal_key: str = "signal",
):
    neighbors, dists = find_nearest_neighbors(
        all_embeddings, anchor_emb_idx, k=k, metric=metric
    )
    print("Anchor embedding index:", anchor_emb_idx)
    print("Nearest neighbors (embedding indices):")

    for neighbor_idx, (i, d) in enumerate(zip(neighbors, dists)):
        print(f" Neighbor{neighbor_idx}: idx={i}, dist={float(d):.4f}")

    plot_neighbor_time_series(
        code_segments,
        valid_indices,
        anchor_emb_idx,
        neighbors,
        signal_key=signal_key,
    )


def get_time_series_embedding(code_ids, codebook_embedding):
    """
    code_ids: Tensor of shape (T,)
    codebook_embedding: Tensor of shape (V, D)
    """
    emb = codebook_embedding[code_ids]  # shape: (T, D)
    # return emb.mean(dim=0)  # shape: (D,)
    return emb.flatten()  # shape: (D,)





def cluster_analysis(args):
    dataset = AnomalyDataset(
        raw_data_paths=args.data_paths,
        indices_paths=args.indices_paths,
        one_channel=args.one_channel,
        max_length=args.max_seq_len,
        min_length=args.min_seq_len,
        data_type=args.data_type,
    )

    code_segments = torch.load(
        f"{args.save_dir}/code_segments.pt",
        map_location='cpu'
    )
    for i, code_segment in enumerate(code_segments):
        code_segment.update({'signal': dataset.__getitem__(i)[0]})

    model_ckpt = torch.load(
        f"{args.save_dir}/vqvae.pt",
        map_location='cpu'
    )
    state_dict = model_ckpt["model_state"]
    # 自动查找 codebook 的 key
    for k in state_dict:
        if 'quantizer' in k and 'weight' in k:
            print(f"Found codebook: {k}")
            codebook_embedding = state_dict[k]
            print("Shape:", codebook_embedding.shape)
            break

    # Step 1: Compute embedding for each time series
    all_embeddings = []
    valid_indices = []  # 保存合法的索引，防止某些项有问题

    for i, segment in enumerate(code_segments):
        if 'ids' not in segment:
            continue
        code_ids = segment['ids']  # shape (T,)
        try:
            emb = get_time_series_embedding(code_ids, codebook_embedding)
            all_embeddings.append(emb)
            valid_indices.append(i)
        except Exception as e:
            print(f"Skipping segment {i} due to error: {e}")

    all_embeddings = torch.stack(all_embeddings)  # (N, D)
    # all_embeddings_np = all_embeddings.cpu().numpy()

    inspect_embedding_neighborhood(
        all_embeddings=all_embeddings,
        code_segments=code_segments,
        valid_indices=valid_indices,
        anchor_emb_idx=10,  # 任选一个
        k=20,
        metric="l2",
        signal_key="signal",
    )





# --------------------------
# Example usage
# --------------------------
def get_args():
    parser = argparse.ArgumentParser(description="parameters for vqvae pretraining")

    parser.add_argument("--min_seq_len", type=int, required=True)
    parser.add_argument("--max_seq_len", type=int, required=True)

    parser.add_argument("--data_paths", type=json.loads, required=True)
    parser.add_argument("--indices_paths", type=json.loads, required=True)
    # parser.add_argument("--anomaly_data_paths", type=json.loads, required=True)
    # parser.add_argument("--anomaly_indices_paths", type=json.loads, required=True)

    parser.add_argument("--data_type", type=str, required=True)
    parser.add_argument("--gpu_id", type=int, required=True)
    parser.add_argument("--save_dir", type=str, required=True)

    parser.add_argument("--code_dim", type=int, required=True)
    parser.add_argument("--code_len", type=int, required=True)
    parser.add_argument("--num_codes", type=int, required=True)

    parser.add_argument("--one_channel", type=int, required=True)
    parser.add_argument("--feat_size", type=int, required=True)

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    # cfg = TrainConfig(
    #     encoder_channels=(16,16,32,32,64,64),
    #     decoder_channels=(64,64,32,32,16,16),
    #     down_ratio=2,
    #     up_ratio=2,
    #     max_length=800,
    #
    #     raw_data_path="../dataset_utils/ECG_datasets/raw_data/106.npz",
    #     indices_path="../dataset_utils/ECG_datasets/indices/slide_windows_106npz/train/mixed.jsonl",
    #     one_channel=True,
    #
    #     batch_size=64,
    #     epochs=50,
    #     lr=1e-3,
    #
    #     hidden=64,
    #     code_dim=8,
    #     code_len=4,
    #     num_codes=500,
    #     beta=0.25,
    #
    #     recon_loss="mse",
    #     vq_loss_weight=1.0,
    #     device="cuda:7",
    #     # device="cpu",
    #     save_path="/root/tianyi/vqvae_save_path/vqvae_1d.pt",
    # )



    cfg = TrainConfig(
        encoder_channels=(64, 64, 64),
        decoder_channels=(64, 64, 32, 32, 16, 16),
        down_ratio=1,
        up_ratio=2,
        min_length=args.min_seq_len,
        max_length=args.max_seq_len,

        data_paths=args.data_paths,
        indices_paths=args.indices_paths,

        # normal_indices_paths=args.normal_indices_paths,
        # normal_indices_paths=args.normal_indices_paths,
        # anomaly_data_paths=args.anomaly_data_paths,
        # anomaly_indices_paths=args.anomaly_indices_paths,

        one_channel=args.one_channel,
        feat_size=args.feat_size,
        data_type=args.data_type,

        batch_size=64,
        epochs=50,
        lr=1e-3,

        hidden=64,
        code_dim=args.code_dim,
        code_len=args.code_len,
        num_codes=args.num_codes,
        beta=0.25,

        recon_loss="mse",
        vq_loss_weight=1.0,
        device=f"cuda:{args.gpu_id}",
        # device="cpu",
        save_path=args.save_dir,

    )
    model = train_vqvae(cfg)
    extract_code_segments(
        in_channels=args.feat_size,
        code_dim=args.code_dim,
        num_codes=args.num_codes,
        model_path=f"{args.save_dir}/vqvae.pt",
        raw_data_paths=args.data_paths,
        indices_paths=args.indices_paths,
        data_type=args.data_type,
        one_channel=args.one_channel,
        # device="cuda:7",
        device="cpu",
        save_path=f"{args.save_dir}/code_segments.pt",
        down_ratio=2,
        up_ratio=2,
        max_length=args.max_seq_len,
        min_length=args.min_seq_len,
        encoder_channels=(16, 16, 32, 32, 64, 64),
        decoder_channels=(64, 64, 32, 32, 16, 16),
        code_len=args.code_len,
    )

    # cluster_analysis(args)

    # extract_code_segments(
    #     in_channels=1,
    #     code_dim=8,
    #     num_codes=500,
    #     model_path="/root/tianyi/vqvae_save_path/vqvae_1d.pt",
    #     raw_data_path="../dataset_utils/ECG_datasets/raw_data/106.npz",
    #     indices_path="../dataset_utils/ECG_datasets/indices/slide_windows_106npz/train/mixed.jsonl",
    #     one_channel=True,
    #     device="cuda:7",
    #     save_path="/root/tianyi/vqvae_save_path/code_segments.pt",
    #     down_ratio=2,
    #     up_ratio=2,
    #     max_length=100,
    #     encoder_channels=(16,16,32,32,64,64),
    #     decoder_channels=(64,64,32,32,16,16),
    #     code_len=4,
    #     seq_len=800
    # )

    # 2️⃣ 离线画图
    # plot_code_waveforms(
    #     "code_segments.pt",
    #     code_ids=[0, 5, 12, 42],
    #     num_samples=10,
    # )