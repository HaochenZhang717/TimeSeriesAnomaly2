import torch
from torch import nn
import math
import torch.nn.functional as F
from einops import reduce


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

        self.z_mean = nn.Linear(code_dim, code_dim)
        self.z_log_var = nn.Linear(code_dim, code_dim)

    def forward(self, x):
        # x: [B, T, C] → [B, C, T]
        x = x.transpose(1, 2)
        h = self.stem(x)
        h = self.stages(h)
        z = self.global_pooling(h)
        z = torch.relu(z)
        z = self.proj(z)          # [B, D, T']
        z = z.transpose(1, 2)     # [B, T', D]
        z_mean = self.z_mean(z)
        z_log_var = self.z_log_var(z)
        std = torch.exp(0.5 * z_log_var)
        dist = torch.distributions.Normal(z_mean, std)
        z = dist.rsample()
        return z_mean, z_log_var, z


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


class CNNVAE(nn.Module):
    def __init__(
        self,
        in_channels, encoder_channels,
        decoder_channels, code_dim,
        down_ratio, up_ratio,
        code_len, seq_len,
        kl_wt
    ):
        super().__init__()
        self.kl_wt = kl_wt

        self.encoder = ResNetEncoder1D(
            seq_len=seq_len,
            in_channels=in_channels,
            channels=encoder_channels,
            blocks_per_stage=1,
            code_dim=code_dim,
            down_ratio=down_ratio,
            code_len=code_len
        )
        self.decoder = ResNetDecoder1D(
            out_channels=in_channels,
            channels=decoder_channels,
            blocks_per_stage=1,
            code_dim=code_dim,
            up_ratio=up_ratio,
            code_len=code_len,
            seq_len=seq_len
        )

    def forward(self, x):
        z_mean, z_log_var, z = self.encoder(x)
        x_hat = self.decoder(z, x.size(1))
        print(z.shape) # 4, 8, 8
        print(x_hat.shape)
        return x_hat

    def _get_reconstruction_loss(self, target, model_out, noise_mask):
        target = target * noise_mask.unsqueeze(-1)
        model_out = model_out * noise_mask.unsqueeze(-1)

        loss = (model_out - target) ** 2
        loss = loss.mean(-1)

        # 只对 anomaly 部分计算误差
        masked_loss = loss * noise_mask  # (B, T)
        # 每个样本 anomaly 的数量
        num_anomalies = reduce(noise_mask, 'b t -> b 1', 'sum')  # shape: (B, 1)

        # 每个样本的 loss = sum(masked_loss) / num_anomalies
        loss_per_sample = reduce(masked_loss, 'b t -> b 1', 'sum') / num_anomalies

        # 最终 batch loss = mean over batch
        return loss_per_sample.mean()


    def loss_function(self, X, X_recons, noise_mask, z_mean, z_log_var):
        reconstruction_loss = self._get_reconstruction_loss(X, X_recons, noise_mask)
        # kl_loss = -0.5 * torch.sum(1 + z_log_var - z_mean.pow(2) - z_log_var.exp())
        # kl_loss = -0.5 * torch.mean(1 + z_log_var - z_mean.pow(2) - z_log_var.exp())

        kl = -0.5 * torch.sum(1 + z_log_var - z_mean.pow(2) - z_log_var.exp(), dim=1)
        kl_loss = torch.mean(kl)

        total_loss = reconstruction_loss + self.kl_wt * kl_loss
        # total_loss = self.reconstruction_wt * reconstruction_loss + kl_loss
        return total_loss, reconstruction_loss, kl_loss


if __name__ == "__main__":
    model = CNNVAE(
        in_channels=2,
        encoder_channels=[64, 64, 64, 64],
        decoder_channels=[64, 64, 32, 32, 16, 16],
        code_dim=16,
        down_ratio=1,
        up_ratio=2,
        code_len=8,
        seq_len=800
    )

    input_ts = torch.randn(4, 800, 2)
    out = model(input_ts)
