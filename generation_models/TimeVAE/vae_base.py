import os
from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import joblib
import wandb
import numpy as np
from einops import reduce

class Sampling(nn.Module):
    def forward(self, inputs):
        z_mean, z_log_var = inputs
        batch = z_mean.size(0)
        dim = z_mean.size(1)
        epsilon = torch.randn(batch, dim).to(z_mean.device)
        return z_mean + torch.exp(0.5 * z_log_var) * epsilon

class BaseVariationalAutoencoder(nn.Module, ABC):
    model_name = None

    def __init__(
        self,
        seq_len,
        feat_dim,
        latent_dim,
        kl_wt,
        **kwargs
    ):
        super(BaseVariationalAutoencoder, self).__init__()
        self.seq_len = seq_len
        self.feat_dim = feat_dim
        self.latent_dim = latent_dim
        self.kl_wt = kl_wt
        # self.batch_size = batch_size
        self.encoder = None
        self.decoder = None
        self.sampling = Sampling()

    def normal_forward(self, X_occluded):
        z_mean, z_log_var, z = self.encoder(X_occluded)
        x_decoded = self.normal_decoder(z_mean)
        return x_decoded


    def normal_predict(self, valid_data):
        self.eval()
        val_loader = DataLoader(valid_data, batch_size=16)
        batch = next(iter(val_loader))

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        X_occluded = batch[0].to(device)
        X_normal = batch[1].to(device)

        with torch.no_grad():
            z_mean, z_log_var, z = self.encoder(X_occluded)
            x_decoded = self.normal_decoder(z_mean)
        return x_decoded.cpu().detach().numpy(), X_occluded.cpu().detach().numpy(), X_normal.cpu().detach().numpy()


    # def anomaly_predict(self, valid_data):
    #     self.eval()
    #     val_loader = DataLoader(valid_data, batch_size=16)
    #     batch = next(iter(val_loader))
    #
    #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #     X_occluded = batch[0].to(device)
    #     X_anomaly = batch[1].to(device)
    #     window_label = batch[2].to(device)
    #
    #     with torch.no_grad():
    #         z_mean, z_log_var, z = self.encoder(X_occluded)
    #         x_decoded = self.anomaly_decoder(z_mean, window_label)
    #     return x_decoded.cpu().detach().numpy(), X_anomaly.cpu().detach().numpy(), window_label.cpu().detach().numpy()


    # def anomaly_inject(self, X_occluded, anomaly_label):
    #     self.eval()
    #     with torch.no_grad():
    #         z_mean, z_log_var, z = self.encoder(X_occluded)
    #         x_decoded = self.anomaly_decoder(z_mean, anomaly_label)
    #     return x_decoded

    def get_num_trainable_variables(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    # def get_prior_normal_samples(self, num_samples):
    #     device = next(self.parameters()).device
    #     Z = torch.randn(num_samples, self.latent_dim).to(device)
    #     samples = self.normal_decoder(Z)
    #     return samples

    def get_anomaly_samples(self, x_occluded, noise_mask):
        z_mean, z_log_var, z = self.encoder(x_occluded)
        sample = self.decoder(z)
        output = sample * noise_mask.unsqueeze(-1) + x_occluded * (1 - noise_mask.unsqueeze(-1))
        return output


    @abstractmethod
    def _get_encoder(self, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def _get_decoder(self, **kwargs):
        raise NotImplementedError


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
    pass