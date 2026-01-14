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
        batch_size=16,
        **kwargs
    ):
        super(BaseVariationalAutoencoder, self).__init__()
        self.seq_len = seq_len
        self.feat_dim = feat_dim
        self.latent_dim = latent_dim
        self.kl_wt = kl_wt
        self.batch_size = batch_size
        self.encoder = None
        self.decoder = None
        self.sampling = Sampling()


    def forward(self, X):
        z_mean, z_log_var, z = self.encoder(X)
        x_decoded = self.decoder(z_mean)
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


    def get_num_trainable_variables(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_prior_samples(self, num_samples):
        device = next(self.parameters()).device
        Z = torch.randn(num_samples, self.latent_dim).to(device)
        samples = self.decoder(Z)
        return samples.cpu().detach().numpy()

    def get_prior_anomaly_samples(self, num_samples):
        device = next(self.parameters()).device
        Z = torch.randn(num_samples, self.latent_dim).to(device)
        window_labels = torch.zeros(num_samples, 64, 1).to(device)

        for i in range(num_samples):
            start = np.random.randint(low=0, high=64-5)
            end = start + np.random.randint(low=0, high=5)
            window_labels[i, start:end] = 1

        samples = self.anomaly_decoder(Z, window_labels)
        return samples.cpu().detach().numpy(), window_labels

    def get_prior_samples_given_Z(self, Z):
        Z = torch.FloatTensor(Z).to(next(self.parameters()).device)
        samples = self.decoder(Z)
        return samples.cpu().detach().numpy()

    @abstractmethod
    def _get_encoder(self, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def _get_decoder(self, **kwargs):
        raise NotImplementedError



if __name__ == "__main__":
    pass