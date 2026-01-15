import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import joblib

# from .vae_base import BaseVariationalAutoencoder, Sampling
from einops import reduce


class TrendLayer(nn.Module):
    def __init__(self, seq_len, feat_dim, latent_dim, trend_poly):
        super(TrendLayer, self).__init__()
        self.seq_len = seq_len
        self.feat_dim = feat_dim
        self.latent_dim = latent_dim
        self.trend_poly = trend_poly
        self.trend_dense1 = nn.Linear(self.latent_dim, self.feat_dim * self.trend_poly)
        self.trend_dense2 = nn.Linear(self.feat_dim * self.trend_poly, self.feat_dim * self.trend_poly)

    def forward(self, z):
        trend_params = F.relu(self.trend_dense1(z))
        trend_params = self.trend_dense2(trend_params)
        trend_params = trend_params.view(-1, self.feat_dim, self.trend_poly)

        lin_space = torch.arange(0, float(self.seq_len), 1, device=z.device) / self.seq_len 
        poly_space = torch.stack([lin_space ** float(p + 1) for p in range(self.trend_poly)], dim=0) 

        trend_vals = torch.matmul(trend_params, poly_space) 
        trend_vals = trend_vals.permute(0, 2, 1) 
        return trend_vals


class SeasonalLayer(nn.Module):
    def __init__(self, seq_len, feat_dim, latent_dim, custom_seas):
        super(SeasonalLayer, self).__init__()
        self.seq_len = seq_len
        self.feat_dim = feat_dim
        self.custom_seas = custom_seas

        self.dense_layers = nn.ModuleList([
            nn.Linear(latent_dim, feat_dim * num_seasons)
            for num_seasons, len_per_season in custom_seas
        ])
        

    def _get_season_indexes_over_seq(self, num_seasons, len_per_season):
        season_indexes = torch.arange(num_seasons).unsqueeze(1) + torch.zeros(
            (num_seasons, len_per_season), dtype=torch.int32
        )
        season_indexes = season_indexes.view(-1)
        season_indexes = season_indexes.repeat(self.seq_len // len_per_season + 1)[: self.seq_len]
        return season_indexes

    def forward(self, z):
        N = z.shape[0]
        ones_tensor = torch.ones((N, self.feat_dim, self.seq_len), dtype=torch.int32, device=z.device)

        all_seas_vals = []
        for i, (num_seasons, len_per_season) in enumerate(self.custom_seas):
            season_params = self.dense_layers[i](z)
            season_params = season_params.view(-1, self.feat_dim, num_seasons)

            season_indexes_over_time = self._get_season_indexes_over_seq(
                num_seasons, len_per_season
            ).to(z.device)

            dim2_idxes = ones_tensor * season_indexes_over_time.view(1, 1, -1)
            season_vals = torch.gather(season_params, 2, dim2_idxes)

            all_seas_vals.append(season_vals)

        all_seas_vals = torch.stack(all_seas_vals, dim=-1) 
        all_seas_vals = torch.sum(all_seas_vals, dim=-1)  
        all_seas_vals = all_seas_vals.permute(0, 2, 1)  

        return all_seas_vals

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.seq_len, self.feat_dim)
    

class LevelModel(nn.Module):
    def __init__(self, latent_dim, feat_dim, seq_len):
        super(LevelModel, self).__init__()
        self.latent_dim = latent_dim
        self.feat_dim = feat_dim
        self.seq_len = seq_len
        self.level_dense1 = nn.Linear(self.latent_dim, self.feat_dim)
        self.level_dense2 = nn.Linear(self.feat_dim, self.feat_dim)
        self.relu = nn.ReLU()

    def forward(self, z):
        level_params = self.relu(self.level_dense1(z))
        level_params = self.level_dense2(level_params)
        level_params = level_params.view(-1, 1, self.feat_dim)

        ones_tensor = torch.ones((1, self.seq_len, 1), dtype=torch.float32, device=z.device)
        level_vals = level_params * ones_tensor
        return level_vals


class ResidualConnection(nn.Module):
    def __init__(self, seq_len, feat_dim, hidden_layer_sizes,
                 latent_dim, encoder_last_dense_dim):
        super(ResidualConnection, self).__init__()
        self.seq_len = seq_len
        self.feat_dim = feat_dim
        self.hidden_layer_sizes = hidden_layer_sizes
        
        self.dense = nn.Linear(latent_dim, encoder_last_dense_dim)
        self.deconv_layers = nn.ModuleList()
        in_channels = hidden_layer_sizes[-1]
        
        for i, num_filters in enumerate(reversed(hidden_layer_sizes[:-1])):
            self.deconv_layers.append(
                nn.ConvTranspose1d(in_channels, num_filters, kernel_size=3, stride=2, padding=1, output_padding=1)
            )
            in_channels = num_filters
            
        self.deconv_layers.append(
            nn.ConvTranspose1d(in_channels, feat_dim, kernel_size=3, stride=2, padding=1, output_padding=1)
        )

        L_in = encoder_last_dense_dim // hidden_layer_sizes[-1] 
        for i in range(len(hidden_layer_sizes)):
            L_in = (L_in - 1) * 2 - 2 * 1 + 3 + 1 
        L_final = L_in 

        self.final_dense = nn.Linear(feat_dim * L_final, seq_len * feat_dim)

    def forward(self, z):
        batch_size = z.size(0)
        x = F.relu(self.dense(z))
        x = x.view(batch_size, -1, self.hidden_layer_sizes[-1])
        x = x.transpose(1, 2)
        
        for deconv in self.deconv_layers[:-1]:
            x = F.relu(deconv(x))
        x = F.relu(self.deconv_layers[-1](x))
        
        x = x.flatten(1)
        x = self.final_dense(x)
        residuals = x.view(-1, self.seq_len, self.feat_dim)
        return residuals
    

class TimeVAEEncoder(nn.Module):
    def __init__(self, seq_len, feat_dim, hidden_layer_sizes, latent_dim):
        super(TimeVAEEncoder, self).__init__()
        self.seq_len = seq_len
        self.feat_dim = feat_dim
        self.latent_dim = latent_dim
        self.hidden_layer_sizes = hidden_layer_sizes
        self.layers = []
        self.layers.append(nn.Conv1d(feat_dim, hidden_layer_sizes[0], kernel_size=3, stride=2, padding=1))
        self.layers.append(nn.ReLU())

        for i, num_filters in enumerate(hidden_layer_sizes[1:]):
            self.layers.append(nn.Conv1d(hidden_layer_sizes[i], num_filters, kernel_size=3, stride=2, padding=1))
            self.layers.append(nn.ReLU())

        self.layers.append(nn.Flatten())
        
        self.encoder_last_dense_dim = self._get_last_dense_dim(seq_len, feat_dim, hidden_layer_sizes)

        self.encoder = nn.Sequential(*self.layers)
        self.z_mean = nn.Linear(self.encoder_last_dense_dim, latent_dim)
        self.psi = nn.Linear(self.encoder_last_dense_dim, 1)
        self.z_log_var = nn.Linear(self.encoder_last_dense_dim, latent_dim)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.encoder(x)
        z_mean = self.z_mean(x)
        z_log_var = self.z_log_var(x)
        psi = torch.exp(self.psi(x))

        normal_std = torch.exp(0.5 * z_log_var)
        abnormal_std = psi * normal_std

        normal_dist = torch.distributions.Normal(z_mean, normal_std)
        abnormal_dist = torch.distributions.Normal(z_mean, abnormal_std)

        return normal_dist, abnormal_dist


    def _get_last_dense_dim(self, seq_len, feat_dim, hidden_layer_sizes):
        with torch.no_grad():
            x = torch.randn(1, feat_dim, seq_len)
            for conv in self.layers:
                x = conv(x)
            return x.numel()


class TimeVAEDecoder(nn.Module):
    def __init__(self,
                 seq_len, feat_dim, hidden_layer_sizes,
                 latent_dim, trend_poly=0, custom_seas=None,
                 use_residual_conn=True, encoder_last_dense_dim=None):
        super(TimeVAEDecoder, self).__init__()
        self.seq_len = seq_len
        self.feat_dim = feat_dim
        self.hidden_layer_sizes = hidden_layer_sizes
        self.latent_dim = latent_dim
        self.trend_poly = trend_poly
        self.custom_seas = custom_seas
        self.use_residual_conn = use_residual_conn
        self.encoder_last_dense_dim = encoder_last_dense_dim
        self.level_model = LevelModel(self.latent_dim, self.feat_dim, self.seq_len)

        self.trend_layer = None
        self.season_layer = None
        self.residual_conn = None

        if use_residual_conn:
            self.residual_conn = ResidualConnection(seq_len, feat_dim, hidden_layer_sizes, latent_dim, encoder_last_dense_dim)

        if self.trend_poly is not None and self.trend_poly > 0:
            self.trend_layer = TrendLayer(self.seq_len, self.feat_dim, self.latent_dim, self.trend_poly)

        if self.custom_seas is not None and len(self.custom_seas) > 0:
            self.season_layer = SeasonalLayer(self.seq_len, self.feat_dim, self.latent_dim, self.custom_seas)


    def forward(self, z):
        outputs = self.level_model(z)
        if self.trend_layer is not None:
            trend_vals = self.trend_layer(z)
            outputs += trend_vals

        # custom seasons
        if self.season_layer is not None:
            cust_seas_vals = self.season_layer(z)
            outputs += cust_seas_vals

        if self.residual_conn is not None:
            residuals = self.residual_conn(z)
            outputs += residuals

        return outputs


class GenIASModel(nn.Module):
    def __init__(
        self,
        hidden_layer_sizes=None,
        trend_poly=0,
        custom_seas=None,
        use_residual_conn=True,
        delta_min=0.1,
        delta_max=0.1,
        reconstruction_weight=1.0,
        perturbation_weight=1.0,
        kl_weight=1.0,
        seq_len=100,
        feat_dim=1,
        latent_dim=32,
        sigma_prior=1.0
    ):
        super(GenIASModel, self).__init__()

        if hidden_layer_sizes is None:
            hidden_layer_sizes = [50, 100, 200]

        self.seq_len = seq_len
        self.feat_dim = feat_dim
        self.latent_dim = latent_dim
        self.sigma_prior = torch.tensor(sigma_prior)

        self.delta_min = delta_min
        self.delta_max = delta_max
        self.reconstruction_weight = reconstruction_weight
        self.perturbation_weight = perturbation_weight
        self.kl_weight = kl_weight


        self.hidden_layer_sizes = hidden_layer_sizes
        self.trend_poly = trend_poly
        self.custom_seas = custom_seas
        self.use_residual_conn = use_residual_conn

        self.encoder = self._get_encoder()
        self.decoder = self._get_decoder()

    def _get_encoder(self):
        return TimeVAEEncoder(self.seq_len, self.feat_dim, self.hidden_layer_sizes, self.latent_dim)

    def _get_decoder(self):
        return TimeVAEDecoder(
            self.seq_len, self.feat_dim, self.hidden_layer_sizes,
            self.latent_dim, self.trend_poly, self.custom_seas,
            self.use_residual_conn, self.encoder.encoder_last_dense_dim)

    def forward(self, x_occluded, x, noise_mask):
        normal_dist, abnormal_dist= self.encoder(x_occluded)
        normal_z = normal_dist.rsample()
        abnormal_z = abnormal_dist.rsample()

        normal_pred = self.decoder(normal_z)
        abnormal_pred = self.decoder(abnormal_z)

        reconstruction_loss = self.reconstruction_loss(x, normal_pred, noise_mask)
        perturbation_loss = self.perturbation_loss(x, normal_pred, abnormal_pred, noise_mask)
        enhanced_kl_loss = self.enhanced_kl_loss(normal_dist)

        total_loss = (self.reconstruction_weight * reconstruction_loss +
                      self.perturbation_weight * perturbation_loss +
                      self.kl_weight * enhanced_kl_loss)

        return total_loss, reconstruction_loss, perturbation_loss, enhanced_kl_loss

    def reconstruction_loss(self, x, normal_pred, noise_mask):
        normal_pred = normal_pred * noise_mask.unsqueeze(-1)
        x = x * noise_mask.unsqueeze(-1)
        loss = (x - normal_pred) ** 2
        loss = loss.mean(-1)

        # 只对 noise 部分计算误差
        masked_loss = loss * noise_mask  # (B, T)
        num_anomalies = reduce(noise_mask, 'b t -> b 1', 'sum')  # shape: (B, 1)
        loss_per_sample = reduce(masked_loss, 'b t -> b 1', 'sum') / num_anomalies
        return loss_per_sample.mean()


    def perturbation_loss(self, x, normal_pred, abnormal_pred, noise_mask):
        first_term  = torch.relu(self.reconstruction_loss(x, normal_pred, noise_mask) - self.reconstruction_loss(x, abnormal_pred, noise_mask) + self.delta_min)
        second_term = torch.relu(self.reconstruction_loss(x, abnormal_pred, noise_mask) - self.delta_max)
        # first_term = torch.relu(nn.functional.mse_loss(x, normal_pred) - nn.functional.mse_loss(x, abnormal_pred) + self.delta_min)
        # second_term = torch.relu(nn.functional.mse_loss(x, abnormal_pred) - self.delta_max)
        return first_term + second_term

    def zero_perturbation_loss(self, x, normal_pred, abnormal_pred):
        raise NotImplementedError

    def enhanced_kl_loss(self, normal_dist):
        kl_loss = -0.5 * torch.mean(1 + 2*torch.log(normal_dist.scale) - normal_dist.loc ** 2 - normal_dist.scale**2 / self.sigma_prior**2 + 2*torch.log(self.sigma_prior))
        return kl_loss

    def conditional_generate(self, x):
        normal_dist, abnormal_dist = self.encoder(x)
        normal_z = normal_dist.rsample()
        abnormal_z = abnormal_dist.rsample()

        normal_pred = self.decoder(normal_z)
        abnormal_pred = self.decoder(abnormal_z)

        return normal_pred, abnormal_pred

    def get_anomaly_samples(self, x_occluded, noise_mask):
        _, abnormal_dist = self.encoder(x_occluded)
        abnormal_z = abnormal_dist.rsample()
        sample = self.decoder(abnormal_z)
        output = sample * noise_mask.unsqueeze(-1) + x_occluded * (1 - noise_mask.unsqueeze(-1))
        return output