import torch
import torch.nn as nn
import torch.nn.functional as F
from encoding import HashEncodingPyTorch, HashEncodingWarp, ZipHashEncodingPyTorch
try:
    from encoding import ZipHashEncodingWarp
except ImportError:
    ZipHashEncodingWarp = None
from sh import SphericalHarmonics, SphericalHarmonicsWarp
from config import Config

class InstantNGP(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.USE_WARP:
            print("[InstantNGP] Using Warp-Accelerated Backend (Hash Grid, SH, Ray Box).")
            self.encoder = HashEncodingWarp(config)
            self.sh_encoder = SphericalHarmonicsWarp(degree=4)
        else:
            print("[InstantNGP] Using Pure PyTorch Backend.")
            self.encoder = HashEncodingPyTorch(config)
            self.sh_encoder = SphericalHarmonics(degree=4) 
            
        input_dim = config.L * config.F 
        self.density_mlp = nn.Sequential(
            nn.Linear(input_dim, config.HIDDEN_DIM_DENSITY),
            nn.ReLU(),
            nn.Linear(config.HIDDEN_DIM_DENSITY, 16)
        )
        
        self.color_mlp = nn.Sequential(
            nn.Linear(16 + 16, config.HIDDEN_DIM_COLOR),
            nn.ReLU(),
            nn.Linear(config.HIDDEN_DIM_COLOR, config.HIDDEN_DIM_COLOR), 
            nn.ReLU(),
            nn.Linear(config.HIDDEN_DIM_COLOR, 3), 
            nn.Sigmoid()
        )
        self.apply(self.init_weights)
        nn.init.constant_(self.density_mlp[2].bias[0], -5.0)
        
    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x, d):
        encoded_x = self.encoder(x)
        density_out = self.density_mlp(encoded_x)

        sigma = torch.exp(torch.clamp(density_out[:, 0], max=15.0))
        geo_features = density_out

        encoded_d = self.sh_encoder(d)
        color_in = torch.cat([geo_features, encoded_d], dim=-1)
        rgb = self.color_mlp(color_in)

        return rgb, sigma


class ZipInstantNGP(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.USE_WARP and ZipHashEncodingWarp is not None:
            print("[ZipInstantNGP] Using Warp backend with Zip multisampling.")
            self.encoder = ZipHashEncodingWarp(config)
            self.sh_encoder = SphericalHarmonicsWarp(degree=4)
        else:
            print("[ZipInstantNGP] Using PyTorch backend with Zip multisampling.")
            self.encoder = ZipHashEncodingPyTorch(config)
            self.sh_encoder = SphericalHarmonics(degree=4)

        self.L = config.L
        # Density MLP input: L*F spatial features + L scale features (Appendix C)
        input_dim = config.L * config.F + config.L
        self.density_mlp = nn.Sequential(
            nn.Linear(input_dim, config.HIDDEN_DIM_DENSITY),
            nn.ReLU(),
            nn.Linear(config.HIDDEN_DIM_DENSITY, 16)
        )
        self.color_mlp = nn.Sequential(
            nn.Linear(16 + 16, config.HIDDEN_DIM_COLOR),
            nn.ReLU(),
            nn.Linear(config.HIDDEN_DIM_COLOR, config.HIDDEN_DIM_COLOR),
            nn.ReLU(),
            nn.Linear(config.HIDDEN_DIM_COLOR, 3),
            nn.Sigmoid()
        )
        self.apply(self.init_weights)
        nn.init.constant_(self.density_mlp[2].bias[0], -5.0)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x_multi, sigma_j, d):
        encoded_x = self.encoder(x_multi, sigma_j)  # (N, L*F + L)
        density_out = self.density_mlp(encoded_x)
        sigma = torch.exp(torch.clamp(density_out[:, 0], max=15.0))
        geo_features = density_out
        encoded_d = self.sh_encoder(d)
        color_in = torch.cat([geo_features, encoded_d], dim=-1)
        rgb = self.color_mlp(color_in)
        return rgb, sigma

    def forward_single(self, x, d):
        encoded_x = self.encoder.forward_single(x)  # (N, L*F)
        scale_pad = torch.zeros(encoded_x.shape[0], self.L, device=encoded_x.device)
        encoded_x = torch.cat([encoded_x, scale_pad], dim=-1)  # (N, L*F + L)
        density_out = self.density_mlp(encoded_x)
        sigma = torch.exp(torch.clamp(density_out[:, 0], max=15.0))
        geo_features = density_out
        encoded_d = self.sh_encoder(d)
        color_in = torch.cat([geo_features, encoded_d], dim=-1)
        rgb = self.color_mlp(color_in)
        return rgb, sigma
