"""Encoder/decoder network used by the Disentangled Learning (DL) baseline to
separate main-task from sensitive-task representations."""

import torch
import torch.nn as nn


class MainEncoder(nn.Module):
    def __init__(self, input_dim=512, latent_dim=496):
        super(MainEncoder, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 512), nn.ReLU(),
            nn.Linear(512, 512), nn.ReLU(),
            nn.Linear(512, latent_dim)
        )

    def forward(self, x):
        return self.fc(x)


# Encoder for Soft-Biometric Attributes (e.g., gender)
class SensitiveEncoder(nn.Module):
    def __init__(self, input_dim=512, latent_dim=16):
        super(SensitiveEncoder, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, latent_dim)
        )

    def forward(self, x):
        return self.fc(x)


# Decoder
class Decoder(nn.Module):
    def __init__(self, latent_dim=512, output_dim=512):
        super(Decoder, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 512), nn.ReLU(),
            nn.Linear(512, 512), nn.ReLU(),
            nn.Linear(512, output_dim)
        )

    def forward(self, z):
        return self.fc(z)


# Complete Autoencoder Model
class PFRNet(nn.Module):
    def __init__(self, input_dim=512, id_latent_dim=496, attr_latent_dim=16):
        super(PFRNet, self).__init__()
        self.encoder_id = MainEncoder(input_dim, id_latent_dim)
        self.encoder_attr = SensitiveEncoder(input_dim, attr_latent_dim)
        self.decoder = Decoder(id_latent_dim + attr_latent_dim, input_dim)

    def forward(self, x):
        z_main = self.encoder_id(x)
        z_priv = self.encoder_attr(x)
        z_combined = torch.cat((z_main, z_priv), dim=1)
        x_reconstructed = self.decoder(z_combined)
        return z_main, z_priv, x_reconstructed
