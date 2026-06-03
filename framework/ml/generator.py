"""VAE for synthetic data generation."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Tuple


class VAE(nn.Module):
    """Variational Autoencoder for expression data."""
    
    def __init__(self, input_dim: int, latent_dim: int = 64,
                 hidden_dims: list = None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 128]
            
        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            encoder_layers.append(nn.Linear(prev_dim, h_dim))
            encoder_layers.append(nn.BatchNorm1d(h_dim))
            encoder_layers.append(nn.ReLU())
            prev_dim = h_dim
        self.encoder = nn.Sequential(*encoder_layers)
        self.fc_mu = nn.Linear(prev_dim, latent_dim)
        self.fc_logvar = nn.Linear(prev_dim, latent_dim)
        
        # Decoder
        decoder_layers = []
        prev_dim = latent_dim
        for h_dim in reversed(hidden_dims):
            decoder_layers.append(nn.Linear(prev_dim, h_dim))
            decoder_layers.append(nn.BatchNorm1d(h_dim))
            decoder_layers.append(nn.ReLU())
            prev_dim = h_dim
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)
        
    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar


class VAEGenerator:
    """VAE training and generation wrapper."""
    
    def __init__(self, input_dim: int, latent_dim: int = 64,
                 hidden_dims: list = None, beta: float = 0.5,
                 learning_rate: float = 1e-3, device: str = None):
        """
        Initialize VAE generator.
        
        Parameters:
        -----------
        input_dim : int
            Input dimension
        latent_dim : int
            Latent space dimension
        hidden_dims : list
            Hidden layer dimensions
        beta : float
            Beta VAE weight
        learning_rate : float
            Learning rate
        device : str
            Device to use ('cuda' or 'cpu')
        """
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)
        
        self.vae = VAE(input_dim, latent_dim, hidden_dims).to(self.device)
        self.optimizer = optim.AdamW(self.vae.parameters(), lr=learning_rate, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', patience=10, factor=0.5)
        self.beta = beta
        
    def vae_loss(self, recon, x, mu, logvar):
        """Compute VAE loss."""
        recon_loss = nn.MSELoss(reduction='sum')(recon, x)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return (recon_loss + self.beta * kl_loss) / x.size(0)
    
    def fit(self, X: np.ndarray, batch_size: int = 128, epochs: int = 50,
            verbose: bool = True) -> list:
        """Train VAE on data."""
        X_tensor = torch.FloatTensor(X).to(self.device)
        dataset = TensorDataset(X_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        losses = []
        for epoch in range(epochs):
            self.vae.train()
            total_loss = 0
            for batch in loader:
                x = batch[0].to(self.device)
                recon, mu, logvar = self.vae(x)
                loss = self.vae_loss(recon, x, mu, logvar)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            
            avg_loss = total_loss / len(loader)
            losses.append(avg_loss)
            self.scheduler.step(avg_loss)
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        
        return losses
    
    def generate(self, n_samples: int) -> np.ndarray:
        """Generate synthetic samples."""
        self.vae.eval()
        with torch.no_grad():
            z = torch.randn(n_samples, self.vae.fc_mu.out_features).to(self.device)
            synthetic = self.vae.decode(z).cpu().numpy()
        return synthetic
