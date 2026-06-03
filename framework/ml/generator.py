"""Variational Autoencoder for synthetic transcriptomic data generation."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Dict


class VAE(nn.Module):
    """Variational Autoencoder for transcriptomic data."""
    
    def __init__(self, input_dim: int, latent_dim: int = 64, hidden_dims: list = [256, 128]):
        super().__init__()
        
        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            encoder_layers.extend([nn.Linear(prev_dim, h_dim), nn.BatchNorm1d(h_dim), nn.ReLU()])
            prev_dim = h_dim
        self.encoder = nn.Sequential(*encoder_layers)
        self.fc_mu = nn.Linear(prev_dim, latent_dim)
        self.fc_logvar = nn.Linear(prev_dim, latent_dim)
        
        # Decoder
        decoder_layers = []
        prev_dim = latent_dim
        for h_dim in reversed(hidden_dims):
            decoder_layers.extend([nn.Linear(prev_dim, h_dim), nn.BatchNorm1d(h_dim), nn.ReLU()])
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
        return self.decode(z), mu, logvar


class VAEGenerator:
    """
    VAE-based generator for synthetic transcriptomic profiles.
    """
    
    def __init__(self, input_dim: int, latent_dim: int = 64, hidden_dims: list = [256, 128],
                 learning_rate: float = 0.001, beta: float = 0.5):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.vae = VAE(input_dim, latent_dim, hidden_dims).to(self.device)
        self.optimizer = optim.AdamW(self.vae.parameters(), lr=learning_rate, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', patience=10, factor=0.5)
        self.beta = beta
        self.scaler = StandardScaler()
        self.latent_dim = latent_dim
    
    def vae_loss(self, recon, x, mu, logvar):
        """Compute VAE loss (reconstruction + KL divergence)."""
        recon_loss = nn.MSELoss(reduction='sum')(recon, x)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return (recon_loss + self.beta * kl_loss) / x.size(0)
    
    def fit(self, data: np.ndarray, epochs: int = 50, batch_size: int = 128, verbose: bool = True) -> Dict:
        """Train the VAE on transcriptomic data."""
        # Standardize data
        data_scaled = self.scaler.fit_transform(data)
        
        # Create DataLoader
        dataset = TensorDataset(torch.FloatTensor(data_scaled))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        history = {'train_loss': []}
        
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
            history['train_loss'].append(avg_loss)
            self.scheduler.step(avg_loss)
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        
        return history
    
    def generate(self, n_samples: int = 1000) -> np.ndarray:
        """Generate synthetic expression profiles."""
        self.vae.eval()
        with torch.no_grad():
            z = torch.randn(n_samples, self.latent_dim).to(self.device)
            synthetic_scaled = self.vae.decode(z).cpu().numpy()
            return self.scaler.inverse_transform(synthetic_scaled)
    
    def save(self, path: str):
        """Save trained model."""
        torch.save({
            'model_state_dict': self.vae.state_dict(),
            'scaler_mean': self.scaler.mean_,
            'scaler_scale': self.scaler.scale_,
            'latent_dim': self.latent_dim
        }, path)
    
    def load(self, path: str):
        """Load trained model."""
        checkpoint = torch.load(path)
        self.vae.load_state_dict(checkpoint['model_state_dict'])
        self.scaler.mean_ = checkpoint['scaler_mean']
        self.scaler.scale_ = checkpoint['scaler_scale']
        self.latent_dim = checkpoint['latent_dim']
