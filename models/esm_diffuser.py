import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from transformers import AutoModel, AutoTokenizer

class SinusoidalPositionEmbeddings(nn.Module):
    """Sinusoidal position embeddings for diffusion timesteps."""
    
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class ESMDiffuser(nn.Module):
    """Diffusion model for peptide sequences based on ESM embeddings."""
    
    def __init__(self, esm_dim=1280, hidden_dim=512, n_steps=1000):
        super(ESMDiffuser, self).__init__()
        self.esm_dim = esm_dim
        self.hidden_dim = hidden_dim
        self.n_steps = n_steps
        
        # Load ESM model
        self.esm_model = AutoModel.from_pretrained("facebook/esm2_t33_650M_UR50D")
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
        
        # Freeze ESM parameters
        for param in self.esm_model.parameters():
            param.requires_grad = False
        
        # Time embeddings
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
        # Noise prediction network
        self.noise_pred = nn.Sequential(
            nn.Linear(esm_dim + hidden_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, esm_dim)
        )
        
        # Beta schedule for diffusion
        self.beta = torch.linspace(0.0001, 0.02, n_steps)
        self.alpha = 1. - self.beta
        self.alpha_cumprod = torch.cumprod(self.alpha, dim=0)
    
    def get_esm_embeddings(self, sequences):
        """Get ESM embeddings for a batch of sequences."""
        # Tokenize sequences
        inputs = self.tokenizer(sequences, return_tensors="pt", padding=True).to(next(self.parameters()).device)
        
        # Get ESM embeddings
        with torch.no_grad():
            outputs = self.esm_model(**inputs)
            # Use the last hidden state, mean pooling over sequence length
            embeddings = outputs.last_hidden_state.mean(dim=1)
        
        return embeddings
    
    def forward_diffusion(self, x_0, t):
        """Forward diffusion process q(x_t | x_0)."""
        noise = torch.randn_like(x_0)
        
        # Extract alpha_cumprod for specific timesteps
        alpha_cumprod_t = self.alpha_cumprod[t].view(-1, 1)
        
        # Compute noisy sample x_t
        x_t = torch.sqrt(alpha_cumprod_t) * x_0 + torch.sqrt(1 - alpha_cumprod_t) * noise
        
        return x_t, noise
    
    def forward(self, sequences, t=None):
        """
        Forward pass of the diffusion model.
        
        Args:
            sequences: List of peptide sequences
            t: Optional timesteps for diffusion (if None, random timesteps are sampled)
            
        Returns:
            Dictionary with model outputs
        """
        device = next(self.parameters()).device
        
        # Get ESM embeddings
        x_0 = self.get_esm_embeddings(sequences)
        
        # Sample random timesteps if not provided
        if t is None:
            t = torch.randint(0, self.n_steps, (x_0.shape[0],), device=device)
        
        # Forward diffusion
        x_t, noise = self.forward_diffusion(x_0, t)
        
        # Time embeddings
        time_emb = self.time_mlp(t)
        
        # Predict noise
        noise_pred = self.noise_pred(torch.cat([x_t, time_emb], dim=1))
        
        return {
            'x_0': x_0,
            'x_t': x_t,
            'noise': noise,
            'noise_pred': noise_pred,
            'time': t
        }
    
    def sample(self, n_samples, seq_len=30, device=None):
        """Sample new sequences from the diffusion model."""
        if device is None:
            device = next(self.parameters()).device
        
        # Start from random noise
        x = torch.randn(n_samples, self.esm_dim, device=device)
        
        # Reverse diffusion process
        for t in reversed(range(self.n_steps)):
            # Create timestep batch
            timesteps = torch.full((n_samples,), t, device=device, dtype=torch.long)
            
            # Time embeddings
            time_emb = self.time_mlp(timesteps)
            
            # Predict noise
            noise_pred = self.noise_pred(torch.cat([x, time_emb], dim=1))
            
            # Extract alpha values for current timestep
            alpha = self.alpha[t]
            alpha_cumprod = self.alpha_cumprod[t]
            beta = self.beta[t]
            
            # No noise for t=0
            noise = torch.zeros_like(x) if t == 0 else torch.randn_like(x)
            
            # Update x
            x = 1 / torch.sqrt(alpha) * (x - beta / torch.sqrt(1 - alpha_cumprod) * noise_pred) + torch.sqrt(beta) * noise
        
        # Convert embeddings to sequences (this would require a decoder network in practice)
        # For now, we'll just return the embeddings
        return x
    
    def ddim_sample(self, n_samples=1, seq_len=30, steps=50, eta=0.0, custom_loss_fn=None, device=None):
        """Sample using DDIM (Denoising Diffusion Implicit Models) for faster sampling."""
        if device is None:
            device = next(self.parameters()).device
        
        # Start from random noise
        x = torch.randn(n_samples, self.esm_dim, device=device)
        
        # Select subset of timesteps for faster sampling
        timesteps = torch.linspace(self.n_steps - 1, 0, steps, dtype=torch.long, device=device)
        
        for i, t in enumerate(timesteps):
            # Create timestep batch
            t_batch = torch.full((n_samples,), t, device=device, dtype=torch.long)
            
            # Time embeddings
            time_emb = self.time_mlp(t_batch)
            
            # Predict noise
            noise_pred = self.noise_pred(torch.cat([x, time_emb], dim=1))
            
            # Apply custom loss if provided
            if custom_loss_fn is not None:
                noise_pred = noise_pred + custom_loss_fn(x, t_batch)
            
            # Extract alpha values
            alpha_cumprod_t = self.alpha_cumprod[t]
            alpha_cumprod_prev = self.alpha_cumprod[t-1] if t > 0 else torch.tensor(1.0, device=device)
            
            # Variance
            sigma = eta * torch.sqrt((1 - alpha_cumprod_prev) / (1 - alpha_cumprod_t) * (1 - alpha_cumprod_t / alpha_cumprod_prev))
            
            # Noise
            noise = torch.zeros_like(x) if i == len(timesteps) - 1 else torch.randn_like(x)
            
            # DDIM update
            pred_x0 = (x - torch.sqrt(1 - alpha_cumprod_t) * noise_pred) / torch.sqrt(alpha_cumprod_t)
            dir_xt = torch.sqrt(1 - alpha_cumprod_prev - sigma**2) * noise_pred
            x = torch.sqrt(alpha_cumprod_prev) * pred_x0 + dir_xt + sigma * noise
        
        return x
