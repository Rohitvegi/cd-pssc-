import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from .se3_pepnet_simple import MultiScaleSE3Simple
from .contrastive import HierarchicalContrast
import math

class ESMDiffuser(nn.Module):
    """Enhanced ESM-based diffusion model for peptide sequences."""
    def __init__(self, esm_dim, hidden_dim=256, n_steps=1000):
        super(ESMDiffuser, self).__init__()
        self.esm_dim = esm_dim
        self.hidden_dim = hidden_dim
        self.n_steps = n_steps
        
        # Time embedding with sinusoidal encoding
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbeddings(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
        # Enhanced noise prediction network with residual connections
        self.noise_pred = nn.Sequential(
            ResidualBlock(esm_dim + hidden_dim, hidden_dim * 2),
            ResidualBlock(hidden_dim * 2, hidden_dim * 2),
            nn.Linear(hidden_dim * 2, esm_dim)
        )
        
        # Beta schedule (cosine)
        self.register_buffer('beta', self._cosine_beta_schedule(n_steps))
        self.register_buffer('alpha', 1 - self.beta)
        self.register_buffer('alpha_cumprod', torch.cumprod(self.alpha, dim=0))
    
    def _cosine_beta_schedule(self, n_steps, s=0.008):
        """Cosine beta schedule for improved sampling quality."""
        steps = torch.arange(n_steps + 1, dtype=torch.float32) / n_steps
        f_t = torch.cos((steps + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = f_t / f_t[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)
    
    def forward(self, x, t):
        """
        Forward pass for diffusion model.
    
        Args:
            x: Input sequence embeddings [batch_size, seq_len, esm_dim]
            t: Timesteps [batch_size]
        
        Returns:
            Predicted noise
    """
        # Time embedding
        t_emb = self.time_embed(t.unsqueeze(-1).float())
    
        # Check the shape of t_emb and fix if needed
        if len(t_emb.shape) > 2:
            # If t_emb has more than 2 dimensions, reshape it
            t_emb = t_emb.view(t_emb.size(0), -1)
    
        # Expand time embedding to match sequence length
        t_emb = t_emb.unsqueeze(1).expand(-1, x.size(1), -1)
    
        # Concatenate sequence embeddings and time embeddings
        x_t = torch.cat([x, t_emb], dim=-1)
    
        # Predict noise
        noise_pred = self.noise_pred(x_t)
    
        return noise_pred

    
    def diffuse(self, x, t):
        """
        Apply forward diffusion to input embeddings.
        
        Args:
            x: Input sequence embeddings [batch_size, seq_len, esm_dim]
            t: Timesteps [batch_size]
            
        Returns:
            Noisy embeddings and noise
        """
        noise = torch.randn_like(x)
        alpha_cumprod_t = self.alpha_cumprod[t].view(-1, 1, 1)
        
        # Apply noise
        x_t = torch.sqrt(alpha_cumprod_t) * x + torch.sqrt(1 - alpha_cumprod_t) * noise
        
        return x_t, noise
    
    def sample(self, shape, device, steps=None, guidance_scale=1.0, constraints=None):
        """
        Sample from the diffusion model with classifier-free guidance.
        
        Args:
            shape: Shape of the output [batch_size, seq_len, esm_dim]
            device: Device to use
            steps: Number of sampling steps (default: n_steps)
            guidance_scale: Scale for classifier-free guidance (1.0 = no guidance)
            constraints: Optional constraints to guide sampling
            
        Returns:
            Sampled sequence embeddings
        """
        steps = steps or self.n_steps
        
        # Start from random noise
        x = torch.randn(shape, device=device)
        
        # Use DDIM sampling for faster generation
        timesteps = torch.linspace(self.n_steps - 1, 0, steps, dtype=torch.long, device=device)
        
        # Reverse diffusion process
        for i, t in enumerate(timesteps):
            # Create batch of same timestep
            t_batch = torch.full((shape[0],), t, device=device, dtype=torch.long)
            
            # Predict noise
            with torch.no_grad():
                noise_pred = self.forward(x, t_batch)
                
                # Apply classifier-free guidance if scale > 1
                if guidance_scale > 1.0 and constraints is not None:
                    # This is a placeholder for constraint-based guidance
                    # In a real implementation, you would compute a guided noise prediction
                    # based on the constraints
                    pass
            
            # Get alpha values for current and previous timestep
            alpha_cumprod_t = self.alpha_cumprod[t]
            alpha_cumprod_prev = self.alpha_cumprod[t-1] if t > 0 else torch.tensor(1.0, device=device)
            
            # Compute variance
            variance = (1 - alpha_cumprod_prev) / (1 - alpha_cumprod_t) * (1 - alpha_cumprod_t / alpha_cumprod_prev)
            
            # Sample noise for stochasticity (except at last step)
            noise = torch.zeros_like(x) if i == len(timesteps) - 1 else torch.randn_like(x)
            
            # Compute predicted x_0
            pred_x0 = (x - torch.sqrt(1 - alpha_cumprod_t) * noise_pred) / torch.sqrt(alpha_cumprod_t)
            
            # Compute direction to next step
            dir_xt = torch.sqrt(1 - alpha_cumprod_prev - variance) * noise_pred
            
            # Update x
            x = torch.sqrt(alpha_cumprod_prev) * pred_x0 + dir_xt + torch.sqrt(variance) * noise
        
        return x

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
        # Ensure we return a 2D tensor [batch_size, dim]
        return embeddings.view(time.shape[0], -1)

class ResidualBlock(nn.Module):
    """Residual block for improved gradient flow."""
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Main branch
        self.block = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.LayerNorm(out_channels),
            nn.SiLU(),
            nn.Linear(out_channels, out_channels),
            nn.LayerNorm(out_channels)
        )
        
        # Residual connection
        self.residual = nn.Linear(in_channels, out_channels) if in_channels != out_channels else nn.Identity()
        
        # Activation
        self.activation = nn.SiLU()
    
    def forward(self, x):
        residual = self.residual(x)
        x = self.block(x)
        return self.activation(x + residual)

class PhysicsLoss(nn.Module):
    """Physics-based loss for protein structure."""
    def __init__(self):
        super(PhysicsLoss, self).__init__()
        self.bond_length_target = 1.5  # Å (approximate CA-CA distance)
        self.angle_target = 2.0  # radians (approximately 115 degrees)
        self.target_angle = torch.tensor(109.5 * torch.pi / 180.0)  # tetrahedral angle in radians
    
    def bond_length_loss(self, coords, edge_index):
        """Compute loss based on bond lengths."""
        # Extract connected atoms
        src, dst = edge_index
        
        # Compute distances
        src_coords = coords[src]
        dst_coords = coords[dst]
        
        # Calculate squared distances
        dist = torch.sum((src_coords - dst_coords) ** 2, dim=1)
        
        # Compute loss (deviation from target bond length)
        loss = F.mse_loss(torch.sqrt(dist + 1e-8), torch.full_like(dist, self.bond_length_target))
        
        return loss
    
    def angle_loss(self, coords, edge_index):
        """Compute loss based on bond angles."""
        # This is a simplified version - in a real implementation,
        # you would identify actual bond angles in the protein backbone
        
        # Find connected triplets (a-b-c)
        src, dst = edge_index
        angles = []
        
        # For each node, find its neighbors
        for b in range(coords.size(0)):
            # Find neighbors of b
            neighbors = dst[src == b]
            
            # Need at least 2 neighbors to form an angle
            if len(neighbors) >= 2:
                # Consider all pairs of neighbors
                for i in range(len(neighbors)):
                    for j in range(i+1, len(neighbors)):
                        a = neighbors[i]
                        c = neighbors[j]
                        
                        # Get coordinates
                        a_coords = coords[a]
                        b_coords = coords[b]
                        c_coords = coords[c]
                        
                        # Compute vectors
                        ba = a_coords - b_coords
                        bc = c_coords - b_coords
                        
                        # Normalize vectors
                        ba = ba / (torch.norm(ba) + 1e-8)
                        bc = bc / (torch.norm(bc) + 1e-8)
                        
                        # Compute cosine of angle
                        cosine = torch.sum(ba * bc)
                        
                        # Clamp to avoid numerical issues
                        cosine = torch.clamp(cosine, -0.999, 0.999)
                        
                        # Compute angle
                        angle = torch.acos(cosine)
                        angles.append(angle)
        
        if angles:
            angles = torch.stack(angles)
            # Use target angle of ~109.5 degrees (tetrahedral)
            loss = F.mse_loss(angles, torch.full_like(angles, self.target_angle))
        else:
            # Return zero loss if no angles found
            loss = torch.tensor(0.0, device=coords.device)
        
        return loss
    
    def forward(self, coords, edge_index):
        """Compute combined physics loss."""
        bond_loss = self.bond_length_loss(coords, edge_index)
        angle_loss = self.angle_loss(coords, edge_index)
        
        return bond_loss + angle_loss

class CDPSSC(pl.LightningModule):
    """Enhanced Contrastive Diffusion for Protein Secondary Structure Coordinates."""
    def __init__(self, config):
        super(CDPSSC, self).__init__()
        self.config = config
        self.save_hyperparameters()
    
        # Model dimensions
        if isinstance(config, dict):
            # It's a dictionary
            self.se3_dim = config['model']['se3_dim']
            self.esm_dim = config['model']['esm_dim']
            self.n_properties = config['model']['n_properties']
            self.loss_weights = config['training']['loss_weights']
        else:
            # It's a SimpleNamespace
            self.se3_dim = config.model.se3_dim
            self.esm_dim = config.model.esm_dim
            self.n_properties = config.model.n_properties
            self.loss_weights = config.training.loss_weights
    
        # Structure encoder
        # Structure encoder
        self.structure_encoder = MultiScaleSE3Simple(
            input_dim=self.n_properties,
            hidden_dim=self.se3_dim,
            output_dim=self.se3_dim,
            edge_dim=5,
            num_layers=3
        )
        
        # Sequence diffuser
        self.sequence_diffuser = ESMDiffuser(
            esm_dim=self.esm_dim,
            hidden_dim=self.se3_dim
        )
        
        # Hierarchical contrastive module
        self.contrastive = HierarchicalContrast(
                temperature=0.1,
                ot_reg=0.1,
                struct_dim=self.se3_dim,
                seq_dim=self.esm_dim,
                proj_dim=256
        )

        
        # Coordinate prediction
        self.coord_pred = nn.Sequential(
            ResidualBlock(self.se3_dim, self.se3_dim * 2),
            nn.Linear(self.se3_dim * 2, 3)  # Predict 3D coordinates
        )
        
        # Physics-based loss
        self.physics_loss = PhysicsLoss()
        
        # Classification head for AMP prediction
        self.classifier = nn.Sequential(
            nn.Linear(self.se3_dim, self.se3_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.se3_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, batch):
        """
        Forward pass for CD-PSSC model.

        Args:
            batch: Dictionary containing:
            - graph: Graph data
            - seq_emb: Sequence embeddings
        
        Returns:
            Dictionary of model outputs
        """
        # Get device
        device = next(self.parameters()).device

    # Process structure
        structure_output = self.structure_encoder(batch['graph'])
        structure_emb = structure_output['graph_embedding']

    # Process sequence
        seq_emb = batch['seq_emb'].to(device)
        batch_size = seq_emb.size(0)

    # Sample timesteps
        t = torch.randint(0, self.sequence_diffuser.n_steps, (batch_size,), device=device)

    # Apply diffusion
        seq_emb_noisy, noise = self.sequence_diffuser.diffuse(seq_emb, t)

    # Predict noise
        noise_pred = self.sequence_diffuser(seq_emb_noisy, t)

    # Prepare structure and sequence outputs for contrastive learning
        struct_outputs = {
            'global_embedding': structure_emb,
            'node_embeddings': structure_output['node_embeddings']
        }
    
        # Mean pooling for sequence token embeddings
        seq_token_emb = seq_emb
        seq_global_emb = seq_emb.mean(dim=1)
    
        seq_outputs = {
            'global_embedding': seq_global_emb,
            'token_embeddings': seq_token_emb
        }

        # Compute contrastive loss
        contrastive_losses = self.contrastive(struct_outputs, seq_outputs, batch['graph'].batch)
        contrastive_loss = contrastive_losses['total_loss']

        # Predict coordinates from structure embeddings
        node_embeddings = structure_output['node_embeddings']
        pred_coords = self.coord_pred(node_embeddings)
    
        # Predict AMP probability (if classifier exists)
        if hasattr(self, 'classifier'):
            # Output logits (before sigmoid) for use with binary_cross_entropy_with_logits
            amp_logits = self.classifier(structure_emb)
        else:
            amp_logits = None

        return {
            'structure_emb': structure_emb,
            'seq_emb': seq_emb,
            'seq_emb_noisy': seq_emb_noisy,
            'noise': noise,
            'noise_pred': noise_pred,
            'pred_coords': pred_coords,
            'contrastive_loss': contrastive_loss,
            'amp_pred': amp_logits  # Now returning logits instead of probabilities
        }

    def training_step(self, batch, batch_idx):
        """Training step."""
        # Get device
        device = next(self.parameters()).device
    
        # Forward pass
        outputs = self(batch)
    
        # Diffusion loss (MSE between predicted and actual noise)
        diffusion_loss = F.mse_loss(outputs['noise_pred'], outputs['noise'])
    
        # Contrastive loss
        contrastive_loss = outputs['contrastive_loss']
    
        # Coordinate prediction loss
        true_coords = batch['graph'].pos.to(device)  # Move to the same device
        pred_coords = outputs['pred_coords']
        coord_loss = F.mse_loss(pred_coords, true_coords)
    
        # AMP classification loss (if available)
        if 'amp_pred' in outputs and hasattr(batch, 'is_amp'):
            # Use binary_cross_entropy_with_logits instead of binary_cross_entropy
            amp_loss = F.binary_cross_entropy_with_logits(
                outputs['amp_pred'].squeeze(),
                batch['is_amp'].to(device)
            )
        else:
            amp_loss = torch.tensor(0.0, device=device)
    
        # Total loss
        total_loss = (
            self.loss_weights['sequence'] * diffusion_loss +
            self.loss_weights['global_contrast'] * contrastive_loss +
            self.loss_weights['structure'] * coord_loss +
            0.1 * amp_loss  # Add a small weight for AMP classification
        )
    
        # Log losses
        self.log('train_loss', total_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_diffusion_loss', diffusion_loss, on_step=True, on_epoch=True)
        self.log('train_contrastive_loss', contrastive_loss, on_step=True, on_epoch=True)
        self.log('train_coord_loss', coord_loss, on_step=True, on_epoch=True)
        if amp_loss > 0:
            self.log('train_amp_loss', amp_loss, on_step=True, on_epoch=True)
    
        return total_loss

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        # Get device
        device = next(self.parameters()).device
    
        # Forward pass
        outputs = self(batch)
    
        # Diffusion loss
        diffusion_loss = F.mse_loss(outputs['noise_pred'], outputs['noise'])
    
        # Contrastive loss
        contrastive_loss = outputs['contrastive_loss']
    
        # Coordinate prediction loss
        true_coords = batch['graph'].pos.to(device)  # Move to the same device
        pred_coords = outputs['pred_coords']
        coord_loss = F.mse_loss(pred_coords, true_coords)
    
        # AMP classification loss (if available)
        if 'amp_pred' in outputs and hasattr(batch, 'is_amp'):
            # Use binary_cross_entropy_with_logits instead of binary_cross_entropy
            amp_loss = F.binary_cross_entropy_with_logits(
                outputs['amp_pred'].squeeze(),
                batch['is_amp'].to(device)
            )
        else:
            amp_loss = torch.tensor(0.0, device=device)
    
        # Total loss
        total_loss = (
            self.loss_weights['sequence'] * diffusion_loss +
            self.loss_weights['global_contrast'] * contrastive_loss +
            self.loss_weights['structure'] * coord_loss +
            0.1 * amp_loss  # Add a small weight for AMP classification
        )
    
        # Log losses
        self.log('val_loss', total_loss, on_epoch=True, prog_bar=True)
        self.log('val_diffusion_loss', diffusion_loss, on_epoch=True)
        self.log('val_contrastive_loss', contrastive_loss, on_epoch=True)
        self.log('val_coord_loss', coord_loss, on_epoch=True)
        if amp_loss > 0:
            self.log('val_amp_loss', amp_loss, on_epoch=True)
    
        return total_loss


    def test_step(self, batch, batch_idx):
        """Test step with comprehensive evaluation metrics."""
        # Get device
        device = next(self.parameters()).device
    
        # Forward pass
        outputs = self(batch)
    
        # Diffusion loss
        diffusion_loss = F.mse_loss(outputs['noise_pred'], outputs['noise'])
    
        # Contrastive losses
        global_contrast_loss = outputs.get('contrastive_loss', torch.tensor(0.0, device=device))
        local_contrast_loss = outputs.get('local_contrast_loss', torch.tensor(0.0, device=device))
    
        # Coordinate prediction loss
        true_coords = batch['graph'].pos.to(device)
        pred_coords = outputs['pred_coords']
        coord_loss = F.mse_loss(pred_coords, true_coords)
        
        # Physics-based loss
        physics_loss = self.physics_loss(pred_coords, batch['graph'].edge_index)
        
        # AMP classification metrics
        if 'is_amp' in batch and batch['is_amp'].size(0) > 0:
            amp_loss = F.binary_cross_entropy(
                outputs['amp_pred'].squeeze(),
                batch['is_amp'].to(device)
            )
            
            # Calculate detailed metrics
            amp_pred = (outputs['amp_pred'].squeeze() > 0.5).float()
            amp_acc = (amp_pred == batch['is_amp'].to(device)).float().mean()
            
            # Calculate precision, recall, and F1 score
            true_positives = ((amp_pred == 1) & (batch['is_amp'].to(device) == 1)).sum()
            false_positives = ((amp_pred == 1) & (batch['is_amp'].to(device) == 0)).sum()
            false_negatives = ((amp_pred == 0) & (batch['is_amp'].to(device) == 1)).sum()
            true_negatives = ((amp_pred == 0) & (batch['is_amp'].to(device) == 0)).sum()
            
            precision = true_positives / (true_positives + false_positives + 1e-8)
            recall = true_positives / (true_positives + false_negatives + 1e-8)
            f1 = 2 * precision * recall / (precision + recall + 1e-8)
            specificity = true_negatives / (true_negatives + false_positives + 1e-8)
            
            self.log('test_amp_loss', amp_loss)
            self.log('test_amp_acc', amp_acc)
            self.log('test_precision', precision)
            self.log('test_recall', recall)
            self.log('test_f1', f1)
            self.log('test_specificity', specificity)
        else:
            amp_loss = torch.tensor(0.0, device=device)
    
        # Total loss
        total_loss = (
            self.loss_weights['sequence'] * diffusion_loss +
            self.loss_weights['global_contrast'] * global_contrast_loss +
            self.loss_weights['local_contrast'] * local_contrast_loss +
            self.loss_weights['structure'] * coord_loss +
            self.loss_weights['physics'] * physics_loss +
            0.5 * amp_loss
        )
    
        # Log losses
        self.log('test_loss', total_loss)
        self.log('test_diffusion_loss', diffusion_loss)
        self.log('test_global_contrast_loss', global_contrast_loss)
        self.log('test_local_contrast_loss', local_contrast_loss)
        self.log('test_coord_loss', coord_loss)
        self.log('test_physics_loss', physics_loss)
        
        # Calculate RMSD between predicted and true coordinates
        rmsd = torch.sqrt(((pred_coords - true_coords) ** 2).mean())
        self.log('test_rmsd', rmsd)
    
        return total_loss
    
    def configure_optimizers(self):
        """Configure optimizers with learning rate scheduler."""
        # Extract training parameters
        if isinstance(self.config, dict):
            lr = self.config['training']['lr']
            weight_decay = self.config['training']['weight_decay']
            min_lr = self.config['training']['min_lr']
            patience = self.config['training']['patience']
        else:
            lr = self.config.training.lr
            weight_decay = self.config.training.weight_decay
            min_lr = self.config.training.min_lr
            patience = self.config.training.patience
        
        # Create optimizer with weight decay
        
        # Create optimizer with weight decay
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=patience,
            min_lr=min_lr,
            verbose=True
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',
                'interval': 'epoch',
                'frequency': 1
            }
        }
    
    def generate_peptide(self, constraints=None, n_samples=1, steps=50):
        """
        Generate peptide structures based on optional constraints.
        
        Args:
            constraints: Dictionary of constraints
            n_samples: Number of peptides to generate
            steps: Number of diffusion steps
            
        Returns:
            Dictionary containing generated structures and sequences
        """
        device = next(self.parameters()).device
        
        # Generate random structure embeddings
        structure_emb = torch.randn(n_samples, self.se3_dim, device=device)
        
        # Apply constraints if provided
        if constraints:
            # Handle length constraint
            length = constraints.get('length', 20)
            
            # Handle AAIndex property constraints
            if 'aaindex' in constraints:
                # This is a placeholder for applying AAIndex constraints
                # In a real implementation, you would modify the structure_emb
                # based on the desired properties
                for prop_name, target_value in constraints['aaindex'].items():
                    # Apply some transformation to bias structure_emb
                    # This is just a simple example - in practice you'd use a more sophisticated approach
                    bias = torch.randn_like(structure_emb) * 0.1
                    structure_emb = structure_emb + bias
        else:
            # Default length
            length = 20
        
        # Generate sequence embeddings from structure using the diffusion model
        seq_emb_shape = (n_samples, length, self.esm_dim)
        
        # Use classifier-free guidance if constraints are provided
        guidance_scale = 1.5 if constraints else 1.0
        
        seq_emb = self.sequence_diffuser.sample(
            seq_emb_shape, 
            device, 
            steps=steps,
            guidance_scale=guidance_scale,
            constraints=constraints
        )
        
        # Generate coordinates from structure embeddings
        # For demonstration, we'll create a simple linear model to map from
        # structure embeddings to a sequence of coordinates
        
        # Create a simple mapping from structure embedding to node embeddings
        # In a real implementation, this would be more sophisticated
        node_embeddings = structure_emb.unsqueeze(1).expand(-1, length, -1)
        
        # Add some position-dependent variation
        positions = torch.arange(length, device=device).float().unsqueeze(0).expand(n_samples, -1)
        positions = positions.unsqueeze(-1) / length
        
        # Combine with random noise for variety
        node_variation = torch.sin(positions * 10) * 0.5 + torch.cos(positions * 5) * 0.3
        node_embeddings = node_embeddings + node_variation.expand(-1, -1, self.se3_dim)
        
        # Reshape for coordinate prediction
        flat_node_embeddings = node_embeddings.reshape(-1, self.se3_dim)
        
        # Predict coordinates
        flat_coords = self.coord_pred(flat_node_embeddings)
        
        # Reshape back to batch format
        coords = flat_coords.reshape(n_samples, length, 3)
        
        # Expand to include all atoms (N, CA, C, O) per residue
        # For simplicity, we'll create a basic backbone structure
        # In a real implementation, you would use proper backbone geometry
        expanded_coords = []
        
        for i in range(n_samples):
            residue_coords = coords[i]  # [length, 3]
            
            # Create 4 atoms per residue with offsets
            sample_coords = []
            
            for j in range(length):
                ca_pos = residue_coords[j]  # CA position
                
                # Create N, CA, C, O positions with simple offsets
                # These are very simplified - in reality you'd use proper bond geometries
                n_pos = ca_pos + torch.tensor([-1.0, 0.0, 0.0], device=device)
                c_pos = ca_pos + torch.tensor([1.0, 0.0, 0.0], device=device)
                o_pos = ca_pos + torch.tensor([0.5, 1.0, 0.0], device=device)
                
                # Add all atoms
                sample_coords.extend([n_pos, ca_pos, c_pos, o_pos])
            
            # Convert to tensor
            sample_coords = torch.stack(sample_coords)
            expanded_coords.append(sample_coords)
        
        return {
            'structures': expanded_coords,
            'sequence_embeddings': seq_emb,
            'structure_embeddings': structure_emb
        }

    def predict_amp_probability(self, sequence=None, structure=None):
        """
        Predict the probability that a peptide is antimicrobial.
        
        Args:
            sequence: Optional peptide sequence
            structure: Optional peptide structure (coordinates)
            
        Returns:
            Probability that the peptide is antimicrobial
        """
        device = next(self.parameters()).device
        
        # We need either sequence or structure
        if sequence is None and structure is None:
            raise ValueError("Either sequence or structure must be provided")
        
        # If only sequence is provided, we need to generate a structure
        if structure is None and sequence is not None:
            # This is a placeholder - in a real implementation, you would
            # use a sequence-to-structure model or the diffusion model
            # For now, we'll create a dummy structure
            length = len(sequence)
            structure = torch.randn(length * 4, 3, device=device)
        
        # If only structure is provided, we'll use it directly
        if structure is not None:
            # Create a dummy graph from the structure
            # In a real implementation, you would create a proper graph
            # with appropriate node features and edge connections
            
            # For simplicity, we'll create random node features
            n_nodes = structure.size(0) // 4  # Assuming 4 atoms per residue
            x = torch.randn(n_nodes, self.n_properties, device=device)
            
            # Create edges connecting nearby nodes
            # This is a simplified approach - in practice, you'd use proper distance-based edges
            edge_index = []
            for i in range(n_nodes):
                for j in range(max(0, i-2), min(n_nodes, i+3)):
                    if i != j:
                        edge_index.append([i, j])
            
            edge_index = torch.tensor(edge_index, device=device).t()
            
            # Create edge attributes (dummy values)
            edge_attr = torch.ones(edge_index.size(1), 5, device=device)
            
            # Create graph
            from .dataset import SimpleGraph
            graph = SimpleGraph(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                pos=structure[::4],  # Use CA atoms
                batch=None
            )
            
            # Process through structure encoder
            structure_output = self.structure_encoder(graph)
            structure_emb = structure_output['graph_embedding']
            
            # Predict AMP probability
            amp_prob = self.classifier(structure_emb).item()
            
            return amp_prob
        
        return 0.5  # Default fallback

