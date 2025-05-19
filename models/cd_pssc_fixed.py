import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from .se3_pepnet_simple_fixed import MultiScaleSE3Simple
from .contrastive import HierarchicalContrast

# Import the original CDPSSC class
from .cd_pssc import CDPSSC as OriginalCDPSSC

# Create a fixed version that overrides the test_step method
class CDPSSC(OriginalCDPSSC):
    def test_step(self, batch, batch_idx):
        """Test step with fixed contrastive loss handling."""
        # Get device
        device = next(self.parameters()).device
    
        # Forward pass
        outputs = self(batch)
    
        # Diffusion loss
        diffusion_loss = F.mse_loss(outputs['noise_pred'], outputs['noise'])
    
        # Contrastive loss - safely access
        contrastive_loss = outputs.get('contrastive_loss', torch.tensor(0.0, device=device))
    
        # Coordinate prediction loss
        true_coords = batch['graph'].pos.to(device)  # Move to the same device
        pred_coords = outputs['pred_coords']
        coord_loss = F.mse_loss(pred_coords, true_coords)
    
        # Total loss
        total_loss = (
            self.loss_weights['sequence'] * diffusion_loss +
            self.loss_weights['global_contrast'] * contrastive_loss +
            self.loss_weights['structure'] * coord_loss
        )
    
        # Log losses
        self.log('test_loss', total_loss)
        self.log('test_diffusion_loss', diffusion_loss)
        self.log('test_contrastive_loss', contrastive_loss)
        self.log('test_coord_loss', coord_loss)
    
        return total_loss
