import torch
import torch.nn as nn
import torch.nn.functional as F
import ot
class HierarchicalContrast(nn.Module):
    """Hierarchical contrastive learning with global and local components."""
    
    def __init__(self, temperature=0.1, ot_reg=0.1, struct_dim=256, seq_dim=1280, proj_dim=256):
        super(HierarchicalContrast, self).__init__()
        self.temperature = temperature
        self.ot_reg = ot_reg
        
        # Projection heads for global contrast
        self.struct_proj = nn.Linear(struct_dim, proj_dim)
        self.seq_proj = nn.Linear(seq_dim, proj_dim)
        
        # Projection heads for local contrast
        self.struct_node_proj = nn.Linear(struct_dim, proj_dim)
        self.seq_token_proj = nn.Linear(seq_dim, proj_dim)

    
    def global_nt_xent_loss(self, struct_emb, seq_emb):
        """Global NT-Xent (normalized temperature-scaled cross entropy) loss."""
        # Project embeddings to the same dimension if they don't match
        if not hasattr(self, 'struct_proj'):
            # Create projection heads if they don't exist
            self.struct_proj = nn.Linear(struct_emb.size(1), 256).to(struct_emb.device)
            self.seq_proj = nn.Linear(seq_emb.size(1), 256).to(seq_emb.device)
    
        # Project embeddings
        struct_proj = self.struct_proj(struct_emb)
        seq_proj = self.seq_proj(seq_emb)
    
        # Normalize embeddings
        struct_proj = F.normalize(struct_proj, dim=1)
        seq_proj = F.normalize(seq_proj, dim=1)
    
        # Compute similarity matrix
        sim_matrix = torch.matmul(struct_proj, seq_proj.T) / self.temperature
    
        # Labels: diagonal is positive pairs
        labels = torch.arange(sim_matrix.size(0), device=sim_matrix.device)
    
    # Compute loss
        loss_i = F.cross_entropy(sim_matrix, labels)
        loss_j = F.cross_entropy(sim_matrix.T, labels)
    
        return (loss_i + loss_j) / 2.0

    
    def local_ot_max_loss(self, struct_node_emb, seq_token_emb, batch_idx=None):
        """Local contrastive loss using optimal transport."""
        device = struct_node_emb.device
    
        # If batch_idx is None, assume each graph is a separate example
        if batch_idx is None:
            batch_idx = torch.zeros(struct_node_emb.size(0), dtype=torch.long, device=device)
    
        # Project embeddings to the same dimension if needed
        if not hasattr(self, 'struct_node_proj'):
            self.struct_node_proj = nn.Linear(struct_node_emb.size(1), 256).to(device)
            self.seq_token_proj = nn.Linear(seq_token_emb.size(2), 256).to(device)
    
        # Project and normalize structure node embeddings
        struct_node_emb = self.struct_node_proj(struct_node_emb)
        struct_node_emb = F.normalize(struct_node_emb, dim=1)
    
        # Compute loss for each example in the batch
        total_loss = 0.0
        num_examples = batch_idx.max().item() + 1
        batch_size = seq_token_emb.size(0)
    
        # Make sure num_examples doesn't exceed batch_size
        num_examples = min(num_examples, batch_size)
    
        for i in range(num_examples):
            # Get structure node embeddings for current example
            struct_nodes = struct_node_emb[batch_idx == i]
        
            # If no nodes for this batch index, skip
            if struct_nodes.size(0) == 0:
                continue
        
            # Get sequence token embeddings for current example
            # seq_token_emb is [batch_size, seq_len, emb_dim]
            seq_tokens = seq_token_emb[i]  # [seq_len, emb_dim]
        
            # Project and normalize sequence token embeddings
            seq_tokens = self.seq_token_proj(seq_tokens)  # [seq_len, proj_dim]
            seq_tokens = F.normalize(seq_tokens, dim=1)
        
            # Compute cost matrix (negative similarity)
            cost_matrix = -torch.matmul(struct_nodes, seq_tokens.T)  # [n_nodes, seq_len]
        
            # Uniform weights
            a = torch.ones(struct_nodes.size(0), device=device) / struct_nodes.size(0)
            b = torch.ones(seq_tokens.size(0), device=device) / seq_tokens.size(0)
        
            try:
                # Solve optimal transport problem
                ot_matrix = ot.sinkhorn(
                    a.cpu().numpy(), 
                    b.cpu().numpy(), 
                    cost_matrix.detach().cpu().numpy(), 
                    reg=self.ot_reg
                )
                ot_matrix = torch.tensor(ot_matrix, device=device)
            
                # Compute loss
                loss = torch.sum(ot_matrix * cost_matrix)
                total_loss += loss
            except Exception as e:
                print(f"Error in OT calculation: {e}")
                # Fallback to simple cosine similarity loss
                sim = torch.matmul(struct_nodes.mean(0, keepdim=True), seq_tokens.mean(0, keepdim=True).T)
                loss = -sim.mean()
                total_loss += loss
    
        # Return average loss
        return total_loss / max(num_examples, 1)

    def forward(self, struct_outputs, seq_outputs, batch_idx=None):
        """
        Compute hierarchical contrastive loss.
        
        Args:
            struct_outputs: Dictionary with structure embeddings
            seq_outputs: Dictionary with sequence embeddings
            batch_idx: Optional batch indices for local contrastive loss
            
        Returns:
            Dictionary with loss components
        """
        # Global contrastive loss
        global_loss = self.global_nt_xent_loss(
            struct_outputs['global_embedding'],
            seq_outputs['global_embedding']
        )
        
        # Local contrastive loss
        local_loss = self.local_ot_max_loss(
            struct_outputs['node_embeddings'],
            seq_outputs['token_embeddings'],
            batch_idx
        )
        
        # Combined loss
        total_loss = global_loss + local_loss
        
        return {
            'global_loss': global_loss,
            'local_loss': local_loss,
            'total_loss': total_loss
        }
