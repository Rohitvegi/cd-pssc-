import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool

class SE3Layer(nn.Module):
    """
    Simple SE(3)-equivariant layer.
    """
    def __init__(self, in_channels, out_channels, edge_dim=5):
        super(SE3Layer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.edge_dim = edge_dim
        
        # MLP for node feature transformation
        self.node_mlp = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels)
        )
        
        # MLP for edge feature transformation
        self.edge_mlp = nn.Sequential(
            nn.Linear(in_channels * 2 + edge_dim, out_channels),  # edge_dim for edge attributes
            nn.ReLU(),
            nn.Linear(out_channels, out_channels)
        )
        
        # Final node update
        self.update = nn.Linear(out_channels * 2, out_channels)
    
    def forward(self, x, edge_index, edge_attr, pos):
        """
        Forward pass for SE3 layer.

        Args:
        x: Node features [num_nodes, in_channels]
        edge_index: Edge indices [2, num_edges]
        edge_attr: Edge attributes [num_edges, edge_dim]
        pos: Node positions [num_nodes, 3]
        
        Returns:
            Updated node features [num_nodes, out_channels]
        """
        # Get device
        device = x.device
    
        # Node feature transformation
        h_nodes = self.node_mlp(x)
    
        # Edge feature transformation
        # If there are no edges, return node features
        if edge_index.size(1) == 0:
            return h_nodes
        
        src, dst = edge_index
    
        # Get source and destination node features
        h_src = x[src]
        h_dst = x[dst]
    
        # Concatenate source, destination features, and edge attributes
        edge_features = torch.cat([h_src, h_dst, edge_attr], dim=1)
    
        # Transform edge features
        h_edges = self.edge_mlp(edge_features)
    
        # Aggregate edge features to nodes
        h_agg = torch.zeros_like(h_nodes, device=device)
        for i in range(edge_index.size(1)):
            h_agg[dst[i]] += h_edges[i]
    
        # Combine node and aggregated edge features
        h_combined = torch.cat([h_nodes, h_agg], dim=1)
    
        # Final update
        h_out = self.update(h_combined)
    
        return h_out


class MultiScaleSE3Simple(nn.Module):
    """
    Multi-scale SE(3)-equivariant network for peptide structure encoding.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, edge_dim=5, num_layers=3):
        super(MultiScaleSE3Simple, self).__init__()
        
        # Input dimensions
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.edge_dim = edge_dim
        self.num_layers = num_layers
        
        # Node embedding layer
        self.node_embedding = nn.Linear(self.input_dim, self.hidden_dim)
        
        # SE3 layers
        self.layers = nn.ModuleList([
            SE3Layer(self.hidden_dim, self.hidden_dim, edge_dim=self.edge_dim) 
            for _ in range(self.num_layers)
        ])
        
        # Output layer
        self.output_layer = nn.Linear(self.hidden_dim, self.output_dim)
    
    def forward(self, graph):
        """
        Forward pass for SE3 network.
        
        Args:
            graph: Graph data with node features, edge indices, edge attributes, and positions
            
        Returns:
            Dictionary containing node embeddings and graph embedding
        """
        # Get device from the node_embedding weight
        device = next(self.parameters()).device
        
        # Move graph components to the correct device
        x = graph.x.to(device)
        edge_index = graph.edge_index.to(device)
        edge_attr = graph.edge_attr.to(device)
        pos = graph.pos.to(device)
        
        # Handle batch information
        if hasattr(graph, 'batch') and graph.batch is not None:
            batch = graph.batch.to(device)
        else:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=device)
        
        # Check if input dimension matches the expected dimension
        if x.size(1) != self.input_dim:
            # Adjust the input to match the expected dimension
            if x.size(1) > self.input_dim:
                # If input has more features than expected, take the first self.input_dim features
                x = x[:, :self.input_dim]
            else:
                # If input has fewer features than expected, pad with zeros
                padding = torch.zeros(x.size(0), self.input_dim - x.size(1), device=device)
                x = torch.cat([x, padding], dim=1)
        
        # Initial node embeddings
        h = self.node_embedding(x)
        
        # Process through SE3 layers
        for layer in self.layers:
            h = layer(h, edge_index, edge_attr, pos)
        
        # Final node embeddings
        node_embeddings = self.output_layer(h)
        
        # Global pooling
        graph_embedding = global_mean_pool(node_embeddings, batch)
        
        return {
            'node_embeddings': node_embeddings,
            'graph_embedding': graph_embedding
        }
