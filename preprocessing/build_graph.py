import torch
import torch_geometric
from torch_geometric.data import Data
import os
from tqdm import tqdm

def create_graph(coords, sequence, aa_embedding, k=20):
    """
    Create a graph from coordinates with k-nearest neighbors connectivity.
    
    Args:
        coords: Tensor of shape [N, 3] containing atom coordinates
        sequence: String of amino acid sequence
        aa_embedding: Tensor of shape [L, N_props] containing AAIndex embeddings
        k: Number of nearest neighbors for graph construction
        
    Returns:
        PyTorch Geometric Data object
    """
    # Calculate pairwise distances
    dist_matrix = torch.cdist(coords, coords)
    
    # Get k nearest neighbors for each atom
    _, indices = torch.topk(-dist_matrix, k=min(k, len(coords)), dim=1)
    
    # Create edge index (COO format)
    rows = torch.arange(len(coords)).view(-1, 1).repeat(1, indices.size(1))
    edge_index = torch.stack([rows.flatten(), indices.flatten()], dim=0)
    
    # Edge features: distances between connected atoms
    edge_attr = dist_matrix[edge_index[0], edge_index[1]].unsqueeze(1)
    
    # Node features: combine position and AAIndex properties
    # We need to map AAIndex properties (per residue) to atoms
    # Assuming coords are ordered as [res1_N, res1_CA, res1_C, res1_O, res2_N, ...]
    atoms_per_residue = 4  # N, CA, C, O
    
    # Expand aa_embedding to match atom count
    expanded_aa_embedding = aa_embedding.repeat_interleave(atoms_per_residue, dim=0)
    
    # If the lengths don't match exactly (e.g., due to missing atoms), truncate
    if expanded_aa_embedding.size(0) > coords.size(0):
        expanded_aa_embedding = expanded_aa_embedding[:coords.size(0)]
    
    # Combine coordinates and AAIndex properties as node features
    node_features = torch.cat([coords, expanded_aa_embedding], dim=1)
    
    return Data(
        x=node_features,
        edge_index=edge_index,
        edge_attr=edge_attr,
        sequence=sequence,
        num_nodes=len(coords)
    )

def build_graphs(processed_dir, output_dir, k=20):
    """Build graphs for all processed data files."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Process AMP and nonAMP directories
    for dataset_type in ['AMP', 'nonAMP']:
        input_dataset_dir = os.path.join(processed_dir, dataset_type)
        output_dataset_dir = os.path.join(output_dir, dataset_type)
        os.makedirs(output_dataset_dir, exist_ok=True)
        
        data_files = [f for f in os.listdir(input_dataset_dir) if f.endswith('.pt')]
        
        for data_file in tqdm(data_files, desc=f"Building graphs for {dataset_type}"):
            data_path = os.path.join(input_dataset_dir, data_file)
            
            # Load processed data
            data = torch.load(data_path)
            
            # Create graph
            graph = create_graph(
                data['coords'], 
                data['sequence'], 
                data['aa_embedding'], 
                k=k
            )
            
            # Add is_amp attribute
            graph.is_amp = data['is_amp']
            
            # Save graph
            base_name = os.path.splitext(data_file)[0]
            torch.save(graph, os.path.join(output_dataset_dir, f"{base_name}_graph.pt"))

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Build graphs from processed peptide data")
    parser.add_argument("--processed_dir", type=str, required=True, help="Directory containing processed data")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for graph data")
    parser.add_argument("--k", type=int, default=20, help="Number of nearest neighbors for graph construction")
    
    args = parser.parse_args()
    
    build_graphs(args.processed_dir, args.output_dir, args.k)
