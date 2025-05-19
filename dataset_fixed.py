import os
import torch
import numpy as np
from torch.utils.data import Dataset

class SimpleGraph:
    """Simple graph data structure to replace torch_geometric.data.Data."""
    def __init__(self, x, edge_index, edge_attr, pos, batch=None):
        self.x = x
        self.edge_index = edge_index
        self.edge_attr = edge_attr
        self.pos = pos
        self.batch = batch

class PeptideDataset(Dataset):
    """Dataset for peptide sequences and structures."""
    def __init__(self, processed_dir, split='train', is_amp=True):
        """
        Initialize dataset.
        
        Args:
            processed_dir: Directory containing processed data
            split: Data split ('train', 'val', or 'test')
            is_amp: Whether to use AMP data (True) or non-AMP data (False)
        """
        self.processed_dir = processed_dir
        self.split = split
        self.is_amp = is_amp
        
        # Get list of peptide IDs
        self.peptide_ids = self._get_peptide_ids()
    
    def _get_peptide_ids(self):
        """Get list of peptide IDs for the current split."""
        # Directory containing the data
        data_dir = os.path.join(self.processed_dir, 'AMP' if self.is_amp else 'nonAMP')
        
        # Check if directory exists
        if not os.path.exists(data_dir):
            print(f"Warning: Directory {data_dir} does not exist. Using dummy data.")
            # Create dummy data for demonstration
            return ['dummy_peptide_1', 'dummy_peptide_2']
        
        # Get all sequence files
        seq_files = [f for f in os.listdir(data_dir) if f.endswith('_seq.txt')]
        
        # Extract peptide IDs
        peptide_ids = [f.split('_seq.txt')[0] for f in seq_files]
        
        # Split data
        n_peptides = len(peptide_ids)
        if n_peptides == 0:
            print(f"Warning: No peptide data found in {data_dir}. Using dummy data.")
            # Create dummy data for demonstration
            return ['dummy_peptide_1', 'dummy_peptide_2']
            
        if self.split == 'train':
            return peptide_ids[:int(0.8 * n_peptides)]
        elif self.split == 'val':
            return peptide_ids[int(0.8 * n_peptides):int(0.9 * n_peptides)]
        elif self.split == 'test':
            return peptide_ids[int(0.9 * n_peptides):]
        else:
            raise ValueError(f"Invalid split: {self.split}")
    
    def __len__(self):
        """Return number of peptides in the dataset."""
        return len(self.peptide_ids)
    
    def __getitem__(self, idx):
        """Get peptide data."""
        peptide_id = self.peptide_ids[idx]
        data_dir = os.path.join(self.processed_dir, 'AMP' if self.is_amp else 'nonAMP')
        
        # Check if this is a dummy peptide
        if peptide_id.startswith('dummy_peptide'):
            return self._get_dummy_data(peptide_id)
        
        try:
            # Load sequence
            seq_path = os.path.join(data_dir, f"{peptide_id}_seq.txt")
            if os.path.exists(seq_path):
                with open(seq_path, 'r') as f:
                    sequence = f.read().strip()
            else:
                # Fallback to dummy sequence
                sequence = "ACDEGFHIKLMNPQRSTVWY"
                
            # Load coordinates
            coords_path = os.path.join(data_dir, f"{peptide_id}_coords.npy")
            if os.path.exists(coords_path):
                try:
                    # Load as numpy array and convert to torch tensor
                    coords = np.load(coords_path, allow_pickle=True)
                    # Convert object array to float array if needed
                    if coords.dtype == np.dtype('O'):
                        coords = np.array(coords.tolist(), dtype=np.float32)
                    # Convert to tensor
                    coords = torch.tensor(coords, dtype=torch.float32)
                except Exception as e:
                    print(f"Error loading coordinates for {peptide_id}: {e}")
                    coords = torch.randn(len(sequence) * 4, 3)
            else:
                # Fallback to dummy coordinates
                coords = torch.randn(len(sequence) * 4, 3)
                
            # Load properties
            props_path = os.path.join(data_dir, f"{peptide_id}_properties.npy")
            if os.path.exists(props_path):
                try:
                    # Load as numpy array and convert to torch tensor
                    properties = np.load(props_path, allow_pickle=True)
                    # Convert object array to float array if needed
                    if properties.dtype == np.dtype('O'):
                        properties = np.array(properties.tolist(), dtype=np.float32)
                    # Convert to tensor
                    properties = torch.tensor(properties, dtype=torch.float32)
                except Exception as e:
                    print(f"Error loading properties for {peptide_id}: {e}")
                    properties = torch.randn(len(sequence), 10)
            else:
                # Fallback to dummy properties
                properties = torch.randn(len(sequence), 10)
                
            # Create graph
            graph = self._create_graph(sequence, coords, properties)
                
            # Create sequence embedding (placeholder - in a real implementation, you would use ESM)
            seq_emb = torch.randn(len(sequence), 1280)  # ESM-2 dim = 1280
                
            return {
                'peptide_id': peptide_id,
                'sequence': sequence,
                'graph': graph,
                'seq_emb': seq_emb,
                'is_amp': torch.tensor([1.0]) if self.is_amp else torch.tensor([0.0])
            }
        except Exception as e:
            print(f"Error loading peptide {peptide_id}: {e}")
            # Fallback to dummy data
            return self._get_dummy_data(peptide_id)
    
    def _get_dummy_data(self, peptide_id):
        """Create dummy data for demonstration."""
        sequence = "ACDEGFHIKLMNPQRSTVWY"
        coords = torch.randn(len(sequence) * 4, 3)
        properties = torch.randn(len(sequence), 10)
        
        # Create graph
        graph = self._create_graph(sequence, coords, properties)
        
        # Create sequence embedding
        seq_emb = torch.randn(len(sequence), 1280)  # ESM-2 dim = 1280
        
        return {
            'peptide_id': peptide_id,
            'sequence': sequence,
            'graph': graph,
            'seq_emb': seq_emb,
            'is_amp': torch.tensor([1.0]) if self.is_amp else torch.tensor([0.0])
        }
    
    def _create_graph(self, sequence, coords, properties):
        """Create graph from sequence, coordinates, and properties."""
        # Number of residues
        n_residues = len(sequence)
    
        # Node features (properties)
        x = properties
    
        # Node positions (coordinates)
        # Ensure we have exactly n_residues * 4 coordinates (4 atoms per residue)
        expected_coords_size = n_residues * 4
        if coords.shape[0] != expected_coords_size:
            # Resize coordinates to match expected size
            new_coords = torch.zeros((expected_coords_size, 3))
            # Copy as many coordinates as we can
            min_size = min(coords.shape[0], expected_coords_size)
            new_coords[:min_size] = coords[:min_size]
            # Fill the rest with random values if needed
            if min_size < expected_coords_size:
                new_coords[min_size:] = torch.randn(expected_coords_size - min_size, 3)
            coords = new_coords
    
        # For the graph, we'll use only the CA atoms (1 per residue)
        # In a real implementation, you'd extract the actual CA atoms
        ca_indices = torch.arange(0, coords.shape[0], 4)  # Every 4th atom is a CA atom
        ca_coords = coords[ca_indices]
    
        # Use CA coordinates for the graph
        pos = ca_coords
    
        # Create edges (connect each residue to its neighbors)
        edge_index = []
        edge_attr = []
    
        # Connect each residue to its neighbors within a cutoff distance
        cutoff = 10.0  # Angstroms
    
        for i in range(n_residues):
            for j in range(n_residues):
                if i != j:
                    # Calculate distance between CA atoms
                    dist = torch.norm(ca_coords[i] - ca_coords[j])
                
                    # Add edge if within cutoff
                    if dist <= cutoff:
                        edge_index.append([i, j])
                    
                        # Edge features (distance, sequence separation)
                        edge_attr.append([
                            float(dist),  # Distance
                            float(abs(i - j)),  # Sequence separation
                            0.0,  # Placeholder for angle
                            0.0,  # Placeholder for dihedral
                            1.0   # Placeholder for edge type
                        ])
    
        # Convert to tensors
        if len(edge_index) == 0:
            # Ensure at least one edge
            edge_index = torch.zeros((2, 1), dtype=torch.long)
            edge_attr = torch.zeros((1, 5))
        else:
            edge_index = torch.tensor(edge_index, dtype=torch.long).t()
            edge_attr = torch.tensor(edge_attr)
    
        # Create graph
        graph = SimpleGraph(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            pos=pos,
            batch=None
        )
    
        return graph
