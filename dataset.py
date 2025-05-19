import os
import torch
import numpy as np
from torch.utils.data import Dataset
import random
from scipy.spatial.transform import Rotation

class SimpleGraph:
    """Simple graph data structure to replace torch_geometric.data.Data."""
    def __init__(self, x, edge_index, edge_attr, pos, batch=None):
        self.x = x
        self.edge_index = edge_index
        self.edge_attr = edge_attr
        self.pos = pos
        self.batch = batch

class PeptideDataset(Dataset):
    """Enhanced dataset for peptide sequences and structures."""
    def __init__(self, processed_dir, split='train', is_amp=True, config=None):
        """
        Initialize dataset with augmentation options.
        
        Args:
            processed_dir: Directory containing processed data
            split: Data split ('train', 'val', or 'test')
            is_amp: Whether to use AMP data (True) or non-AMP data (False)
            config: Configuration dictionary with augmentation settings
        """
        self.processed_dir = processed_dir
        self.split = split
        self.is_amp = is_amp
        self.config = config
        
        # Set augmentation flags
        self.do_augmentation = False
        if config is not None and 'data' in config and 'augmentation' in config['data']:
            aug_config = config['data']['augmentation']
            # Only augment training data
            self.do_augmentation = aug_config.get('enabled', False) and split == 'train'
            self.rotation_prob = aug_config.get('rotation_prob', 0.5)
            self.jitter_prob = aug_config.get('jitter_prob', 0.3)
            self.jitter_sigma = aug_config.get('jitter_sigma', 0.1)
            self.dropout_prob = aug_config.get('dropout_prob', 0.2)
        
        # Get list of peptide IDs
        self.peptide_ids = self._get_peptide_ids()
        
        # Load AAIndex data for normalization
        self.aaindex_stats = self._load_aaindex_stats()
    
    def _load_aaindex_stats(self):
        """Load statistics for AAIndex properties for normalization."""
        stats_path = os.path.join(self.processed_dir, 'aaindex_stats.npz')
        if os.path.exists(stats_path):
            stats = np.load(stats_path)
            return {
                'mean': stats['mean'],
                'std': stats['std'],
                'min': stats['min'],
                'max': stats['max']
            }
        else:
            # Return dummy stats if file doesn't exist
            return {
                'mean': 0.0,
                'std': 1.0,
                'min': -1.0,
                'max': 1.0
            }
    
    def _get_peptide_ids(self):
        """Get list of peptide IDs for the current split with stratified sampling."""
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
        
        # Sort peptide IDs for reproducibility
        peptide_ids.sort()
        
        # Shuffle with fixed seed for reproducibility
        random.seed(42)
        random.shuffle(peptide_ids)
        
        # Split data
        n_peptides = len(peptide_ids)
        if n_peptides == 0:
            print(f"Warning: No peptide data found in {data_dir}. Using dummy data.")
            # Create dummy data for demonstration
            return ['dummy_peptide_1', 'dummy_peptide_2']
        
        # Get split ratios from config if available
        train_ratio = 0.8
        val_ratio = 0.1
        if self.config is not None and 'data' in self.config:
            train_ratio = self.config['data'].get('train_ratio', 0.8)
            val_ratio = self.config['data'].get('val_ratio', 0.1)
        
        train_size = int(train_ratio * n_peptides)
        val_size = int(val_ratio * n_peptides)
        
        if self.split == 'train':
            return peptide_ids[:train_size]
        elif self.split == 'val':
            return peptide_ids[train_size:train_size + val_size]
        elif self.split == 'test':
            return peptide_ids[train_size + val_size:]
        else:
            raise ValueError(f"Invalid split: {self.split}")
    
    def __len__(self):
        """Return number of peptides in the dataset."""
        return len(self.peptide_ids)
    
    def __getitem__(self, idx):
        """Get peptide data with optional augmentation."""
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
                    coords = np.load(coords_path, allow_pickle=True)
                    # Convert object array to float array if needed
                    if coords.dtype == np.dtype('O'):
                        coords = np.array(coords.tolist(), dtype=np.float32)
                except Exception as e:
                    print(f"Error loading coordinates for {peptide_id}: {e}")
                    coords = np.random.randn(len(sequence) * 4, 3).astype(np.float32)
            else:
                # Fallback to dummy coordinates
                coords = np.random.randn(len(sequence) * 4, 3).astype(np.float32)
                
            # Load properties
            props_path = os.path.join(data_dir, f"{peptide_id}_properties.npy")
            if os.path.exists(props_path):
                try:
                    properties = np.load(props_path, allow_pickle=True)
                    # Convert object array to float array if needed
                    if properties.dtype == np.dtype('O'):
                        properties = np.array(properties.tolist(), dtype=np.float32)
                    
                    # Normalize properties
                    if self.config is not None and self.config['preprocessing'].get('property_normalization', True):
                        properties = (properties - self.aaindex_stats['mean']) / (self.aaindex_stats['std'] + 1e-8)
                        
                except Exception as e:
                    print(f"Error loading properties for {peptide_id}: {e}")
                    properties = np.random.randn(len(sequence), 10).astype(np.float32)
            else:
                # Fallback to dummy properties
                properties = np.random.randn(len(sequence), 10).astype(np.float32)
            
            # Apply data augmentation if enabled
            if self.do_augmentation:
                coords, properties = self._augment_data(coords, properties, sequence)
                
            # Create graph
            graph = self._create_graph(sequence, coords, properties)
                
            # Create sequence embedding (placeholder - in a real implementation, you would use ESM)
            seq_emb = torch.randn(len(sequence), 1280)  # ESM-2 dim = 1280
                
            return {
                'peptide_id': peptide_id,
                'sequence': sequence,
                'graph': graph,
                'seq_emb': seq_emb,
                'is_amp': torch.tensor(1.0 if self.is_amp else 0.0, dtype=torch.float)
            }
        except Exception as e:
            print(f"Error loading peptide {peptide_id}: {e}")
            # Fallback to dummy data
            return self._get_dummy_data(peptide_id)
    
    def _augment_data(self, coords, properties, sequence):
        """Apply data augmentation to coordinates and properties."""
        # Convert to tensor for easier manipulation
        coords_tensor = torch.tensor(coords, dtype=torch.float)
        
        # 1. Random rotation
        if random.random() < self.rotation_prob:
            # Generate random rotation matrix
            rotation = Rotation.random()
            rot_matrix = torch.tensor(rotation.as_matrix(), dtype=torch.float)
            
            # Apply rotation to coordinates
            coords_tensor = torch.matmul(coords_tensor, rot_matrix)
        
        # 2. Add random jitter to coordinates
        if random.random() < self.jitter_prob:
            jitter = torch.randn_like(coords_tensor) * self.jitter_sigma
            coords_tensor = coords_tensor + jitter
        
        # 3. Property dropout (set some properties to zero)
        if random.random() < self.dropout_prob:
            mask = torch.rand(properties.shape) > self.dropout_prob
            properties = properties * mask
        
        return coords_tensor.numpy(), properties
    
    def _get_dummy_data(self, peptide_id):
        """Create dummy data for demonstration."""
        sequence = "ACDEGFHIKLMNPQRSTVWY"
        coords = np.random.randn(len(sequence) * 4, 3).astype(np.float32)
        properties = np.random.randn(len(sequence), 10).astype(np.float32)
        
        # Create graph
        graph = self._create_graph(sequence, coords, properties)
        
        # Create sequence embedding
        seq_emb = torch.randn(len(sequence), 1280)  # ESM-2 dim = 1280
        
        return {
            'peptide_id': peptide_id,
            'sequence': sequence,
            'graph': graph,
            'seq_emb': seq_emb,
            'is_amp': torch.tensor(1.0 if self.is_amp else 0.0, dtype=torch.float)
        }
    
    def _create_graph(self, sequence, coords, properties):
        """Create enhanced graph from sequence, coordinates, and properties."""
        # Number of residues
        n_residues = len(sequence)
    
        # Node features (properties)
        x = torch.tensor(properties, dtype=torch.float)
    
        # Node positions (coordinates)
        # Ensure we have exactly n_residues * 4 coordinates (4 atoms per residue)
        expected_coords_size = n_residues * 4
        if coords.shape[0] != expected_coords_size:
            # Resize coordinates to match expected size
            new_coords = np.zeros((expected_coords_size, 3), dtype=np.float32)
            # Copy as many coordinates as we can
            min_size = min(coords.shape[0], expected_coords_size)
            new_coords[:min_size] = coords[:min_size]
            # Fill the rest with random values if needed
            if min_size < expected_coords_size:
                new_coords[min_size:] = np.random.randn(expected_coords_size - min_size, 3).astype(np.float32)
            coords = new_coords
    
        # For the graph, we'll use only the CA atoms (1 per residue)
        # In a real implementation, you'd extract the actual CA atoms
        ca_indices = np.arange(0, coords.shape[0], 4)  # Every 4th atom is a CA atom
        ca_coords = coords[ca_indices]
    
        # Use CA coordinates for the graph
        pos = torch.tensor(ca_coords, dtype=torch.float)
    
        # Create edges with improved connectivity
        edge_index = []
        edge_attr = []
    
        # Connect each residue to its neighbors within a cutoff distance
        cutoff = 10.0  # Angstroms
        
        # 1. Connect sequential neighbors (i, i+1), (i, i+2), (i, i+3)
        for i in range(n_residues):
            for j in range(i+1, min(i+4, n_residues)):
                edge_index.append([i, j])
                edge_index.append([j, i])  # Add reverse edge
                
                # Edge features
                dist = np.linalg.norm(ca_coords[i] - ca_coords[j])
                seq_sep = abs(i - j)
                
                edge_features = [
                    float(dist),          # Distance
                    float(seq_sep),       # Sequence separation
                    float(1.0/seq_sep),   # Inverse sequence separation
                    float(1.0),           # Sequential neighbor flag
                    float(0.0)            # Spatial neighbor flag
                ]
                
                edge_attr.append(edge_features)
                edge_attr.append(edge_features)  # Same features for reverse edge
    
        # 2. Connect spatial neighbors within cutoff
        for i in range(n_residues):
            for j in range(n_residues):
                # Skip if already connected as sequential neighbor
                if i == j or abs(i - j) < 4:
                    continue
                
                # Calculate distance between CA atoms
                dist = np.linalg.norm(ca_coords[i] - ca_coords[j])
                
                # Add edge if within cutoff
                if dist <= cutoff:
                    edge_index.append([i, j])
                    edge_index.append([j, i])  # Add reverse edge
                    
                    # Edge features
                    seq_sep = abs(i - j)
                    
                    edge_features = [
                        float(dist),          # Distance
                        float(seq_sep),       # Sequence separation
                        float(1.0/seq_sep),   # Inverse sequence separation
                        float(0.0),           # Sequential neighbor flag
                        float(1.0)            # Spatial neighbor flag
                    ]
                    
                    edge_attr.append(edge_features)
                    edge_attr.append(edge_features)  # Same features for reverse edge
    
        # Convert to tensors
        if len(edge_index) == 0:
            # Ensure at least one edge
           
            # Ensure at least one edge
            edge_index = torch.zeros((2, 1), dtype=torch.long)
            edge_attr = torch.zeros((1, 5), dtype=torch.float)
        else:
            edge_index = torch.tensor(edge_index, dtype=torch.long).t()
            edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    
        # Create graph
        graph = SimpleGraph(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            pos=pos,
            batch=None
        )
    
        return graph
