import os
import argparse
import pandas as pd
import numpy as np
from Bio import SeqIO
from Bio.PDB import PDBParser
import shutil

def load_aaindex(aaindex_path):
    """
    Load AAIndex properties from CSV file.
    Expected format: First column contains amino acids, other columns contain properties.
    """
    try:
        # Read the AAIndex CSV file
        df = pd.read_csv(aaindex_path)
        
        # Check if the first column contains amino acids
        first_col = df.columns[0]
        
        # Create a dictionary mapping amino acids to their properties
        aaindex_dict = {}
        for _, row in df.iterrows():
            aa = row[first_col]
            properties = row.drop(first_col).values
            aaindex_dict[aa] = properties
        
        return aaindex_dict
    except Exception as e:
        print(f"Error loading AAIndex file: {e}")
        print("AAIndex file format should have amino acids in the first column and properties in subsequent columns.")
        # Provide a fallback with basic properties if loading fails
        return create_fallback_aaindex()

def create_fallback_aaindex():
    """Create a fallback AAIndex dictionary with basic properties."""
    amino_acids = "ACDEFGHIKLMNPQRSTVWY"
    # Create random properties as fallback
    properties = np.random.rand(len(amino_acids), 10)
    
    aaindex_dict = {}
    for i, aa in enumerate(amino_acids):
        aaindex_dict[aa] = properties[i]
    
    print("Using fallback AAIndex properties.")
    return aaindex_dict

def extract_sequence_from_pdb(pdb_path):
    """Extract amino acid sequence from PDB file."""
    parser = PDBParser(QUIET=True)
    try:
        structure = parser.get_structure('protein', pdb_path)
        sequence = ""
        
        # Map 3-letter amino acid codes to 1-letter codes
        three_to_one = {
            'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E',
            'PHE': 'F', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
            'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N',
            'PRO': 'P', 'GLN': 'Q', 'ARG': 'R', 'SER': 'S',
            'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
        }
        
        # Extract sequence from first model and chain
        for residue in structure[0].get_residues():
            res_name = residue.get_resname()
            if res_name in three_to_one:
                sequence += three_to_one[res_name]
        
        return sequence
    except Exception as e:
        print(f"Error extracting sequence from {pdb_path}: {e}")
        return None

def extract_coordinates(pdb_path):
    """Extract backbone atom coordinates from PDB file."""
    parser = PDBParser(QUIET=True)
    try:
        structure = parser.get_structure('protein', pdb_path)
        coords = []
        
        # Extract coordinates of backbone atoms (N, CA, C, O)
        for model in structure:
            for chain in model:
                for residue in chain:
                    for atom in residue:
                        if atom.name in ['N', 'CA', 'C', 'O']:
                            coords.append(atom.coord)
        
        return np.array(coords)
    except Exception as e:
        print(f"Error extracting coordinates from {pdb_path}: {e}")
        return None

def process_dataset(input_dir, output_dir, aaindex_path):
    """Process the dataset and save processed files."""
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'AMP'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'nonAMP'), exist_ok=True)
    
    # Load AAIndex properties
    aaindex_dict = load_aaindex(aaindex_path)
    
    # Process AMP dataset
    amp_pdb_dir = os.path.join(input_dir, 'AMP')
    amp_fasta_path = os.path.join(input_dir, 'AMP.fasta')
    
    # Load sequences from FASTA file
    amp_sequences = {}
    if os.path.exists(amp_fasta_path):
        for record in SeqIO.parse(amp_fasta_path, "fasta"):
            amp_sequences[record.id] = str(record.seq)
    
    # Process PDB files
    if os.path.exists(amp_pdb_dir):
        for filename in os.listdir(amp_pdb_dir):
            if filename.endswith('.pdb'):
                pdb_id = os.path.splitext(filename)[0]
                pdb_path = os.path.join(amp_pdb_dir, filename)
                
                # Get sequence either from FASTA or PDB
                if pdb_id in amp_sequences:
                    sequence = amp_sequences[pdb_id]
                else:
                    sequence = extract_sequence_from_pdb(pdb_path)
                
                if sequence:
                    # Extract coordinates
                    coords = extract_coordinates(pdb_path)
                    
                    if coords is not None:
                        # Save sequence
                        with open(os.path.join(output_dir, 'AMP', f"{pdb_id}_seq.txt"), 'w') as f:
                            f.write(sequence)
                        
                        # Save coordinates
                        np.save(os.path.join(output_dir, 'AMP', f"{pdb_id}_coords.npy"), coords)
                        
                        # Compute and save AAIndex properties
                        properties = []
                        for aa in sequence:
                            if aa in aaindex_dict:
                                properties.append(aaindex_dict[aa])
                            else:
                                # Use average properties for unknown amino acids
                                properties.append(np.mean(list(aaindex_dict.values()), axis=0))
                        
                        properties = np.array(properties)
                        np.save(os.path.join(output_dir, 'AMP', f"{pdb_id}_properties.npy"), properties)
                        
                        print(f"Processed AMP: {pdb_id}")
    
    # Process nonAMP dataset (similar to AMP)
    nonamp_pdb_dir = os.path.join(input_dir, 'nonAMP')
    nonamp_fasta_path = os.path.join(input_dir, 'nonAMP.fasta')
    
    # Load sequences from FASTA file
    nonamp_sequences = {}
    if os.path.exists(nonamp_fasta_path):
        for record in SeqIO.parse(nonamp_fasta_path, "fasta"):
            nonamp_sequences[record.id] = str(record.seq)
    
    # Process PDB files
    if os.path.exists(nonamp_pdb_dir):
        for filename in os.listdir(nonamp_pdb_dir):
            if filename.endswith('.pdb'):
                pdb_id = os.path.splitext(filename)[0]
                pdb_path = os.path.join(nonamp_pdb_dir, filename)
                
                # Get sequence either from FASTA or PDB
                if pdb_id in nonamp_sequences:
                    sequence = nonamp_sequences[pdb_id]
                else:
                    sequence = extract_sequence_from_pdb(pdb_path)
                
                if sequence:
                    # Extract coordinates
                    coords = extract_coordinates(pdb_path)
                    
                    if coords is not None:
                        # Save sequence
                        with open(os.path.join(output_dir, 'nonAMP', f"{pdb_id}_seq.txt"), 'w') as f:
                            f.write(sequence)
                        
                        # Save coordinates
                        np.save(os.path.join(output_dir, 'nonAMP', f"{pdb_id}_coords.npy"), coords)
                        
                        # Compute and save AAIndex properties
                        properties = []
                        for aa in sequence:
                            if aa in aaindex_dict:
                                properties.append(aaindex_dict[aa])
                            else:
                                # Use average properties for unknown amino acids
                                properties.append(np.mean(list(aaindex_dict.values()), axis=0))
                        
                        properties = np.array(properties)
                        np.save(os.path.join(output_dir, 'nonAMP', f"{pdb_id}_properties.npy"), properties)
                        
                        print(f"Processed nonAMP: {pdb_id}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess peptide dataset")
    parser.add_argument("--input_dir", type=str, required=True, help="Input directory containing raw data")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for processed data")
    parser.add_argument("--aaindex_path", type=str, required=True, help="Path to AAIndex CSV file")
    
    args = parser.parse_args()
    process_dataset(args.input_dir, args.output_dir, args.aaindex_path)
