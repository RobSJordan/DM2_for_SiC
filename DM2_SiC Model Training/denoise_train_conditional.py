#!/usr/bin/env python3
"""
Training script for a neural network denoiser model using PyTorch Geometric.
The model is designed to work with atomic structures and implements the NequIP architecture.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import ase.io
import warnings
from ase.neighborlist import primitive_neighbor_list
from torch_geometric.data import Data, Dataset
from sklearn.preprocessing import LabelEncoder
from torch_geometric.loader import DataLoader
from torch import nn
from functools import partial
from graphite.nn.basis import bessel
from graphite.nn.models.e3nn_nequip import NequIP_CoolingRateEmbed
from graphite.transforms import RattleParticles, DownselectEdges
from tqdm import tqdm
import os
import time

# Suppress specific warnings
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')
warnings.filterwarnings("ignore", category=UserWarning, module='torch.jit._check')

def ase_graph(data, cutoff):
    """Convert atomic structure to graph representation."""
    i, j, D = primitive_neighbor_list('ijD', cutoff=cutoff, pbc=data.pbc, 
                                    cell=data.cell, positions=data.pos.numpy(), 
                                    numbers=data.numbers)
    data.edge_index = torch.tensor(np.stack((i, j)), dtype=torch.long)
    data.edge_attr = torch.tensor(D, dtype=torch.float)
    return data

class PeriodicStructureDataset(Dataset):
    """Dataset class for periodic atomic structures with cooling rate info."""
    
    def __init__(self, atoms_list, cooling_rates, large_cutoff, duplicate=128):
        super().__init__(None, transform=None, pre_transform=None)
        
        self.structures = []
        self.duplicate = duplicate
        print(f"DEBUG: Initializing PeriodicStructureDataset with {len(atoms_list)} base structures.")
        for i, atoms in enumerate(atoms_list):
            print(f"DEBUG: Processing structure {i+1}/{len(atoms_list)}...")
            x = LabelEncoder().fit_transform(atoms.numbers)
            # Log transform the cooling rate 
            log_cooling_rate = np.log10(cooling_rates[i])
            
            data = Data(
                x            = torch.tensor(x).long(),
                pos          = torch.tensor(atoms.positions).float(),
                cell         = atoms.cell,
                pbc          = atoms.pbc,
                numbers      = atoms.numbers,
                cooling_rate = torch.tensor([log_cooling_rate]).float(),  # Add cooling rate as a graph attribute
            )
            data = ase_graph(data, large_cutoff)
            self.structures.append(data)

        print(f"DEBUG: Dataset creation complete with {len(self.structures)} base structures.")
        print(f"DEBUG: Effective dataset size (with on-the-fly duplication): {self.len()}")

    def len(self):
        return len(self.structures) * self.duplicate
    
    def get(self, idx):
        # Return a clone of the structure to avoid in-place modification of the original data
        original_idx = idx % len(self.structures)
        return self.structures[original_idx].clone()
    
class InitialEmbedding(nn.Module):
    """Initial embedding layer for the neural network."""
    
    def __init__(self, num_species, cutoff):
        super().__init__()
        self.embed_node_x = nn.Embedding(num_species, 8)
        self.embed_node_z = nn.Embedding(num_species, 8)
        self.embed_edge = partial(bessel, start=0.0, end=cutoff, num_basis=16)
    
    def forward(self, data):
        data.h_node_x = self.embed_node_x(data.x)
        data.h_node_z = self.embed_node_z(data.x)
        data.h_edge = self.embed_edge(data.edge_attr.norm(dim=-1))
        return data

def loss_fn(model, data):
    """Calculate MSE loss between predicted and actual displacement."""
    # Get only the first cooling rate for each batch item
    cooling_rates = data.cooling_rate
    
    # Since the same cooling rate applies to all nodes in a structure,
    # we only need to pass one value per batch
    pred_dx = model(data, cooling_rates)
    return torch.nn.functional.mse_loss(pred_dx, data.dx)

def train(loader, model, optimizer, device, rattle_particles, downselect_edges, PIN_MEMORY, epoch):
    """Training loop for one epoch."""
    start_time = time.time()
    model.train()
    total_loss = 0.0
    pbar = tqdm(loader, desc=f"Epoch {epoch} Training", leave=False)
    for data in pbar:
        optimizer.zero_grad(set_to_none=True)
        data = data.to(device, non_blocking=PIN_MEMORY)
        data = rattle_particles(data)
        data = downselect_edges(data)
        loss = loss_fn(model, data)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        pbar.set_postfix(loss=f'{loss.item():.4f}')

    epoch_time = time.time() - start_time
    return total_loss / len(loader), epoch_time

@torch.no_grad()
def test(loader, model, device, rattle_particles, downselect_edges, PIN_MEMORY, epoch):
    model.eval()
    total_loss = 0.0
    pbar = tqdm(loader, desc=f"Epoch {epoch} Validation", leave=False)
    for data in pbar:
        data = data.to(device, non_blocking=PIN_MEMORY)
        data = rattle_particles(data)
        data = downselect_edges(data)
        loss = loss_fn(model, data)  # This calls your modified loss_fn which includes cooling_rate
        total_loss += loss.item()
        pbar.set_postfix(loss=f'{loss.item():.4f}')
    return total_loss / len(loader)

def set_gpu(gpu_id):
    """Set PyTorch to use a specific GPU."""
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        print(f"DEBUG: CUDA is available. Found {device_count} GPU(s).")
        if gpu_id >= device_count:
            print(f"DEBUG: Invalid GPU ID {gpu_id} requested. {device_count} GPUs available. Using device 0 instead.")
            gpu_id = 0
        print(f"DEBUG: Setting GPU to device ID {gpu_id}")
        torch.cuda.set_device(gpu_id)
    else:
        print("DEBUG: CUDA not available. Using CPU.")

# Main function
def main():
    print("DEBUG: Script started.")
    # === Main configuration parameter ===
    NUM_SPECIES = 2         # important to check with your system or will casue error.
    gpu_id = 0              # adjustable based on your device.
    PIN_MEMORY  = True      # related to optimization for training, revert to False if you see any issues.
    NUM_WORKERS = 0         # related to optimization for training, revert to 1 if you see any issues.
    BATCH_SIZE  = 1         # adjust so that each minibatch fits in the (GPU) memory.
    LARGE_CUTOFF = 10       # recommend.
    CUTOFF      = 5         # recommend. May not the best value for every system and can affect model performace.
    LEARN_RATE  = 2e-4      # adjustable.
    NUM_UPDATES = 30000    # need to see convergence in loss plot (more or less).
    train_ratio = 0.9       # adjustable.
    sigma_max_value = 0.75  # adjust added noise in noisying process (recommend 0.75).
    path_save_loss_fig = './loss_figure.png'      # adjustable
    path_save_model = './model.pt'    # adjustable
    log_file_path = './training_log.txt'

    print("DEBUG: Configuration loaded.")

    # ==== Load data ====
    print("DEBUG: Loading data files...")
    try:
        in_0 = ase.io.read('./simu_data/SiC_glass_23_100.data',format='lammps-data')
        in_1 = ase.io.read('./simu_data/SiC_glass_24_100.data',format='lammps-data')
        in_2 = ase.io.read('./simu_data/SiC_glass_25_100.data',format='lammps-data')
        in_3 = ase.io.read('./simu_data/SiC_glass_26_100.data',format='lammps-data')
        in_4 = ase.io.read('./simu_data/SiC_glass_27_100.data',format='lammps-data')
        in_5 = ase.io.read('./simu_data/SiC_glass_29_10.data',format='lammps-data')
        in_6 = ase.io.read('./simu_data/SiC_glass_30_10.data',format='lammps-data')
        in_7 = ase.io.read('./simu_data/SiC_glass_31_10.data',format='lammps-data')
        in_8 = ase.io.read('./simu_data/SiC_glass_32_10.data',format='lammps-data')
        in_9 = ase.io.read('./simu_data/SiC_glass_33_10.data',format='lammps-data')
        in_10 = ase.io.read('./simu_data/SiC_glass_34_1.data',format='lammps-data')
        in_11 = ase.io.read('./simu_data/SiC_glass_35_1.data',format='lammps-data')
        in_12 = ase.io.read('./simu_data/SiC_glass_36_1.data',format='lammps-data')
        in_13 = ase.io.read('./simu_data/SiC_glass_37_1.data',format='lammps-data')
        in_14 = ase.io.read('./simu_data/SiC_glass_38_1.data',format='lammps-data')
        print("DEBUG: All data files loaded successfully.")
    except Exception as e:
        print(f"DEBUG: Error loading data files: {e}")
        raise e

    # Create a list of all structures
    ideal_atoms_list = [
        in_0, in_1, in_2, in_3, in_4, in_5,      
        in_6, in_7, in_8, in_9, in_10, in_11,     
        in_12, in_13, in_14
    ]

    # Create a corresponding list of conditions (i.e, cooling rate)
    cooling_rates = [
        100.0, 100.0, 100.0, 100.0, 100.0,  # 5 samples at 100K/ps
        10.0, 10.0, 10.0, 10.0, 10.0,       # 5 samples at 10K/ps
        1.0, 1.0, 1.0, 1.0, 1.0,             # 5 samples at 1K/ps
    ]
    # ====================

    # Initialize model with GaussianBasisEmbedding for cooling rate
    print("DEBUG: Initializing NequIP_CoolingRateEmbed model...")
    model = NequIP_CoolingRateEmbed(
        init_embed     = InitialEmbedding(num_species=NUM_SPECIES, cutoff=CUTOFF),
        irreps_node_x  = '8x0e',
        irreps_node_z  = '8x0e',
        irreps_hidden  = '64x0e + 32x1e',
        irreps_edge    = '4x0e + 4x1e + 2x2e',
        irreps_out     = '1x1e',
        num_convs      = 3,
        radial_neurons = [16, 64],
        num_neighbors  = 10,
    )
    print("DEBUG: Model initialized.")
    
    # Setup transformations
    print("DEBUG: Setting up transformations (RattleParticles, DownselectEdges)...")
    rattle_particles = RattleParticles(sigma_max=sigma_max_value)
    downselect_edges = DownselectEdges(cutoff=CUTOFF)
    
    # Prepare dataset
    print("DEBUG: Preparing PeriodicStructureDataset...")
    dataset = PeriodicStructureDataset(
        atoms_list=ideal_atoms_list, 
        cooling_rates=cooling_rates, 
        large_cutoff=LARGE_CUTOFF
    )

    # train valid split
    print("DEBUG: Splitting dataset into training and validation sets...")
    num_train = int(train_ratio * len(dataset))
    num_valid = len(dataset) - num_train
    ds_train, ds_valid = torch.utils.data.random_split(dataset, [num_train, num_valid])
    print(f"DEBUG: Train size: {len(ds_train)}, Validation size: {len(ds_valid)}")
    
    # Create dataloaders
    print("DEBUG: Creating DataLoaders...")
    train_loader = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True,
                            num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    valid_loader = DataLoader(ds_valid, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    
    # Training setup
    num_samples = len(dataset)
    num_epochs = int(NUM_UPDATES/(num_samples/BATCH_SIZE))
    print(f'DEBUG: {num_epochs} epochs needed to update the model {NUM_UPDATES} times.')
    
    print("DEBUG: Setting up optimizer and device...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARN_RATE)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_gpu(gpu_id)
    model.to(device)
    print(f"DEBUG: Model moved to {device}.")
    
    with open(log_file_path, 'w') as log_file:
        # Log configuration parameters
        log_file.write("=== Training Configuration ===\n")
        log_file.write(f"NUM_SPECIES: {NUM_SPECIES}\n")
        log_file.write(f"gpu_id: {gpu_id}\n")
        log_file.write(f"PIN_MEMORY: {PIN_MEMORY}\n")
        log_file.write(f"NUM_WORKERS: {NUM_WORKERS}\n")
        log_file.write(f"BATCH_SIZE: {BATCH_SIZE}\n")
        log_file.write(f"LARGE_CUTOFF: {LARGE_CUTOFF}\n")
        log_file.write(f"CUTOFF: {CUTOFF}\n")
        log_file.write(f"LEARN_RATE: {LEARN_RATE}\n")
        log_file.write(f"NUM_UPDATES: {NUM_UPDATES}\n")
        log_file.write(f"train_ratio: {train_ratio}\n")
        log_file.write(f"sigma_max_value: {sigma_max_value}\n")
        log_file.write(f"num_epochs: {num_epochs}\n\n")
        log_file.write("=== Training Log ===\n")
        log_file.write("Epoch,Train Loss,Valid Loss,Epoch Time (s),Avg it/s\n")

        # Training loop
        L_train, L_valid = [], []
        total_start_time = time.time()
        total_training_time = 0
        
        print("DEBUG: Starting training loop...")
        for epoch in range(num_epochs):
            train_loss, epoch_time = train(train_loader, model, optimizer, device,
                                         rattle_particles, downselect_edges, PIN_MEMORY, epoch + 1)
            valid_loss = test(valid_loader, model, device,
                             rattle_particles, downselect_edges, PIN_MEMORY, epoch + 1)
            total_training_time += epoch_time
            
            L_train.append(train_loss)
            L_valid.append(valid_loss)
            avg_its_per_sec = len(train_loader) / epoch_time if epoch_time > 0 else 0
            log_file.write(f"{epoch+1},{train_loss:.4f},{valid_loss:.4f},{epoch_time:.2f},{avg_its_per_sec:.2f}\n")
            print(f'Epoch [{epoch+1}/{num_epochs}] Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}, Time: {epoch_time:.2f}s, Avg it/s: {avg_its_per_sec:.2f}')
            
            if (epoch + 1) % 5 == 0:
                base, ext = os.path.splitext(path_save_model)
                checkpoint_path = f"{base}_epoch_{epoch+1}{ext}"
                print(f"\nDEBUG: Saving checkpoint to {checkpoint_path}...")
                torch.save(model, checkpoint_path)
                print("DEBUG: Checkpoint saved.")
    
    total_time = time.time() - total_start_time
    print(f'\nTraining completed:')
    print(f'Total training time: {total_time:.2f} seconds ({total_time/3600:.2f} hours)')
    print(f'Average time per epoch: {total_training_time/num_epochs:.2f} seconds')

    # Save model
    print(f"DEBUG: Saving model to {path_save_model}...")
    torch.save(model, path_save_model)
    print("DEBUG: Model saved.")

    # Plot results
    print(f"DEBUG: Plotting loss to {path_save_loss_fig}...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 3))
    ax1.plot(L_train, label='train')
    ax1.plot(L_valid, label='valid')
    ax1.set_ylabel('Loss')
    ax1.set_xlabel('Epochs')
    ax1.legend()
    
    ax2.semilogy(L_train, label='train')
    ax2.semilogy(L_valid, label='valid')
    ax2.set_xlabel('Epochs')
    ax2.legend()
    plt.show()
    plt.savefig(path_save_loss_fig, bbox_inches='tight', dpi=300)
    print("DEBUG: Plot saved. Script finished.")
    
    

if __name__ == '__main__':
    main()