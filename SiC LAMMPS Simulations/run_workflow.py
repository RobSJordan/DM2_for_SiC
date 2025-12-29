#!/usr/bin/env python3
import os
import subprocess
import argparse
import random
import ase.io
from pathlib import Path

def run_command(command, cwd, stdin_file=None):
    """Runs a command, streaming its output and checking for errors."""
    cmd_str = ' '.join(command)
    if stdin_file:
        cmd_str += f" < {stdin_file.name}"
    print(f"Executing in '{cwd}': {cmd_str}")

    stdin_stream = None
    if stdin_file:
        stdin_stream = open(stdin_file, 'r')

    process = subprocess.Popen(
        command,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        stdin=stdin_stream,
        text=True
    )
    
    stdout, stderr = process.communicate()
    if stdin_stream:
        stdin_stream.close()

    if process.returncode != 0:
        print("--- ERROR ---")
        print(f"STDOUT:\n{stdout}")
        print(f"STDERR:\n{stderr}")
        raise RuntimeError(f"Command failed with exit code {process.returncode}: {cmd_str}")
    
    # Print stdout to see the log messages from the executed command
    print(stdout)
    print("--- SUCCESS ---")

def get_next_run_id(base_dir):
    """
    Scans the base directory for existing 'run_XX' folders and determines
    the next sequential run ID.
    """
    run_dirs = list(base_dir.glob('run_*'))
    if not run_dirs:
        return "01"

    max_id = 0
    for run_dir in run_dirs:
        try:
            dir_id = int(run_dir.name.split('_')[1])
            if dir_id > max_id:
                max_id = dir_id
        except (ValueError, IndexError):
            # Ignore directories that don't match the 'run_XX' pattern
            continue
            
    next_id = max_id + 1
    return f"{next_id:02d}"

def main():
    """
    Main function to run the SiC melt-quench workflow. It automatically
    determines the next available run ID to avoid overwriting previous runs.
    """
    parser = argparse.ArgumentParser(description="Automate the SiC melt-quench simulation workflow.")
    parser.add_argument("--natoms", type=int, default=3000, help="Total number of atoms (default: 3000)")
    parser.add_argument("--rate", type=float, default=100.0, help="Cooling rate in K/ps (default: 100.0)")
    args = parser.parse_args()
    n_atoms = args.natoms
    rate = args.rate

    # --- Configuration ---
    script_dir = Path(__file__).parent.resolve()
    # LAMMPS command based on your notes
    lmp_command = "lmp -k on g 1 -sf kk -pk kokkos newton on neigh half".split()

    # --- Determine Run ID ---
    run_id = get_next_run_id(script_dir)
    print(f"INFO: Next available run ID is '{run_id}'. Starting new workflow with {n_atoms} atoms at {rate} K/ps.")

    # --- Directory and File Setup ---
    run_dir = script_dir / f"run_{run_id}_{n_atoms}_{rate}"
    run_dir.mkdir(exist_ok=True)
    print(f"INFO: Using run directory: {run_dir}")

    # Check for required source files
    for f in ["si.xyz", "c.xyz", "pack_SiC.inp", "SiC.vashishta"]:
        if not (script_dir / f).exists():
            raise FileNotFoundError(f"Source file '{f}' not found in '{script_dir}'.")

    # Define file paths *within* the run directory
    packmol_input_fname = "pack.inp"
    xyz_initial_fname = f"SiC_initial_{run_id}.xyz"
    data_initial_fname = f"SiC_initial_{run_id}.data"
    data_minimized_fname = f"SiC_minimized_{run_id}.data"
    data_glass_fname = f"SiC_glass_{run_id}.data"
    traj_glass_fname = f"trajectory_{run_id}.lammpstrj"

    # Paths to the NEW auto-ready LAMMPS scripts
    minimize_script_path = script_dir / "in.minimize_auto"
    melt_quench_script_path = script_dir / "in.SiC_melt_quench_auto"

    # --- STEP 1: Create initial structure with Packmol ---
    print("\n--- Step 1: Running Packmol ---")
    packmol_template_path = script_dir / "pack_SiC.inp"
    with open(packmol_template_path, 'r') as f:
        packmol_content = f.read()
    
    # Calculate atom counts and box size (maintaining density)
    # Original template: 3000 atoms (1500 each) in 33.2 Angstrom box
    n_si = n_atoms // 2
    n_c = n_atoms - n_si
    box_len = 33.2 * (n_atoms / 3000.0)**(1/3)

    # Comment out the non-standard 'check' command to avoid errors
    packmol_content = packmol_content.replace("check", "# check")

    # Update atom counts and box size in packmol input
    packmol_content = packmol_content.replace("number 1500", f"number {n_si}")
    packmol_content = packmol_content.replace("33.2", f"{box_len:.4f}")

    # Replace the placeholder output filename with the run-specific one
    packmol_content = packmol_content.replace("SiC_initial_01.xyz", xyz_initial_fname)
    
    # Add a random seed instruction for packmol to ensure varied initial states
    # and write the modified packmol input to the run directory.
    packmol_content = "seed -1\n" + packmol_content
    with open(run_dir / packmol_input_fname, 'w') as f:
        f.write(packmol_content)

    # Run packmol (assuming si.xyz and c.xyz are in the parent dir, packmol needs to find them)
    # We will symlink the source xyz files to the run dir so packmol finds them easily
    for f in ["si.xyz", "c.xyz"]:
        if not (run_dir / f).exists():
            os.symlink(script_dir / f, run_dir / f)

    run_command(["packmol"], cwd=run_dir, stdin_file=run_dir / packmol_input_fname)

    # --- STEP 2: Convert XYZ to LAMMPS Data with ASE ---
    print("\n--- Step 2: Converting to LAMMPS data file with ASE ---")
    atoms = ase.io.read(run_dir / xyz_initial_fname)
    atoms.set_cell([box_len, box_len, box_len])
    atoms.set_pbc(True)
    ase.io.write(run_dir / data_initial_fname, atoms, format='lammps-data', masses=True)

    # --- STEP 3: Energy Minimization ---
    print("\n--- Step 3: Running Energy Minimization with LAMMPS ---")
    # Symlink potential file
    if not (run_dir / "SiC.vashishta").exists():
        os.symlink(script_dir / "SiC.vashishta", run_dir / "SiC.vashishta")

    minimize_cmd = lmp_command + [
        "-v", "infile", data_initial_fname,
        "-v", "outfile", data_minimized_fname,
        "-in", str(minimize_script_path)
    ]
    run_command(minimize_cmd, cwd=run_dir)

    # --- STEP 4: Melt-Quench Simulation ---
    print("\n--- Step 4: Running Melt-Quench with LAMMPS ---")
    # Generate a random seed for velocity creation
    random_seed = random.randint(10000, 999999)

    # Calculate quench steps from rate.
    # Quench is from 5000K to 300K = 4700 K. Timestep is 0.001 ps.
    # duration_ps = 4700 K / (rate K/ps)
    # steps = duration_ps / timestep_ps = (4700 / rate) / 0.001
    quench_steps = int(4700000 / rate)
    print(f"INFO: Cooling rate {rate} K/ps corresponds to {quench_steps} steps.")
    
    melt_quench_cmd = lmp_command + [
        "-v", "infile", data_minimized_fname,
        "-v", "outfile", data_glass_fname,
        "-v", "restart_stem", f"SiC_glass_{run_id}",
        "-v", "dumpfile", traj_glass_fname,
        "-v", "seed", str(random_seed),
        "-v", "quench_steps", str(quench_steps),
        "-v", "is_resume", "0", # Tell LAMMPS this is a new run
        "-in", str(melt_quench_script_path)
    ]
    run_command(melt_quench_cmd, cwd=run_dir)

    print(f"\nWorkflow for run '{run_id}' completed successfully!")
    print(f"Output: {run_dir}/{data_glass_fname}")

if __name__ == "__main__":
    main()