import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def load_unified_data(filepath):
    if filepath.endswith('.csv'):
        data = np.loadtxt(filepath, delimiter=',', skiprows=1)
    elif filepath.endswith('.bin'):
        data = np.fromfile(filepath, dtype=np.float32)
        data = data.reshape(-1, 7)
    else:
        raise ValueError("Unsupported file format")
    
    positions = data[:, 0:3]
    normals = data[:, 3:6]
    sk = data[:, 6]
    
    return positions, normals, sk

def visualize_particles(positions, normals, sk, title="Particle Visualization"):
    fig = plt.figure(figsize=(14, 6))
    
    ax1 = fig.add_subplot(121, projection='3d')
    scatter = ax1.scatter(positions[:, 0], positions[:, 1], positions[:, 2], 
                          c=sk, cmap='viridis', s=5, alpha=0.6)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title(f'{title}\nParticle Positions (colored by sk)')
    plt.colorbar(scatter, ax=ax1, label='sk value')
    
    ax2 = fig.add_subplot(122, projection='3d')
    sample_idx = np.random.choice(len(positions), min(500, len(positions)), replace=False)
    sample_pos = positions[sample_idx]
    sample_norm = normals[sample_idx]
    
    ax2.scatter(sample_pos[:, 0], sample_pos[:, 1], sample_pos[:, 2], 
                c='blue', s=10, alpha=0.5, label='Particles')
    
    scale = 0.15
    ax2.quiver(sample_pos[:, 0], sample_pos[:, 1], sample_pos[:, 2],
               sample_norm[:, 0] * scale, sample_norm[:, 1] * scale, sample_norm[:, 2] * scale,
               color='red', alpha=0.7, arrow_length_ratio=0.3)
    
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.set_title('Normal Vectors (sample)')
    
    plt.tight_layout()
    plt.savefig('particle_visualization.png', dpi=150)
    plt.show()

def print_statistics(positions, normals, sk):
    print("\n=== Data Statistics ===")
    print(f"Number of particles: {len(positions)}")
    
    min_pos = np.min(positions, axis=0)
    max_pos = np.max(positions, axis=0)
    center = np.mean(positions, axis=0)
    
    print(f"\nBounding Box:")
    print(f"  Min: ({min_pos[0]:.4f}, {min_pos[1]:.4f}, {min_pos[2]:.4f})")
    print(f"  Max: ({max_pos[0]:.4f}, {max_pos[1]:.4f}, {max_pos[2]:.4f})")
    print(f"  Size: ({max_pos[0]-min_pos[0]:.4f}, {max_pos[1]-min_pos[1]:.4f}, {max_pos[2]-min_pos[2]:.4f})")
    print(f"  Center: ({center[0]:.4f}, {center[1]:.4f}, {center[2]:.4f})")
    
    print(f"\nSurface Area (sum of sk): {np.sum(sk):.4f}")
    print(f"Average sk: {np.mean(sk):.6f}")
    print(f"Min sk: {np.min(sk):.6f}")
    print(f"Max sk: {np.max(sk):.6f}")
    
    norm_lengths = np.linalg.norm(normals, axis=1)
    print(f"\nNormal vector lengths:")
    print(f"  Mean: {np.mean(norm_lengths):.6f}")
    print(f"  Min: {np.min(norm_lengths):.6f}")
    print(f"  Max: {np.max(norm_lengths):.6f}")
    
    return min_pos, max_pos, center

def check_lbm_compatibility(positions, Nx=128, Ny=128, Nz=128):
    print("\n=== LBM Domain Compatibility Check ===")
    print(f"LBM Domain: Nx={Nx}, Ny={Ny}, Nz={Nz}")
    
    min_pos = np.min(positions, axis=0)
    max_pos = np.max(positions, axis=0)
    
    in_domain = True
    for i, (name, N) in enumerate([('X', Nx), ('Y', Ny), ('Z', Nz)]):
        if min_pos[i] < 0 or max_pos[i] >= N:
            print(f"WARNING: {name} range [{min_pos[i]:.2f}, {max_pos[i]:.2f}] is outside domain [0, {N})")
            in_domain = False
        else:
            print(f"{name} range [{min_pos[i]:.2f}, {max_pos[i]:.2f}] is within domain [0, {N})")
    
    if in_domain:
        print("\nAll particles are within the LBM domain.")
    else:
        print("\nWARNING: Some particles are outside the LBM domain!")
        print("You may need to translate the particle positions.")
    
    return in_domain

def apply_scale(positions, sk, scale=20.0):
    print(f"\n=== Applying Scale Factor: {scale} ===")
    scaled_positions = positions * scale
    scaled_sk = sk * (scale * scale)
    
    min_pos = np.min(scaled_positions, axis=0)
    max_pos = np.max(scaled_positions, axis=0)
    print(f"Scaled Bounding Box: ({min_pos[0]:.2f}, {min_pos[1]:.2f}, {min_pos[2]:.2f}) to ({max_pos[0]:.2f}, {max_pos[1]:.2f}, {max_pos[2]:.2f})")
    print(f"Scaled Total Surface Area: {np.sum(scaled_sk):.4f}")
    
    return scaled_positions, scaled_sk

if __name__ == "__main__":
    import os
    
    csv_path = "models/box_unified_data.csv"
    bin_path = "models/box_unified_data.bin"
    
    if os.path.exists(csv_path):
        print(f"Loading from CSV: {csv_path}")
        positions, normals, sk = load_unified_data(csv_path)
    elif os.path.exists(bin_path):
        print(f"Loading from BIN: {bin_path}")
        positions, normals, sk = load_unified_data(bin_path)
    else:
        print("ERROR: No data file found!")
        print("Please ensure box_unified_data.csv or box_unified_data.bin exists in the current directory.")
        exit(1)
    
    print_statistics(positions, normals, sk)
    check_lbm_compatibility(positions)
    
    SCALE = 20.0
    scaled_positions, scaled_sk = apply_scale(positions, sk, SCALE)
    check_lbm_compatibility(scaled_positions)
    
    visualize_particles(scaled_positions, normals, scaled_sk, f"Box Model Particles (scale={SCALE})")
