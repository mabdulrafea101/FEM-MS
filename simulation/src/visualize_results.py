import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from fem_core import BeamFEM

def plot_mode_shapes():
    print("Plotting mode shapes...")
    L = 5.0
    b = 0.3
    h = 0.5
    fc = 30
    
    # 1. Normal
    beam_norm = BeamFEM(L, b, h, fc, damage_type='none')
    res_norm = beam_norm.solve_eigenvalues()
    
    # 2. Corroded (20%)
    beam_corr = BeamFEM(L, b, h, fc, damage_type='corrosion', damage_params={'level': 20})
    res_corr = beam_corr.solve_eigenvalues()
    
    # 3. Cracked (Mid-span, 50% severity)
    beam_crack = BeamFEM(L, b, h, fc, damage_type='crack', damage_params={'location': L/2, 'severity': 0.5, 'width': 0.5})
    res_crack = beam_crack.solve_eigenvalues()
    
    x = res_norm.nodes
    
    plt.figure(figsize=(12, 6))
    
    # Normalize amplitudes
    m1_n = res_norm.mode_shapes[:, 0] / np.max(np.abs(res_norm.mode_shapes[:, 0]))
    m1_c = res_corr.mode_shapes[:, 0] / np.max(np.abs(res_corr.mode_shapes[:, 0]))
    m1_k = res_crack.mode_shapes[:, 0] / np.max(np.abs(res_crack.mode_shapes[:, 0]))
    
    plt.plot(x, m1_n, 'b-', linewidth=2, label=f'Normal (f={res_norm.frequencies[0]:.2f} Hz)')
    plt.plot(x, m1_c, 'r--', linewidth=2, label=f'Corroded 20% (f={res_corr.frequencies[0]:.2f} Hz)')
    plt.plot(x, m1_k, 'g-.', linewidth=2, label=f'Cracked Mid-span (f={res_crack.frequencies[0]:.2f} Hz)')
    
    plt.title("Mode Shape 1 Comparison: Normal vs Corroded vs Cracked")
    plt.xlabel("Length (m)")
    plt.ylabel("Normalized Amplitude")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    output_path = os.path.join(os.path.dirname(__file__), '../outputs/figures/mode_shape_comparison.png')
    plt.savefig(output_path)
    print(f"Saved {output_path}")
    plt.close()

def plot_dataset_distribution():
    print("Plotting dataset distribution...")
    csv_path = os.path.join(os.path.dirname(__file__), '../data/beam_vibration_dataset.csv')
    if not os.path.exists(csv_path):
        print("Dataset not found. Run generate_dataset.py first.")
        return
        
    df = pd.read_csv(csv_path)
    
    plt.figure(figsize=(10, 6))
    
    for dtype in df['Damage_Type'].unique():
        subset = df[df['Damage_Type'] == dtype]
        plt.hist(subset['Freq_Mode_1'], bins=30, alpha=0.5, label=dtype)
    
    plt.title("Distribution of Natural Frequencies by Damage Type")
    plt.xlabel("Frequency Mode 1 (Hz)")
    plt.ylabel("Count")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    output_path = os.path.join(os.path.dirname(__file__), '../outputs/figures/dataset_distribution.png')
    plt.savefig(output_path)
    print(f"Saved {output_path}")
    plt.close()

def plot_severity_impact():
    print("Plotting severity impact...")
    csv_path = os.path.join(os.path.dirname(__file__), '../data/beam_vibration_dataset.csv')
    if not os.path.exists(csv_path):
        return

    df = pd.read_csv(csv_path)
    
    # Filter for a specific geometry to see trends clearly, or just scatter all
    # Scatter all might be messy due to geometry variations.
    # Let's pick a narrow range of L, b, h to visualize trends
    
    # Approximate filtering
    mask = (df['Length'] > 4.9) & (df['Length'] < 5.1) & \
           (df['Width'] > 0.28) & (df['Width'] < 0.32)
           
    subset = df[mask]
    
    if len(subset) < 10:
        print("Not enough samples for scatter plot. Skipping.")
        return

    plt.figure(figsize=(10, 6))
    
    colors = {'none': 'blue', 'corrosion': 'red', 'crack': 'green', 'random': 'orange'}
    
    for dtype in subset['Damage_Type'].unique():
        data = subset[subset['Damage_Type'] == dtype]
        plt.scatter(data['Damage_Severity'], data['Freq_Mode_1'], 
                   c=colors.get(dtype, 'black'), label=dtype, alpha=0.7)
                   
    plt.title("Impact of Damage Severity on Frequency (L~5m, b~0.3m)")
    plt.xlabel("Damage Severity Metric (0-100)")
    plt.ylabel("Frequency Mode 1 (Hz)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    output_path = os.path.join(os.path.dirname(__file__), '../outputs/figures/severity_impact.png')
    plt.savefig(output_path)
    print(f"Saved {output_path}")
    plt.close()

if __name__ == "__main__":
    plot_mode_shapes()
    plot_dataset_distribution()
    plot_severity_impact()
