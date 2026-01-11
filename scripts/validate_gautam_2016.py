"""
Validation Script: Gautam et al. (2016) Fixed-Fixed Beam
========================================================

This script validates the FEM implementation against Gautam et al. (2016):
"Modal Analysis of Beam Through Analytically and FEM"
International Conference on Innovative Trends in Science, Engineering and Management
ICITSEM-16, New Delhi, India, pp. 375-383.

Target Results (Table 5 from paper - Fixed-Fixed Beam):
- Mode 1: Analytical = 132.04 Hz, ANSYS FEM = 132.04 Hz
- Mode 2: Analytical = 357.30 Hz, ANSYS FEM = 357.80 Hz
- Mode 3: Analytical = 687.72 Hz, ANSYS FEM = 687.19 Hz

Beam Parameters (Table 4):
- Material: Mild Steel
- E = 20.5 × 10¹⁰ N/m² (205 GPa)
- ρ = 7830 kg/m³
- ν = 0.33
- L = 2.0 m, B = 0.3 m, H = 0.1 m

Author: Generated for MS Thesis Validation
Date: January 2025
"""

import numpy as np
from scipy.linalg import eigh
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import PolyCollection
from pathlib import Path

# ============================================================================
# Gautam et al. (2016) Beam Parameters - Table 4
# ============================================================================

# Geometric properties
L = 2.0            # Beam length (m)
b = 0.3            # Width (m)
h = 0.1            # Height/Depth (m)
A = b * h          # Cross-sectional area (m²) = 0.03 m²
I = b * h**3 / 12  # Moment of inertia (m⁴)

# Material properties (Mild Steel)
E = 20.5e10        # Elastic modulus (Pa) = 205 GPa
rho = 7830         # Density (kg/m³)
nu = 0.33          # Poisson's ratio (not used in EBT)

# FEM discretization
n_elements = 20    # Number of elements (standard for convergence)
n_nodes = n_elements + 1
Le = L / n_elements  # Element length

# DOF configuration
dof_per_node = 2   # Transverse displacement + rotation
total_dof = n_nodes * dof_per_node

# ============================================================================
# Target frequencies from Gautam et al. (2016) Table 5 - Fixed-Fixed
# ============================================================================

TARGET_ANALYTICAL = {'f1': 132.04, 'f2': 357.3022, 'f3': 687.72}
TARGET_ANSYS = {'f1': 132.04, 'f2': 357.80, 'f3': 687.19}

# (βL)² values for fixed-fixed beam (Table 3)
BETA_L_SQUARED = {1: 4.730041, 2: 7.853205, 3: 10.995608}

# ============================================================================
# Theoretical Frequency Calculation
# ============================================================================

def calculate_theoretical_frequencies(E, I, rho, A, L, n_modes=3):
    """
    Calculate theoretical natural frequencies using Euler-Bernoulli theory.

    From Gautam et al. (2016) Eq. 9:
    ω = (βL)² √(EI/ρAL⁴)
    f = ω / (2π)

    For fixed-fixed beam, the frequency equation is:
    f_n = (λ_n² / 2πL²) √(EI/ρA)

    where λ_n = (βL)_n values for fixed-fixed beam (Table 3):
    Mode 1: 4.730041
    Mode 2: 7.853205
    Mode 3: 10.995608
    """
    # λ = βL values (NOT squared!)
    lambda_n = [4.730041, 7.853205, 10.995608, 14.137165, 17.278760]

    frequencies = []
    for i in range(n_modes):
        lam = lambda_n[i]
        # f = (λ²/2πL²) √(EI/ρA)
        f_n = (lam**2 / (2 * np.pi * L**2)) * np.sqrt(E * I / (rho * A))
        frequencies.append(f_n)

    return frequencies


def calculate_mode_shape(beta_L, x_norm):
    """
    Calculate normalized mode shape for fixed-fixed beam.

    From beam vibration theory:
    φ(x) = cosh(βx) - cos(βx) - σ[sinh(βx) - sin(βx)]

    where σ = [cosh(βL) - cos(βL)] / [sinh(βL) - sin(βL)]
    """
    # For fixed-fixed, the mode shapes follow this pattern
    beta = beta_L / 1.0  # Since x_norm is already normalized to L
    x = x_norm

    # Calculate sigma
    cosh_bL = np.cosh(beta_L)
    cos_bL = np.cos(beta_L)
    sinh_bL = np.sinh(beta_L)
    sin_bL = np.sin(beta_L)

    sigma = (cosh_bL - cos_bL) / (sinh_bL - sin_bL)

    # Mode shape
    phi = (np.cosh(beta_L * x) - np.cos(beta_L * x) -
           sigma * (np.sinh(beta_L * x) - np.sin(beta_L * x)))

    # Normalize to unit maximum
    phi = phi / np.max(np.abs(phi))

    return phi


# ============================================================================
# FEM Implementation
# ============================================================================

def element_stiffness_matrix(E, I, Le):
    """
    Generate 4x4 Euler-Bernoulli element stiffness matrix.

    DOF order: [v1, θ1, v2, θ2]
    where v = transverse displacement, θ = rotation
    """
    coeff = E * I / Le**3
    ke = coeff * np.array([
        [12,      6*Le,    -12,      6*Le   ],
        [6*Le,    4*Le**2, -6*Le,    2*Le**2],
        [-12,    -6*Le,     12,     -6*Le   ],
        [6*Le,    2*Le**2, -6*Le,    4*Le**2]
    ])
    return ke


def element_mass_matrix(rho, A, Le):
    """
    Generate 4x4 consistent mass matrix for Euler-Bernoulli beam element.
    """
    coeff = rho * A * Le / 420
    me = coeff * np.array([
        [156,     22*Le,    54,     -13*Le  ],
        [22*Le,   4*Le**2,  13*Le,  -3*Le**2],
        [54,      13*Le,    156,    -22*Le  ],
        [-13*Le, -3*Le**2, -22*Le,   4*Le**2]
    ])
    return me


def assemble_global_matrices(n_elements, E, I, rho, A, Le):
    """
    Assemble global stiffness and mass matrices.
    """
    n_dof = (n_elements + 1) * 2
    K = np.zeros((n_dof, n_dof))
    M = np.zeros((n_dof, n_dof))

    for elem in range(n_elements):
        ke = element_stiffness_matrix(E, I, Le)
        me = element_mass_matrix(rho, A, Le)

        # Global DOF indices
        dof_start = elem * 2
        dof_indices = [dof_start, dof_start+1, dof_start+2, dof_start+3]

        for i in range(4):
            for j in range(4):
                K[dof_indices[i], dof_indices[j]] += ke[i, j]
                M[dof_indices[i], dof_indices[j]] += me[i, j]

    return K, M


def apply_boundary_conditions(K, M, fixed_dof):
    """
    Apply fixed boundary conditions by eliminating DOFs.
    """
    n_total = K.shape[0]
    free_dof = [i for i in range(n_total) if i not in fixed_dof]

    K_reduced = K[np.ix_(free_dof, free_dof)]
    M_reduced = M[np.ix_(free_dof, free_dof)]

    return K_reduced, M_reduced, free_dof


def solve_eigenvalue_problem(K, M, n_modes=3):
    """
    Solve generalized eigenvalue problem: Kφ = ω²Mφ
    """
    eigenvalues, eigenvectors = eigh(K, M)

    eigenvalues = eigenvalues[:n_modes]
    eigenvectors = eigenvectors[:, :n_modes]

    frequencies = np.sqrt(eigenvalues) / (2 * np.pi)

    return frequencies, eigenvectors


def run_fem_analysis(E, I, rho, A, L, n_elements, n_modes=3):
    """
    Complete FEM analysis for a fixed-fixed beam.
    """
    Le = L / n_elements

    K, M = assemble_global_matrices(n_elements, E, I, rho, A, Le)

    # Fixed-fixed boundary conditions
    n_dof = (n_elements + 1) * 2
    fixed_dof = [0, 1, n_dof-2, n_dof-1]

    K_red, M_red, free_dof = apply_boundary_conditions(K, M, fixed_dof)

    frequencies, mode_shapes = solve_eigenvalue_problem(K_red, M_red, n_modes)

    return frequencies, mode_shapes, free_dof


# ============================================================================
# Validation Functions
# ============================================================================

def validate_beam():
    """Validate FEM against Gautam et al. (2016) results."""
    print("=" * 70)
    print("VALIDATION: Gautam et al. (2016) - Fixed-Fixed Steel Beam")
    print("ANSYS 14.5 vs Python FEM vs Analytical (Euler-Bernoulli)")
    print("=" * 70)

    print(f"\nBeam Parameters (Table 4):")
    print(f"   Material: Mild Steel")
    print(f"   E = {E/1e9:.1f} GPa")
    print(f"   ρ = {rho} kg/m³")
    print(f"   L = {L} m, b = {b} m, h = {h} m")
    print(f"   A = {A} m², I = {I:.6e} m⁴")

    # Run FEM analysis
    print(f"\nRunning FEM analysis with {n_elements} elements...")
    frequencies, mode_shapes, free_dof = run_fem_analysis(E, I, rho, A, L, n_elements, n_modes=3)

    # Calculate theoretical
    theoretical = calculate_theoretical_frequencies(E, I, rho, A, L, n_modes=3)

    # Display results
    print("\n" + "=" * 70)
    print("TABLE 4.2: Three-Way Validation for Fixed-Fixed Beam")
    print("=" * 70)
    print(f"{'Mode':<8} {'Gautam ANSYS (Hz)':<20} {'Our Python FEM (Hz)':<22} {'Theoretical EBT (Hz)':<22} {'Error vs ANSYS':<15}")
    print("-" * 90)

    errors = []
    for i in range(3):
        mode_key = f'f{i+1}'
        ansys = TARGET_ANSYS[mode_key]
        analytical = TARGET_ANALYTICAL[mode_key]
        our_fem = frequencies[i]
        theory = theoretical[i]
        error_ansys = abs(our_fem - ansys) / ansys * 100
        errors.append(error_ansys)
        print(f"{i+1:<8} {ansys:<20.2f} {our_fem:<22.2f} {theory:<22.2f} {error_ansys:<14.3f}%")

    print("-" * 90)
    print(f"\nMaximum error vs ANSYS: {max(errors):.3f}%")
    print(f"Target: ≤ 1.0%")
    print(f"Status: {'PASSED ✓' if max(errors) < 1.0 else 'CHECK - Within acceptable range' if max(errors) < 5.0 else 'NEEDS REVIEW'}")

    return frequencies, mode_shapes, free_dof, theoretical


# ============================================================================
# Visualization Functions
# ============================================================================

def plot_fem_model():
    """
    Create FE model diagram for Gautam et al. (2016) beam.
    """
    fig, ax = plt.subplots(figsize=(14, 5))

    # Beam dimensions for visualization
    beam_length = L
    beam_height = 0.12
    n_elem = n_elements
    elem_length = beam_length / n_elem

    # Draw beam elements
    colors = plt.cm.Blues(np.linspace(0.3, 0.7, n_elem))

    for i in range(n_elem):
        x_start = i * elem_length
        rect = mpatches.FancyBboxPatch(
            (x_start, -beam_height/2),
            elem_length - 0.002,
            beam_height,
            boxstyle="round,pad=0.003",
            facecolor=colors[i],
            edgecolor='black',
            linewidth=1.0
        )
        ax.add_patch(rect)

        # Element number (only show every 5th for clarity with 20 elements)
        if (i + 1) % 5 == 0 or i == 0:
            ax.text(x_start + elem_length/2, 0, str(i+1),
                    ha='center', va='center', fontsize=8, fontweight='bold')

    # Draw nodes at ends and midpoint
    key_nodes = [0, n_elem//4, n_elem//2, 3*n_elem//4, n_elem]
    for idx in key_nodes:
        x = idx * elem_length
        ax.plot(x, beam_height/2, 'ko', markersize=6)
        ax.plot(x, -beam_height/2, 'ko', markersize=6)
        ax.text(x, beam_height/2 + 0.04, str(idx+1),
                ha='center', va='bottom', fontsize=8)

    # Fixed supports
    support_width = 0.06
    support_height = 0.10

    # Left support
    left_support = plt.Polygon([
        (-support_width, -beam_height/2 - support_height),
        (0, -beam_height/2),
        (-support_width, beam_height/2 + support_height)
    ], facecolor='gray', edgecolor='black')
    ax.add_patch(left_support)

    # Right support
    right_support = plt.Polygon([
        (beam_length + support_width, -beam_height/2 - support_height),
        (beam_length, -beam_height/2),
        (beam_length + support_width, beam_height/2 + support_height)
    ], facecolor='gray', edgecolor='black')
    ax.add_patch(right_support)

    # Ground hatching
    for x in np.arange(-support_width - 0.01, -0.01, 0.02):
        ax.plot([x, x - 0.02], [-beam_height/2 - support_height - 0.01,
                -beam_height/2 - support_height - 0.04], 'k-', lw=0.8)
    for x in np.arange(beam_length + 0.01, beam_length + support_width + 0.01, 0.02):
        ax.plot([x, x + 0.02], [-beam_height/2 - support_height - 0.01,
                -beam_height/2 - support_height - 0.04], 'k-', lw=0.8)

    # Dimension annotations
    ax.annotate('', xy=(beam_length, -0.28), xytext=(0, -0.28),
                arrowprops=dict(arrowstyle='<->', color='black', lw=1.5))
    ax.text(beam_length/2, -0.34, f'L = {L} m', ha='center', fontsize=11)

    # Cross-section annotation
    ax.text(beam_length + 0.15, 0.15, f'b × h = {b} × {h} m', fontsize=10, ha='left')
    ax.text(beam_length + 0.15, 0.08, f'E = {E/1e9:.0f} GPa', fontsize=10, ha='left')
    ax.text(beam_length + 0.15, 0.01, f'ρ = {rho} kg/m³', fontsize=10, ha='left')

    # BC indicators
    ax.annotate('Fixed BC\n(v=0, θ=0)', xy=(-0.08, 0), xytext=(-0.35, 0.22),
                fontsize=9, ha='center',
                arrowprops=dict(arrowstyle='->', color='red', lw=1.2))
    ax.annotate('Fixed BC\n(v=0, θ=0)', xy=(beam_length + 0.08, 0), xytext=(beam_length + 0.35, 0.22),
                fontsize=9, ha='center',
                arrowprops=dict(arrowstyle='->', color='red', lw=1.2))

    ax.set_xlim(-0.5, beam_length + 0.6)
    ax.set_ylim(-0.45, 0.35)
    ax.set_aspect('equal')
    ax.axis('off')

    ax.set_title(f'FE Model of Fixed-Fixed Beam ({n_elements} Euler-Bernoulli Elements)\n'
                 'Based on Gautam et al. (2016)', fontsize=12, fontweight='bold')

    plt.tight_layout()
    return fig


def plot_validation_comparison(frequencies, theoretical):
    """
    Create bar chart comparing Gautam ANSYS, Our FEM, and Theoretical frequencies.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    modes = ['Mode 1', 'Mode 2', 'Mode 3']
    ansys_freqs = [TARGET_ANSYS['f1'], TARGET_ANSYS['f2'], TARGET_ANSYS['f3']]
    our_freqs = list(frequencies[:3])
    theory_freqs = theoretical[:3]

    x = np.arange(len(modes))
    width = 0.25

    bars1 = ax.bar(x - width, ansys_freqs, width, label='Gautam et al. (2016) ANSYS',
                   color='#2ecc71', edgecolor='black', linewidth=1.2)
    bars2 = ax.bar(x, our_freqs, width, label='Our Python FEM',
                   color='#3498db', edgecolor='black', linewidth=1.2)
    bars3 = ax.bar(x + width, theory_freqs, width, label='Theoretical EBT',
                   color='#e74c3c', edgecolor='black', linewidth=1.2)

    ax.set_ylabel('Natural Frequency (Hz)', fontsize=12)
    ax.set_xlabel('Vibration Mode', fontsize=12)
    ax.set_title('Three-Way Validation: Fixed-Fixed Steel Beam\nGautam et al. (2016) vs Python FEM vs Analytical',
                 fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(modes)
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    # Add value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Add error annotations
    for i, (ansys, our) in enumerate(zip(ansys_freqs, our_freqs)):
        error = abs(our - ansys) / ansys * 100
        ax.annotate(f'Error: {error:.2f}%',
                   xy=(i, max(ansys, our) + 30),
                   ha='center', fontsize=9, color='darkgreen')

    plt.tight_layout()
    return fig


def plot_mode_shapes(frequencies, mode_shapes, free_dof):
    """
    Create ANSYS-style mode shape visualization using actual FEM results.
    Shows the mode shapes computed from our Python FEM simulation.
    Similar to Gautam et al. (2016) Figure 7.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Get our computed frequencies
    freq_values = frequencies[:3]

    # Reconstruct full mode shapes including fixed DOFs
    n_total_dof = (n_elements + 1) * 2  # Total DOFs before BC application

    for idx in range(3):
        ax = axes[idx]
        freq = freq_values[idx]

        # Get mode shape from reduced system
        mode_reduced = mode_shapes[:, idx]

        # Reconstruct full mode shape vector (with zeros at fixed DOFs)
        mode_full = np.zeros(n_total_dof)
        for i, dof_idx in enumerate(free_dof):
            mode_full[dof_idx] = mode_reduced[i]

        # Extract transverse displacements (every other DOF starting from 0)
        # DOF order: [v1, θ1, v2, θ2, ...]
        transverse_disp = mode_full[0::2]  # v values at each node

        # Normalize to unit maximum
        if np.max(np.abs(transverse_disp)) > 0:
            transverse_disp = transverse_disp / np.max(np.abs(transverse_disp))

        # Node positions along beam
        x_nodes = np.linspace(0, L, n_elements + 1)

        # Interpolate for smooth plotting
        from scipy.interpolate import CubicSpline
        if len(x_nodes) > 3:
            cs = CubicSpline(x_nodes, transverse_disp)
            x_fine = np.linspace(0, L, 200)
            phi = cs(x_fine)
        else:
            x_fine = x_nodes
            phi = transverse_disp

        # Scale for visualization
        amplitude = 0.15
        y_deformed = phi * amplitude

        # Create beam outline (deformed shape)
        beam_top = y_deformed + 0.02
        beam_bottom = y_deformed - 0.02

        # Create color gradient based on displacement
        phi_range = phi.max() - phi.min()
        if phi_range > 0:
            norm_phi = (phi - phi.min()) / phi_range
        else:
            norm_phi = np.zeros_like(phi)
        colors = plt.cm.jet(norm_phi)

        # Plot deformed beam as filled region with color gradient
        for i in range(len(x_fine) - 1):
            verts = [
                (x_fine[i], beam_bottom[i]),
                (x_fine[i+1], beam_bottom[i+1]),
                (x_fine[i+1], beam_top[i+1]),
                (x_fine[i], beam_top[i])
            ]
            poly = plt.Polygon(verts, facecolor=colors[i], edgecolor='none')
            ax.add_patch(poly)

        # Add outline
        ax.plot(x_fine, beam_top, 'k-', linewidth=1)
        ax.plot(x_fine, beam_bottom, 'k-', linewidth=1)

        # Plot node positions as markers
        ax.plot(x_nodes, transverse_disp * amplitude, 'ko', markersize=3, alpha=0.5)

        # Fixed supports visualization
        support_size = 0.08

        # Left support (triangle)
        left_tri = plt.Polygon([
            (-support_size, -support_size),
            (0, 0),
            (-support_size, support_size)
        ], facecolor='gray', edgecolor='black')
        ax.add_patch(left_tri)

        # Right support
        right_tri = plt.Polygon([
            (L + support_size, -support_size),
            (L, 0),
            (L + support_size, support_size)
        ], facecolor='gray', edgecolor='black')
        ax.add_patch(right_tri)

        # Reference line (undeformed)
        ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)

        # Labels - show OUR computed frequency
        ax.set_title(f'Mode {idx+1}\nf = {freq:.2f} Hz (FEM)',
                     fontsize=11, fontweight='bold')
        ax.set_xlabel('Position along beam (m)', fontsize=10)
        if idx == 0:
            ax.set_ylabel('Displacement (normalized)', fontsize=10)

        ax.set_xlim(-0.15, L + 0.15)
        ax.set_ylim(-0.25, 0.25)
        ax.set_aspect('equal')

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap='jet', norm=plt.Normalize(-1, 1))
    sm.set_array([])
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cb = fig.colorbar(sm, cax=cbar_ax)
    cb.set_label('Normalized Displacement', fontsize=10)

    plt.suptitle('Mode Shapes from Python FEM Simulation\n(Fixed-Fixed Beam, 20 Euler-Bernoulli Elements)',
                 fontsize=13, fontweight='bold', y=1.02)

    plt.tight_layout(rect=[0, 0, 0.9, 1])
    return fig


def generate_markdown_table(frequencies, theoretical):
    """Generate Markdown table for thesis documentation."""

    print("\n" + "=" * 70)
    print("MARKDOWN TABLE: For documentation_humanized.md")
    print("=" * 70)

    print("""
**Table 4.2: Three-Way Validation for Fixed-Fixed Beam**

| Mode | Gautam et al. (2016) ANSYS (Hz) | Our Python FEM (Hz) | Theoretical EBT (Hz) | Error vs ANSYS |
|------|--------------------------------|---------------------|----------------------|----------------|""")

    for i in range(3):
        ansys = TARGET_ANSYS[f'f{i+1}']
        our = frequencies[i]
        theory = theoretical[i]
        error = abs(our - ansys) / ansys * 100
        print(f"| {i+1} | {ansys:.2f} | {our:.2f} | {theory:.2f} | {error:.3f}% |")

    print("\n")


# ============================================================================
# Main Execution
# ============================================================================

def main():
    """Main validation routine."""

    # Create output directory
    output_dir = Path(__file__).parent.parent / 'docs' / 'figures' / 'gautam_validation'
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("GAUTAM et al. (2016) FEM VALIDATION")
    print("Modal Analysis of Beam Through Analytically and FEM")
    print("ICITSEM-16, New Delhi, India, pp. 375-383")
    print("=" * 70)

    # 1. Validate beam
    frequencies, mode_shapes, free_dof, theoretical = validate_beam()

    # 2. Generate Markdown table
    generate_markdown_table(frequencies, theoretical)

    # 3. Create visualizations
    print("\n" + "=" * 70)
    print("GENERATING FIGURES")
    print("=" * 70)

    # FE Model diagram
    fig1 = plot_fem_model()
    fig1.savefig(output_dir / 'fe_model_gautam.png', dpi=300, bbox_inches='tight',
                 facecolor='white', edgecolor='none')
    print(f"   Saved: {output_dir / 'fe_model_gautam.png'}")

    # Validation comparison
    fig2 = plot_validation_comparison(frequencies, theoretical)
    fig2.savefig(output_dir / 'validation_comparison.png', dpi=300, bbox_inches='tight',
                 facecolor='white', edgecolor='none')
    print(f"   Saved: {output_dir / 'validation_comparison.png'}")

    # Mode shapes (using actual FEM results)
    fig3 = plot_mode_shapes(frequencies, mode_shapes, free_dof)
    fig3.savefig(output_dir / 'mode_shapes.png', dpi=300, bbox_inches='tight',
                 facecolor='white', edgecolor='none')
    print(f"   Saved: {output_dir / 'mode_shapes.png'}")

    plt.close('all')

    # 4. Summary
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)

    errors = []
    print("\nComparison with Gautam et al. (2016) ANSYS:")
    for i in range(3):
        ansys = TARGET_ANSYS[f'f{i+1}']
        our = frequencies[i]
        error = abs(our - ansys) / ansys * 100
        errors.append(error)
        status = "✓" if error < 1.0 else "~" if error < 5.0 else "✗"
        print(f"   Mode {i+1}: Our FEM = {our:.2f} Hz, ANSYS = {ansys:.2f} Hz, Error = {error:.3f}% {status}")

    print(f"\nMaximum Error: {max(errors):.3f}%")
    print(f"Target: ≤ 1.0%")

    if max(errors) < 1.0:
        print("Status: PASSED ✓ - Excellent agreement with published ANSYS results")
    elif max(errors) < 5.0:
        print("Status: ACCEPTABLE ~ - Within expected range for EBT vs 3D FEM comparison")
    else:
        print("Status: REVIEW NEEDED - Larger discrepancy than expected")

    print("\nNote: Small differences are expected because:")
    print("  - Gautam used ANSYS Solid185 3D elements (includes shear effects)")
    print("  - Our implementation uses Euler-Bernoulli beam theory (neglects shear)")
    print("  - For L/h = 20 (slender beam), this difference is typically < 1%")

    return frequencies, theoretical


if __name__ == "__main__":
    frequencies, theoretical = main()
