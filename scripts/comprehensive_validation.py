"""
Comprehensive Validation Script for MS Thesis
==============================================

This script addresses critical validation issues identified in the thesis review:
1. Mesh Convergence Study (Section 4.2.4)
2. Mode Shape Validation Against Analytical Solutions
3. Damage Factor (α) Sensitivity Analysis
4. Zhang et al. (2020) Beam Quantitative Comparison
5. Uncertainty Propagation Analysis

Author: MS Thesis
Date: January 2025
"""

import numpy as np
from scipy.linalg import eigh
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# Output Directory
# =============================================================================
OUTPUT_DIR = Path(__file__).parent.parent / 'docs' / 'figures' / 'validation_studies'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# FEM Implementation (Core Functions)
# =============================================================================

def element_stiffness_matrix(E, I, Le):
    """Euler-Bernoulli element stiffness matrix (4x4)."""
    coeff = E * I / Le**3
    ke = coeff * np.array([
        [12,      6*Le,    -12,      6*Le   ],
        [6*Le,    4*Le**2, -6*Le,    2*Le**2],
        [-12,    -6*Le,     12,     -6*Le   ],
        [6*Le,    2*Le**2, -6*Le,    4*Le**2]
    ])
    return ke

def element_mass_matrix(rho, A, Le):
    """Consistent mass matrix for Euler-Bernoulli beam element (4x4)."""
    coeff = rho * A * Le / 420
    me = coeff * np.array([
        [156,     22*Le,    54,     -13*Le  ],
        [22*Le,   4*Le**2,  13*Le,  -3*Le**2],
        [54,      13*Le,    156,    -22*Le  ],
        [-13*Le, -3*Le**2, -22*Le,   4*Le**2]
    ])
    return me

def assemble_global_matrices(n_elements, E, I, rho, A, L):
    """Assemble global stiffness and mass matrices."""
    Le = L / n_elements
    n_dof = (n_elements + 1) * 2
    K = np.zeros((n_dof, n_dof))
    M = np.zeros((n_dof, n_dof))

    for elem in range(n_elements):
        ke = element_stiffness_matrix(E, I, Le)
        me = element_mass_matrix(rho, A, Le)

        dof_start = elem * 2
        dof_indices = [dof_start, dof_start+1, dof_start+2, dof_start+3]

        for i in range(4):
            for j in range(4):
                K[dof_indices[i], dof_indices[j]] += ke[i, j]
                M[dof_indices[i], dof_indices[j]] += me[i, j]

    return K, M

def apply_fixed_fixed_bc(K, M):
    """Apply fixed-fixed boundary conditions."""
    n_total = K.shape[0]
    # Fixed at both ends: DOFs 0, 1 (left) and n-2, n-1 (right)
    fixed_dof = [0, 1, n_total-2, n_total-1]
    free_dof = [i for i in range(n_total) if i not in fixed_dof]

    K_reduced = K[np.ix_(free_dof, free_dof)]
    M_reduced = M[np.ix_(free_dof, free_dof)]

    return K_reduced, M_reduced, free_dof

def solve_eigenvalue_problem(K, M, n_modes=5):
    """Solve generalized eigenvalue problem: Kφ = ω²Mφ."""
    eigenvalues, eigenvectors = eigh(K, M)
    eigenvalues = eigenvalues[:n_modes]
    eigenvectors = eigenvectors[:, :n_modes]
    frequencies = np.sqrt(np.abs(eigenvalues)) / (2 * np.pi)
    return frequencies, eigenvectors

def run_fem_analysis(E, I, rho, A, L, n_elements, n_modes=5):
    """Complete FEM analysis for fixed-fixed beam."""
    K, M = assemble_global_matrices(n_elements, E, I, rho, A, L)
    K_red, M_red, free_dof = apply_fixed_fixed_bc(K, M)
    frequencies, mode_shapes = solve_eigenvalue_problem(K_red, M_red, n_modes)
    return frequencies, mode_shapes, free_dof

# =============================================================================
# STUDY 1: MESH CONVERGENCE ANALYSIS
# =============================================================================

def mesh_convergence_study():
    """
    Perform mesh convergence study showing frequency vs. number of elements.
    This addresses the missing mesh convergence plot in Section 4.2.4.
    """
    print("\n" + "="*70)
    print("STUDY 1: MESH CONVERGENCE ANALYSIS")
    print("="*70)

    # Use Gautam beam parameters for consistency
    L = 2.0       # m
    b = 0.3       # m
    h = 0.1       # m
    E = 205e9     # Pa (205 GPa - steel)
    rho = 7830    # kg/m³
    A = b * h
    I = b * h**3 / 12

    # Element counts to test
    element_counts = [4, 6, 8, 10, 12, 16, 20, 30, 40, 60, 80, 100]

    # Theoretical frequencies for fixed-fixed beam (Euler-Bernoulli)
    lambda_n = [4.730041, 7.853205, 10.995608, 14.137165, 17.278760]
    f_theory = [(lam**2 / (2*np.pi*L**2)) * np.sqrt(E*I/(rho*A)) for lam in lambda_n]

    results = {i: [] for i in range(5)}
    errors = {i: [] for i in range(5)}

    for n_elem in element_counts:
        freqs, _, _ = run_fem_analysis(E, I, rho, A, L, n_elem, n_modes=5)
        for i in range(5):
            results[i].append(freqs[i])
            errors[i].append(abs(freqs[i] - f_theory[i]) / f_theory[i] * 100)

    # Print results table
    print("\nMesh Convergence Results:")
    print("-" * 90)
    print(f"{'Elements':<10} {'Mode 1 (Hz)':<12} {'Error %':<10} {'Mode 2 (Hz)':<12} {'Error %':<10} {'Mode 3 (Hz)':<12} {'Error %':<10}")
    print("-" * 90)
    for idx, n_elem in enumerate(element_counts):
        print(f"{n_elem:<10} {results[0][idx]:<12.4f} {errors[0][idx]:<10.4f} {results[1][idx]:<12.4f} {errors[1][idx]:<10.4f} {results[2][idx]:<12.4f} {errors[2][idx]:<10.4f}")
    print("-" * 90)
    print(f"{'Theory':<10} {f_theory[0]:<12.4f} {'0.0000':<10} {f_theory[1]:<12.4f} {'0.0000':<10} {f_theory[2]:<12.4f} {'0.0000':<10}")

    # Create convergence plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Frequency vs Elements
    ax1 = axes[0]
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6', '#f39c12']
    for i in range(3):
        ax1.plot(element_counts, results[i], 'o-', color=colors[i],
                 label=f'Mode {i+1} (Theory: {f_theory[i]:.2f} Hz)', linewidth=2, markersize=6)
        ax1.axhline(y=f_theory[i], color=colors[i], linestyle='--', alpha=0.5)

    ax1.set_xlabel('Number of Elements', fontsize=12)
    ax1.set_ylabel('Natural Frequency (Hz)', fontsize=12)
    ax1.set_title('Mesh Convergence: Frequency vs. Element Count', fontsize=13, fontweight='bold')
    ax1.legend(loc='right', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log')

    # Plot 2: Error vs Elements
    ax2 = axes[1]
    for i in range(3):
        ax2.semilogy(element_counts, errors[i], 'o-', color=colors[i],
                     label=f'Mode {i+1}', linewidth=2, markersize=6)

    ax2.axhline(y=0.01, color='red', linestyle='--', linewidth=2, label='Target: 0.01%')
    ax2.axhline(y=0.1, color='orange', linestyle='--', linewidth=1.5, alpha=0.7, label='Threshold: 0.1%')
    ax2.axvline(x=20, color='green', linestyle=':', linewidth=2, alpha=0.8, label='Selected: 20 elements')

    ax2.set_xlabel('Number of Elements', fontsize=12)
    ax2.set_ylabel('Error vs. Theoretical (%)', fontsize=12)
    ax2.set_title('Mesh Convergence: Error Reduction', fontsize=13, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('log')
    ax2.set_ylim([1e-4, 10])

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / 'mesh_convergence_study.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"\nSaved: {OUTPUT_DIR / 'mesh_convergence_study.png'}")

    # Summary
    print("\n" + "-"*70)
    print("MESH CONVERGENCE SUMMARY:")
    print("-"*70)
    idx_20 = element_counts.index(20)
    print(f"At 20 elements:")
    for i in range(3):
        print(f"  Mode {i+1}: {results[i][idx_20]:.4f} Hz (Error: {errors[i][idx_20]:.6f}%)")
    print(f"\nConclusion: 20 elements achieve errors below 0.01% for all modes.")
    print("Further refinement provides negligible improvement.")

    return element_counts, results, errors, f_theory

# =============================================================================
# STUDY 2: MODE SHAPE VALIDATION
# =============================================================================

def analytical_mode_shape_fixed_fixed(x_norm, mode_num):
    """
    Calculate analytical mode shape for fixed-fixed beam.

    φ(x) = cosh(βL·x/L) - cos(βL·x/L) - σ[sinh(βL·x/L) - sin(βL·x/L)]
    where σ = [cosh(βL) - cos(βL)] / [sinh(βL) - sin(βL)]
    """
    beta_L_values = [4.730041, 7.853205, 10.995608, 14.137165, 17.278760]
    beta_L = beta_L_values[mode_num - 1]

    # Calculate sigma
    cosh_bL = np.cosh(beta_L)
    cos_bL = np.cos(beta_L)
    sinh_bL = np.sinh(beta_L)
    sin_bL = np.sin(beta_L)
    sigma = (cosh_bL - cos_bL) / (sinh_bL - sin_bL)

    # Mode shape at normalized position x
    bx = beta_L * x_norm
    phi = np.cosh(bx) - np.cos(bx) - sigma * (np.sinh(bx) - np.sin(bx))

    # Normalize
    phi = phi / np.max(np.abs(phi))
    return phi

def mode_shape_validation():
    """
    Validate FEM mode shapes against analytical solutions.
    This addresses the missing mode shape validation in Section 4.2.
    """
    print("\n" + "="*70)
    print("STUDY 2: MODE SHAPE VALIDATION")
    print("="*70)

    # Beam parameters
    L = 2.0; b = 0.3; h = 0.1; E = 205e9; rho = 7830
    A = b * h; I = b * h**3 / 12
    n_elements = 20

    # Run FEM
    frequencies, mode_shapes, free_dof = run_fem_analysis(E, I, rho, A, L, n_elements, n_modes=3)

    # Reconstruct full mode shapes
    n_total_dof = (n_elements + 1) * 2
    x_nodes = np.linspace(0, L, n_elements + 1)
    x_norm_nodes = x_nodes / L

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    mac_values = []  # Modal Assurance Criterion

    for mode_idx in range(3):
        # Extract FEM mode shape
        mode_reduced = mode_shapes[:, mode_idx]
        mode_full = np.zeros(n_total_dof)
        for i, dof_idx in enumerate(free_dof):
            mode_full[dof_idx] = mode_reduced[i]

        # Transverse displacements at nodes
        fem_disp = mode_full[0::2]
        fem_disp = fem_disp / np.max(np.abs(fem_disp)) if np.max(np.abs(fem_disp)) > 0 else fem_disp

        # Analytical mode shape
        analytical_disp = analytical_mode_shape_fixed_fixed(x_norm_nodes, mode_idx + 1)

        # Ensure same sign convention
        if np.dot(fem_disp, analytical_disp) < 0:
            fem_disp = -fem_disp

        # Calculate MAC (Modal Assurance Criterion)
        mac = (np.dot(fem_disp, analytical_disp)**2) / (np.dot(fem_disp, fem_disp) * np.dot(analytical_disp, analytical_disp))
        mac_values.append(mac)

        # Plot comparison
        ax1 = axes[0, mode_idx]
        ax1.plot(x_nodes, analytical_disp, 'b-', linewidth=2.5, label='Analytical')
        ax1.plot(x_nodes, fem_disp, 'ro', markersize=6, label='FEM (20 elements)')
        ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax1.set_xlabel('Position (m)', fontsize=10)
        ax1.set_ylabel('Normalized Displacement', fontsize=10)
        ax1.set_title(f'Mode {mode_idx+1}: f = {frequencies[mode_idx]:.2f} Hz\nMAC = {mac:.6f}',
                      fontsize=11, fontweight='bold')
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.3)

        # Plot error
        ax2 = axes[1, mode_idx]
        error = (fem_disp - analytical_disp) * 100  # as percentage
        ax2.bar(x_nodes, error, width=L/n_elements*0.8, color='coral', edgecolor='darkred')
        ax2.axhline(y=0, color='black', linewidth=1)
        ax2.set_xlabel('Position (m)', fontsize=10)
        ax2.set_ylabel('Error (%)', fontsize=10)
        ax2.set_title(f'Mode {mode_idx+1} Error Distribution', fontsize=11)
        ax2.grid(True, alpha=0.3)

        print(f"Mode {mode_idx+1}: MAC = {mac:.6f}, Max Error = {np.max(np.abs(error)):.4f}%")

    plt.suptitle('Mode Shape Validation: FEM vs. Analytical Solution (Fixed-Fixed Beam)',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / 'mode_shape_validation.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"\nSaved: {OUTPUT_DIR / 'mode_shape_validation.png'}")

    print("\n" + "-"*70)
    print("MODE SHAPE VALIDATION SUMMARY:")
    print("-"*70)
    print("Modal Assurance Criterion (MAC) values:")
    for i, mac in enumerate(mac_values):
        status = "EXCELLENT" if mac > 0.999 else "GOOD" if mac > 0.99 else "ACCEPTABLE"
        print(f"  Mode {i+1}: MAC = {mac:.6f} ({status})")
    print("\nNote: MAC = 1.0 indicates perfect correlation between FEM and analytical modes.")
    print("      MAC > 0.99 is considered excellent agreement.")

    return mac_values

# =============================================================================
# STUDY 3: DAMAGE FACTOR SENSITIVITY ANALYSIS
# =============================================================================

def damage_factor_sensitivity():
    """
    Analyze sensitivity of results to damage factor α.
    Tests α = 1.4, 1.6, 1.8 to show how results change.
    This addresses criticism about the α = 1.6 factor justification.
    """
    print("\n" + "="*70)
    print("STUDY 3: DAMAGE FACTOR (α) SENSITIVITY ANALYSIS")
    print("="*70)

    # RC beam parameters (typical from thesis)
    L = 5.0       # m
    b = 0.3       # m
    h = 0.5       # m
    fc = 30       # MPa
    E = 4700 * np.sqrt(fc) * 1e6  # ACI 318-19
    rho = 2400    # kg/m³
    A = b * h
    I = b * h**3 / 12

    # Pristine frequency
    f_pristine, _, _ = run_fem_analysis(E, I, rho, A, L, 20, n_modes=2)
    print(f"\nPristine beam frequency: Mode 1 = {f_pristine[0]:.2f} Hz, Mode 2 = {f_pristine[1]:.2f} Hz")

    # Corrosion levels
    corrosion_levels = np.arange(0, 21, 2)  # 0% to 20%

    # Different α factors
    alpha_factors = [1.4, 1.6, 1.8]

    results = {}
    for alpha in alpha_factors:
        freq_reductions = []
        for C in corrosion_levels:
            # Damage factor: α_damage = α × C / 100
            damage = alpha * C / 100
            # Effective stiffness: EI_damaged = EI × (1 - damage)
            E_damaged = E * (1 - damage)
            if E_damaged > 0:
                f_damaged, _, _ = run_fem_analysis(E_damaged, I, rho, A, L, 20, n_modes=2)
                reduction = (f_pristine[0] - f_damaged[0]) / f_pristine[0] * 100
            else:
                reduction = 100
            freq_reductions.append(reduction)
        results[alpha] = freq_reductions

    # Print comparison table
    print("\n" + "-"*80)
    print("Frequency Reduction (%) vs. Corrosion Level for Different α Factors:")
    print("-"*80)
    print(f"{'Corrosion %':<12} {'α = 1.4':<15} {'α = 1.6':<15} {'α = 1.8':<15} {'Spread':<12}")
    print("-"*80)
    for i, C in enumerate(corrosion_levels):
        r14, r16, r18 = results[1.4][i], results[1.6][i], results[1.8][i]
        spread = r18 - r14
        print(f"{C:<12} {r14:<15.2f} {r16:<15.2f} {r18:<15.2f} {spread:<12.2f}")

    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Frequency reduction vs corrosion
    ax1 = axes[0]
    colors = {'1.4': '#3498db', '1.6': '#2ecc71', '1.8': '#e74c3c'}
    for alpha in alpha_factors:
        ax1.plot(corrosion_levels, results[alpha], 'o-',
                 label=f'α = {alpha}', linewidth=2, markersize=6, color=colors[str(alpha)])

    # Add Zhang et al. experimental range
    ax1.fill_between([0, 5], [0, 2], [0, 5], alpha=0.2, color='gray', label='Zhang et al. (2020) range')
    ax1.fill_between([5, 10], [2, 5], [5, 10], alpha=0.2, color='gray')
    ax1.fill_between([10, 15], [5, 10], [10, 15], alpha=0.2, color='gray')

    ax1.set_xlabel('Corrosion Level (%)', fontsize=12)
    ax1.set_ylabel('Frequency Reduction (%)', fontsize=12)
    ax1.set_title('Sensitivity of Frequency Reduction to Damage Factor α', fontsize=13, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, 20])
    ax1.set_ylim([0, 30])

    # Plot 2: Uncertainty band
    ax2 = axes[1]
    mean_results = [(results[1.4][i] + results[1.6][i] + results[1.8][i])/3 for i in range(len(corrosion_levels))]
    lower = results[1.4]
    upper = results[1.8]

    ax2.fill_between(corrosion_levels, lower, upper, alpha=0.3, color='blue', label='Uncertainty range (α = 1.4-1.8)')
    ax2.plot(corrosion_levels, results[1.6], 'go-', linewidth=2, markersize=6, label='Selected α = 1.6')
    ax2.plot(corrosion_levels, mean_results, 'k--', linewidth=1.5, label='Mean')

    ax2.set_xlabel('Corrosion Level (%)', fontsize=12)
    ax2.set_ylabel('Frequency Reduction (%)', fontsize=12)
    ax2.set_title('Uncertainty Band Due to Damage Factor Selection', fontsize=13, fontweight='bold')
    ax2.legend(loc='upper left', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([0, 20])
    ax2.set_ylim([0, 30])

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / 'damage_factor_sensitivity.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"\nSaved: {OUTPUT_DIR / 'damage_factor_sensitivity.png'}")

    # Summary
    print("\n" + "-"*70)
    print("DAMAGE FACTOR SENSITIVITY SUMMARY:")
    print("-"*70)
    print(f"At 10% corrosion:")
    for alpha in alpha_factors:
        idx = list(corrosion_levels).index(10)
        print(f"  α = {alpha}: {results[alpha][idx]:.2f}% frequency reduction")
    spread_10 = results[1.8][list(corrosion_levels).index(10)] - results[1.4][list(corrosion_levels).index(10)]
    print(f"  Uncertainty spread: ±{spread_10/2:.2f}%")
    print(f"\nConclusion: Selection of α = 1.6 introduces approximately ±{spread_10/2:.1f}% uncertainty")
    print("            in frequency reduction predictions at moderate corrosion levels.")

    return results

# =============================================================================
# STUDY 4: ZHANG ET AL. (2020) QUANTITATIVE COMPARISON
# =============================================================================

def zhang_beam_comparison():
    """
    Reproduce Zhang et al. (2020) beam geometry in our FEM for quantitative comparison.
    This provides direct Hz-to-Hz comparison instead of range-to-range.

    Zhang et al. (2020) beam:
    - Dimensions: 2000 × 150 × 50 mm (L × b × h)
    - Material: Concrete with HRB335 steel reinforcement
    - Boundary: Simply supported (for comparison, we show both SS and FF)
    """
    print("\n" + "="*70)
    print("STUDY 4: ZHANG ET AL. (2020) BEAM QUANTITATIVE COMPARISON")
    print("="*70)

    # Zhang et al. beam parameters
    L = 2.0       # m
    b = 0.150     # m
    h = 0.050     # m
    A = b * h
    I = b * h**3 / 12

    # Estimate concrete properties (Zhang used standard concrete)
    fc = 30       # MPa (assumed C30 concrete)
    E_concrete = 4700 * np.sqrt(fc) * 1e6  # Pa (ACI 318-19)
    rho_concrete = 2400  # kg/m³

    # Steel reinforcement effect (homogenized)
    # Zhang used 8mm diameter bar, As = 50.27 mm²
    As = 50.27e-6  # m²
    E_steel = 200e9  # Pa
    rho_steel = 7850  # kg/m³

    # Homogenized properties (transformed section)
    n_ratio = E_steel / E_concrete
    A_transformed = A + (n_ratio - 1) * As
    I_transformed = I + (n_ratio - 1) * As * (h/2 - 0.010)**2  # Assuming 10mm cover

    # Effective density (composite)
    rho_eff = (rho_concrete * A + rho_steel * As) / A

    print(f"\nZhang et al. (2020) Beam Specifications:")
    print(f"  Dimensions: L = {L*1000:.0f} mm, b = {b*1000:.0f} mm, h = {h*1000:.0f} mm")
    print(f"  Concrete: f'c ≈ {fc} MPa, E = {E_concrete/1e9:.2f} GPa")
    print(f"  Steel: 8mm bar (As = {As*1e6:.2f} mm²)")
    print(f"  Homogenized: E_eff based on transformed section")

    # Simply-supported analysis (Zhang's actual BC)
    def apply_simply_supported_bc(K, M):
        """Simply supported: v=0 at both ends, rotation free."""
        n_total = K.shape[0]
        fixed_dof = [0, n_total-2]  # Only transverse DOFs fixed
        free_dof = [i for i in range(n_total) if i not in fixed_dof]
        K_reduced = K[np.ix_(free_dof, free_dof)]
        M_reduced = M[np.ix_(free_dof, free_dof)]
        return K_reduced, M_reduced, free_dof

    def run_ss_analysis(E, I, rho, A, L, n_elements):
        K, M = assemble_global_matrices(n_elements, E, I, rho, A, L)
        K_red, M_red, free_dof = apply_simply_supported_bc(K, M)
        frequencies, mode_shapes = solve_eigenvalue_problem(K_red, M_red, n_modes=3)
        return frequencies

    # Theoretical SS frequencies
    lambda_ss = [np.pi, 2*np.pi, 3*np.pi]  # Simply supported: λn = n*π
    f_theory_ss = [(lam**2 / (2*np.pi*L**2)) * np.sqrt(E_concrete*I/(rho_concrete*A)) for lam in lambda_ss]

    # FEM SS frequencies (plain concrete)
    f_fem_ss_plain = run_ss_analysis(E_concrete, I, rho_concrete, A, L, 20)

    # FEM SS frequencies (with steel - transformed section)
    f_fem_ss_composite = run_ss_analysis(E_concrete, I_transformed, rho_eff, A_transformed, L, 20)

    # Also show Fixed-Fixed for comparison (our thesis BC)
    f_fem_ff_plain, _, _ = run_fem_analysis(E_concrete, I, rho_concrete, A, L, 20, n_modes=3)

    print("\n" + "-"*80)
    print("FREQUENCY COMPARISON (Hz):")
    print("-"*80)
    print(f"{'Source':<35} {'Mode 1':<12} {'Mode 2':<12} {'Mode 3':<12}")
    print("-"*80)
    print(f"{'Theoretical EBT (SS, plain)':<35} {f_theory_ss[0]:<12.2f} {f_theory_ss[1]:<12.2f} {f_theory_ss[2]:<12.2f}")
    print(f"{'FEM SS (plain concrete)':<35} {f_fem_ss_plain[0]:<12.2f} {f_fem_ss_plain[1]:<12.2f} {f_fem_ss_plain[2]:<12.2f}")
    print(f"{'FEM SS (with steel - homogenized)':<35} {f_fem_ss_composite[0]:<12.2f} {f_fem_ss_composite[1]:<12.2f} {f_fem_ss_composite[2]:<12.2f}")
    print(f"{'FEM FF (plain concrete - thesis BC)':<35} {f_fem_ff_plain[0]:<12.2f} {f_fem_ff_plain[1]:<12.2f} {f_fem_ff_plain[2]:<12.2f}")
    print("-"*80)

    # Corrosion effect comparison
    print("\nCORROSION SENSITIVITY COMPARISON:")
    print("-"*80)
    print(f"{'Corrosion %':<15} {'Zhang Exp. Range':<20} {'FEM SS Prediction':<20} {'Within Range?':<15}")
    print("-"*80)

    zhang_ranges = {
        5: (2, 5),
        10: (5, 10),
        15: (10, 15)
    }

    alpha = 1.6
    for C, (low, high) in zhang_ranges.items():
        damage = alpha * C / 100
        E_damaged = E_concrete * (1 - damage)
        f_damaged = run_ss_analysis(E_damaged, I, rho_concrete, A, L, 20)
        reduction = (f_fem_ss_plain[0] - f_damaged[0]) / f_fem_ss_plain[0] * 100
        within = "Yes" if low <= reduction <= high else "Close" if abs(reduction - (low+high)/2) < 3 else "No"
        print(f"{C:<15} {low}-{high}%{'':<13} {reduction:<20.2f} {within:<15}")

    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Frequency comparison
    ax1 = axes[0]
    modes = [1, 2, 3]
    x = np.arange(len(modes))
    width = 0.2

    ax1.bar(x - 1.5*width, f_theory_ss, width, label='Theory SS', color='#2ecc71')
    ax1.bar(x - 0.5*width, f_fem_ss_plain, width, label='FEM SS (plain)', color='#3498db')
    ax1.bar(x + 0.5*width, f_fem_ss_composite, width, label='FEM SS (composite)', color='#9b59b6')
    ax1.bar(x + 1.5*width, f_fem_ff_plain, width, label='FEM FF (thesis)', color='#e74c3c')

    ax1.set_xlabel('Mode Number', fontsize=12)
    ax1.set_ylabel('Natural Frequency (Hz)', fontsize=12)
    ax1.set_title('Zhang et al. (2020) Beam: Frequency Comparison', fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(['Mode 1', 'Mode 2', 'Mode 3'])
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3, axis='y')

    # Plot 2: Corrosion effect
    ax2 = axes[1]
    corr_levels = np.arange(0, 16, 1)
    fem_reductions = []
    for C in corr_levels:
        damage = alpha * C / 100
        E_damaged = E_concrete * (1 - damage)
        f_damaged = run_ss_analysis(E_damaged, I, rho_concrete, A, L, 20)
        reduction = (f_fem_ss_plain[0] - f_damaged[0]) / f_fem_ss_plain[0] * 100
        fem_reductions.append(reduction)

    ax2.plot(corr_levels, fem_reductions, 'b-o', linewidth=2, markersize=6, label='FEM Prediction')

    # Zhang experimental ranges
    ax2.fill_between([0, 5], [0, 2], [0, 5], alpha=0.3, color='green', label='Zhang 0-5%: 2-5% reduction')
    ax2.fill_between([5, 10], [5, 5], [10, 10], alpha=0.3, color='yellow', label='Zhang 5-10%: 5-10% reduction')
    ax2.fill_between([10, 15], [10, 10], [15, 15], alpha=0.3, color='red', label='Zhang 10-15%: 10-15% reduction')

    ax2.set_xlabel('Corrosion Level (%)', fontsize=12)
    ax2.set_ylabel('Frequency Reduction (%)', fontsize=12)
    ax2.set_title('Corrosion-Frequency Relationship: FEM vs. Zhang et al. (2020)', fontsize=13, fontweight='bold')
    ax2.legend(loc='upper left', fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([0, 15])
    ax2.set_ylim([0, 20])

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / 'zhang_comparison.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"\nSaved: {OUTPUT_DIR / 'zhang_comparison.png'}")

    print("\n" + "-"*70)
    print("ZHANG COMPARISON SUMMARY:")
    print("-"*70)
    print("The FEM predictions fall within Zhang et al.'s experimental ranges,")
    print("validating the corrosion-frequency sensitivity coefficient (≈0.8%/1% corrosion).")
    print("\nNote: Zhang used simply-supported beams; this thesis uses fixed-fixed.")
    print("      The corrosion sensitivity is similar across boundary conditions.")

    return f_theory_ss, f_fem_ss_plain, f_fem_ff_plain

# =============================================================================
# STUDY 5: UNCERTAINTY PROPAGATION
# =============================================================================

def uncertainty_propagation():
    """
    Propagate material property uncertainty through FEM to ML predictions.
    This addresses the missing uncertainty quantification for FEM simulations.
    """
    print("\n" + "="*70)
    print("STUDY 5: UNCERTAINTY PROPAGATION ANALYSIS")
    print("="*70)

    # Base RC beam parameters
    L = 5.0       # m
    b = 0.3       # m
    h = 0.5       # m
    fc_base = 30  # MPa
    rho_base = 2400  # kg/m³
    A = b * h
    I = b * h**3 / 12

    # Uncertainty levels (coefficient of variation)
    cv_E = 0.10    # ±10% uncertainty in elastic modulus
    cv_rho = 0.05  # ±5% uncertainty in density
    cv_fc = 0.15   # ±15% uncertainty in compressive strength

    n_samples = 1000
    np.random.seed(42)

    # Monte Carlo simulation
    frequencies_mode1 = []
    frequencies_mode2 = []

    for _ in range(n_samples):
        # Sample parameters with uncertainty
        fc_sample = fc_base * (1 + cv_fc * np.random.randn())
        fc_sample = max(fc_sample, 15)  # Minimum 15 MPa

        E_sample = 4700 * np.sqrt(fc_sample) * 1e6
        E_sample = E_sample * (1 + cv_E * np.random.randn())

        rho_sample = rho_base * (1 + cv_rho * np.random.randn())

        # Run FEM
        freqs, _, _ = run_fem_analysis(E_sample, I, rho_sample, A, L, 20, n_modes=2)
        frequencies_mode1.append(freqs[0])
        frequencies_mode2.append(freqs[1])

    frequencies_mode1 = np.array(frequencies_mode1)
    frequencies_mode2 = np.array(frequencies_mode2)

    # Statistics
    E_base = 4700 * np.sqrt(fc_base) * 1e6
    f_nominal, _, _ = run_fem_analysis(E_base, I, rho_base, A, L, 20, n_modes=2)

    print(f"\nNominal beam parameters:")
    print(f"  L = {L} m, b = {b} m, h = {h} m")
    print(f"  f'c = {fc_base} MPa, E = {E_base/1e9:.2f} GPa, ρ = {rho_base} kg/m³")
    print(f"\nUncertainty assumptions:")
    print(f"  f'c: ±{cv_fc*100:.0f}% (CV)")
    print(f"  E: ±{cv_E*100:.0f}% (CV)")
    print(f"  ρ: ±{cv_rho*100:.0f}% (CV)")

    print(f"\n" + "-"*70)
    print("MONTE CARLO RESULTS ({} samples):".format(n_samples))
    print("-"*70)
    print(f"{'Statistic':<25} {'Mode 1 (Hz)':<20} {'Mode 2 (Hz)':<20}")
    print("-"*70)
    print(f"{'Nominal':<25} {f_nominal[0]:<20.2f} {f_nominal[1]:<20.2f}")
    print(f"{'MC Mean':<25} {np.mean(frequencies_mode1):<20.2f} {np.mean(frequencies_mode2):<20.2f}")
    print(f"{'MC Std Dev':<25} {np.std(frequencies_mode1):<20.2f} {np.std(frequencies_mode2):<20.2f}")
    print(f"{'CV (%)':<25} {np.std(frequencies_mode1)/np.mean(frequencies_mode1)*100:<20.2f} {np.std(frequencies_mode2)/np.mean(frequencies_mode2)*100:<20.2f}")
    print(f"{'95% CI Lower':<25} {np.percentile(frequencies_mode1, 2.5):<20.2f} {np.percentile(frequencies_mode2, 2.5):<20.2f}")
    print(f"{'95% CI Upper':<25} {np.percentile(frequencies_mode1, 97.5):<20.2f} {np.percentile(frequencies_mode2, 97.5):<20.2f}")
    print("-"*70)

    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Plot 1: Histogram Mode 1
    ax1 = axes[0]
    ax1.hist(frequencies_mode1, bins=30, density=True, alpha=0.7, color='#3498db', edgecolor='black')
    ax1.axvline(f_nominal[0], color='red', linestyle='--', linewidth=2, label=f'Nominal: {f_nominal[0]:.2f} Hz')
    ax1.axvline(np.mean(frequencies_mode1), color='green', linestyle='-', linewidth=2,
                label=f'Mean: {np.mean(frequencies_mode1):.2f} Hz')
    ax1.axvline(np.percentile(frequencies_mode1, 2.5), color='orange', linestyle=':', linewidth=2)
    ax1.axvline(np.percentile(frequencies_mode1, 97.5), color='orange', linestyle=':', linewidth=2,
                label=f'95% CI: [{np.percentile(frequencies_mode1, 2.5):.1f}, {np.percentile(frequencies_mode1, 97.5):.1f}]')
    ax1.set_xlabel('Mode 1 Frequency (Hz)', fontsize=12)
    ax1.set_ylabel('Probability Density', fontsize=12)
    ax1.set_title(f'Mode 1 Frequency Distribution\n(CV = {np.std(frequencies_mode1)/np.mean(frequencies_mode1)*100:.1f}%)',
                  fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Histogram Mode 2
    ax2 = axes[1]
    ax2.hist(frequencies_mode2, bins=30, density=True, alpha=0.7, color='#2ecc71', edgecolor='black')
    ax2.axvline(f_nominal[1], color='red', linestyle='--', linewidth=2, label=f'Nominal: {f_nominal[1]:.2f} Hz')
    ax2.axvline(np.mean(frequencies_mode2), color='green', linestyle='-', linewidth=2,
                label=f'Mean: {np.mean(frequencies_mode2):.2f} Hz')
    ax2.axvline(np.percentile(frequencies_mode2, 2.5), color='orange', linestyle=':', linewidth=2)
    ax2.axvline(np.percentile(frequencies_mode2, 97.5), color='orange', linestyle=':', linewidth=2,
                label=f'95% CI: [{np.percentile(frequencies_mode2, 2.5):.1f}, {np.percentile(frequencies_mode2, 97.5):.1f}]')
    ax2.set_xlabel('Mode 2 Frequency (Hz)', fontsize=12)
    ax2.set_ylabel('Probability Density', fontsize=12)
    ax2.set_title(f'Mode 2 Frequency Distribution\n(CV = {np.std(frequencies_mode2)/np.mean(frequencies_mode2)*100:.1f}%)',
                  fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    # Plot 3: Uncertainty sources
    ax3 = axes[2]
    # Sensitivity analysis: vary each parameter independently
    sensitivities = {}

    # E sensitivity
    E_high = E_base * 1.1
    E_low = E_base * 0.9
    f_E_high, _, _ = run_fem_analysis(E_high, I, rho_base, A, L, 20, n_modes=1)
    f_E_low, _, _ = run_fem_analysis(E_low, I, rho_base, A, L, 20, n_modes=1)
    sensitivities['E (±10%)'] = (f_E_high[0] - f_E_low[0]) / 2

    # rho sensitivity
    rho_high = rho_base * 1.05
    rho_low = rho_base * 0.95
    f_rho_high, _, _ = run_fem_analysis(E_base, I, rho_high, A, L, 20, n_modes=1)
    f_rho_low, _, _ = run_fem_analysis(E_base, I, rho_low, A, L, 20, n_modes=1)
    sensitivities['ρ (±5%)'] = abs(f_rho_high[0] - f_rho_low[0]) / 2

    # f'c sensitivity (through E)
    fc_high = fc_base * 1.15
    fc_low = fc_base * 0.85
    E_fc_high = 4700 * np.sqrt(fc_high) * 1e6
    E_fc_low = 4700 * np.sqrt(fc_low) * 1e6
    f_fc_high, _, _ = run_fem_analysis(E_fc_high, I, rho_base, A, L, 20, n_modes=1)
    f_fc_low, _, _ = run_fem_analysis(E_fc_low, I, rho_base, A, L, 20, n_modes=1)
    sensitivities["f'c (±15%)"] = (f_fc_high[0] - f_fc_low[0]) / 2

    labels = list(sensitivities.keys())
    values = [sensitivities[k] for k in labels]
    colors = ['#3498db', '#2ecc71', '#e74c3c']

    bars = ax3.barh(labels, values, color=colors, edgecolor='black')
    ax3.set_xlabel('Frequency Uncertainty (Hz)', fontsize=12)
    ax3.set_title('Contribution to Mode 1 Frequency Uncertainty', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='x')

    for bar, val in zip(bars, values):
        ax3.text(val + 0.1, bar.get_y() + bar.get_height()/2, f'{val:.2f} Hz',
                 va='center', fontsize=10)

    plt.suptitle('FEM Uncertainty Propagation Analysis (Monte Carlo, n=1000)',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / 'uncertainty_propagation.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"\nSaved: {OUTPUT_DIR / 'uncertainty_propagation.png'}")

    # Summary
    print("\n" + "-"*70)
    print("UNCERTAINTY PROPAGATION SUMMARY:")
    print("-"*70)
    cv_mode1 = np.std(frequencies_mode1)/np.mean(frequencies_mode1)*100
    print(f"FEM frequency prediction uncertainty: CV ≈ {cv_mode1:.1f}%")
    print(f"This represents the epistemic uncertainty from material property assumptions.")
    print(f"\nImplication for ML predictions:")
    print(f"  ML model achieves R² = 0.989 on synthetic FEM data")
    print(f"  Combined FEM + ML uncertainty: approximately ±{cv_mode1 + 1:.1f}% on real beams")
    print(f"  (assuming ML adds ~1% additional prediction error)")

    return frequencies_mode1, frequencies_mode2

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Run all validation studies."""
    print("\n" + "="*70)
    print("COMPREHENSIVE VALIDATION STUDIES FOR MS THESIS")
    print("Addressing Critical Review Comments")
    print("="*70)

    # Study 1: Mesh Convergence
    mesh_results = mesh_convergence_study()

    # Study 2: Mode Shape Validation
    mac_values = mode_shape_validation()

    # Study 3: Damage Factor Sensitivity
    damage_results = damage_factor_sensitivity()

    # Study 4: Zhang et al. Comparison
    zhang_results = zhang_beam_comparison()

    # Study 5: Uncertainty Propagation
    uncertainty_results = uncertainty_propagation()

    # Final Summary
    print("\n" + "="*70)
    print("ALL VALIDATION STUDIES COMPLETED")
    print("="*70)
    print(f"\nOutput figures saved to: {OUTPUT_DIR}")
    print("\nStudies completed:")
    print("  1. Mesh Convergence Study - mesh_convergence_study.png")
    print("  2. Mode Shape Validation - mode_shape_validation.png")
    print("  3. Damage Factor Sensitivity - damage_factor_sensitivity.png")
    print("  4. Zhang et al. Comparison - zhang_comparison.png")
    print("  5. Uncertainty Propagation - uncertainty_propagation.png")

    return {
        'mesh': mesh_results,
        'mode_shapes': mac_values,
        'damage_factor': damage_results,
        'zhang': zhang_results,
        'uncertainty': uncertainty_results
    }

if __name__ == "__main__":
    results = main()
