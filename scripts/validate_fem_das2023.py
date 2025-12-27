"""
FEM Validation Script Against Das (2023) ANSYS Results
=========================================================

This script validates the Python FEM implementation against published ANSYS results
from Das (2023) "Prediction of the natural frequencies of various beams using
regression machine learning models."

Three-Way Validation:
1. Das (2023) ANSYS results (commercially validated)
2. Das (2023) EBT FEM results
3. Our Python FEM implementation

Validation Case: Aluminum Al 7075 Cantilever Beam
- Length: 1.2 m
- Width: 0.025 m
- Height: 0.025 m
- h/L ratio: 1/48 (thin beam, Euler-Bernoulli valid)
- Material: E = 72 GPa, rho = 2810 kg/m³, nu = 0.33

Reference: Das (2023), Table 3, Case A
"""

import numpy as np
import scipy.linalg
from dataclasses import dataclass


@dataclass
class ValidationResult:
    frequencies: np.ndarray
    eigenvectors: np.ndarray
    mode_shapes: np.ndarray
    nodes: np.ndarray


class CantileverBeamFEM:
    """
    FEM Model for Cantilever (Fixed-Free) Beam
    Uses Euler-Bernoulli beam theory for validation against Das (2023)
    """

    def __init__(self, length, width, height, E, rho, n_elements=50):
        """
        Initialize the beam FEM model.

        Args:
            length: Beam length in meters
            width: Beam width (b) in meters
            height: Beam height (h) in meters
            E: Young's modulus in Pa
            rho: Density in kg/m³
            n_elements: Number of finite elements
        """
        self.L = length
        self.b = width
        self.h = height
        self.E = E
        self.rho = rho
        self.n_elements = n_elements

        # Cross-section properties
        self.A = self.b * self.h
        self.I = (self.b * self.h**3) / 12

        # Discretization
        self.node_coords = np.linspace(0, self.L, self.n_elements + 1)
        self.element_lengths = np.diff(self.node_coords)
        self.n_nodes = len(self.node_coords)
        self.dof_per_node = 2  # vertical displacement (v) and rotation (theta)
        self.total_dof = self.n_nodes * self.dof_per_node

    def _element_matrices(self, le):
        """
        Compute local stiffness (k) and mass (m) matrices for a beam element.
        """
        E = self.E
        I = self.I
        A = self.A
        rho = self.rho

        # Euler-Bernoulli Stiffness Matrix
        k_factor = (E * I) / (le**3)
        k = k_factor * np.array([
            [12, 6*le, -12, 6*le],
            [6*le, 4*le**2, -6*le, 2*le**2],
            [-12, -6*le, 12, -6*le],
            [6*le, 2*le**2, -6*le, 4*le**2]
        ])

        # Consistent Mass Matrix
        m_factor = (rho * A * le) / 420
        m = m_factor * np.array([
            [156, 22*le, 54, -13*le],
            [22*le, 4*le**2, 13*le, -3*le**2],
            [54, 13*le, 156, -22*le],
            [-13*le, -3*le**2, -22*le, 4*le**2]
        ])

        return k, m

    def assemble_global_matrices(self):
        """
        Assemble global stiffness [K] and mass [M] matrices.
        """
        K = np.zeros((self.total_dof, self.total_dof))
        M = np.zeros((self.total_dof, self.total_dof))

        for i in range(self.n_elements):
            le = self.element_lengths[i]
            k_local, m_local = self._element_matrices(le)

            dofs = [2*i, 2*i+1, 2*i+2, 2*i+3]

            for row_local in range(4):
                global_row = dofs[row_local]
                for col_local in range(4):
                    global_col = dofs[col_local]
                    K[global_row, global_col] += k_local[row_local, col_local]
                    M[global_row, global_col] += m_local[row_local, col_local]

        return K, M

    def apply_cantilever_bc(self, K, M):
        """
        Apply Fixed-Free (cantilever) boundary conditions.
        Fixed at Node 0: DOF 0 (v) and DOF 1 (theta) are constrained.
        Free at Node N: No constraints.
        """
        fixed_dofs = [0, 1]  # Only fix the first node

        free_dofs = np.ones(self.total_dof, dtype=bool)
        free_dofs[fixed_dofs] = False

        K_reduced = K[np.ix_(free_dofs, free_dofs)]
        M_reduced = M[np.ix_(free_dofs, free_dofs)]

        return K_reduced, M_reduced, free_dofs

    def solve_eigenvalues(self):
        """
        Solve the generalized eigenvalue problem: [K]{u} = omega^2 [M]{u}
        """
        K, M = self.assemble_global_matrices()
        K_red, M_red, free_dofs_mask = self.apply_cantilever_bc(K, M)

        eigenvalues, eigenvectors_red = scipy.linalg.eigh(K_red, M_red)

        eigenvalues = np.maximum(eigenvalues, 0)
        frequencies = np.sqrt(eigenvalues) / (2 * np.pi)

        num_modes = len(frequencies)
        full_eigenvectors = np.zeros((self.total_dof, num_modes))
        free_indices = np.where(free_dofs_mask)[0]

        for i in range(num_modes):
            full_eigenvectors[free_indices, i] = eigenvectors_red[:, i]

        mode_shapes = full_eigenvectors[0::2, :]

        return ValidationResult(
            frequencies=frequencies,
            eigenvectors=full_eigenvectors,
            mode_shapes=mode_shapes,
            nodes=self.node_coords
        )


def theoretical_cantilever_frequency(n, L, E, I, rho, A):
    """
    Calculate theoretical natural frequency for cantilever beam.

    f_n = (lambda_n^2 / (2*pi*L^2)) * sqrt(E*I / (rho*A))

    where lambda_n are eigenvalues for cantilever: cos(lambda)*cosh(lambda) + 1 = 0
    First 5 values: 1.8751, 4.6941, 7.8548, 10.9955, 14.1372
    """
    # First 5 eigenvalues for cantilever beam
    lambda_values = [1.8751, 4.6941, 7.8548, 10.9955, 14.1372]

    if n > len(lambda_values):
        raise ValueError(f"Only first {len(lambda_values)} modes available")

    lambda_n = lambda_values[n - 1]

    # Standard formula: omega_n = (lambda_n/L)^2 * sqrt(EI/(rho*A))
    omega_n = (lambda_n / L)**2 * np.sqrt(E * I / (rho * A))
    f_n = omega_n / (2 * np.pi)

    return f_n


def run_validation():
    """
    Run 3-way validation against Das (2023) ANSYS results.
    """
    print("=" * 80)
    print("FEM VALIDATION AGAINST DAS (2023) ANSYS RESULTS")
    print("=" * 80)
    print()

    # Das (2023) Validation Case A: h/L = 1/48, Aluminum Al 7075 Cantilever
    L = 1.2       # m
    b = 0.025     # m
    h = 0.025     # m
    E = 72e9      # Pa (72 GPa)
    rho = 2810    # kg/m³

    print("Beam Configuration (Das 2023, Case A, h/L = 1/48):")
    print(f"  Length: {L} m")
    print(f"  Width: {b} m")
    print(f"  Height: {h} m")
    print(f"  h/L ratio: {h/L:.4f} (thin beam)")
    print(f"  Material: Aluminum Al 7075")
    print(f"  E = {E/1e9} GPa, rho = {rho} kg/m³")
    print(f"  Boundary: Fixed-Free (Cantilever)")
    print()

    # Das (2023) ANSYS Results (Table 3)
    das_ansys = [13.552, 84.816, 237.030, 463.260, 763.260]

    # Das (2023) EBT FEM Results (Table 3)
    das_ebt = [13.555, 84.909, 237.570, 465.060, 767.780]

    # Run our Python FEM
    beam = CantileverBeamFEM(L, b, h, E, rho, n_elements=50)
    result = beam.solve_eigenvalues()
    our_fem = result.frequencies[:5]

    # Calculate theoretical values
    I = (b * h**3) / 12
    A = b * h
    theoretical = [theoretical_cantilever_frequency(n, L, E, I, rho, A) for n in range(1, 6)]

    # Print comparison table
    print("-" * 80)
    print("THREE-WAY VALIDATION COMPARISON")
    print("-" * 80)
    print(f"{'Mode':<6} {'Das ANSYS':>12} {'Das EBT FEM':>12} {'Theoretical':>12} {'Our FEM':>12} {'Error vs ANSYS':>15}")
    print(f"{'':6} {'(Hz)':>12} {'(Hz)':>12} {'(Hz)':>12} {'(Hz)':>12} {'(%)':>15}")
    print("-" * 80)

    for i in range(5):
        error = abs(our_fem[i] - das_ansys[i]) / das_ansys[i] * 100
        print(f"{i+1:<6} {das_ansys[i]:>12.3f} {das_ebt[i]:>12.3f} {theoretical[i]:>12.3f} {our_fem[i]:>12.3f} {error:>15.3f}")

    print("-" * 80)
    print()

    # Calculate average error
    avg_error = np.mean([abs(our_fem[i] - das_ansys[i]) / das_ansys[i] * 100 for i in range(5)])
    max_error = np.max([abs(our_fem[i] - das_ansys[i]) / das_ansys[i] * 100 for i in range(5)])

    print("VALIDATION SUMMARY:")
    print(f"  Average Error vs ANSYS: {avg_error:.3f}%")
    print(f"  Maximum Error vs ANSYS: {max_error:.3f}%")
    print()

    if max_error < 1.0:
        print("RESULT: VALIDATION PASSED (All modes within 1% of ANSYS reference)")
    else:
        print("RESULT: VALIDATION REQUIRES REVIEW")

    print()
    print("=" * 80)

    # Return results for documentation
    return {
        'das_ansys': das_ansys,
        'das_ebt': das_ebt,
        'theoretical': theoretical,
        'our_fem': list(our_fem),
        'errors': [abs(our_fem[i] - das_ansys[i]) / das_ansys[i] * 100 for i in range(5)],
        'avg_error': avg_error,
        'max_error': max_error
    }


def generate_validation_table_markdown():
    """
    Generate markdown table for thesis documentation.
    """
    results = run_validation()

    print("\n\nMARKDOWN TABLE FOR DOCUMENTATION:")
    print("=" * 80)
    print()
    print("| Mode | Das (2023) ANSYS (Hz) | Das (2023) EBT FEM (Hz) | Theoretical (Hz) | Our Python FEM (Hz) | Error vs ANSYS |")
    print("|------|----------------------|------------------------|-----------------|---------------------|----------------|")

    for i in range(5):
        print(f"| {i+1} | {results['das_ansys'][i]:.3f} | {results['das_ebt'][i]:.3f} | {results['theoretical'][i]:.3f} | {results['our_fem'][i]:.3f} | {results['errors'][i]:.3f}% |")

    print()


if __name__ == "__main__":
    generate_validation_table_markdown()
