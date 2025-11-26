import numpy as np
import scipy.linalg
from dataclasses import dataclass

@dataclass
class SimulationResult:
    frequencies: np.ndarray
    eigenvectors: np.ndarray
    mode_shapes: np.ndarray
    nodes: np.ndarray

class BeamFEM:
    """
    Finite Element Model for a Fixed-Fixed Reinforced Concrete Beam.
    Simulates the effect of corrosion on natural frequencies using the Stiffness Reduction Method.
    """

    def __init__(self, length, width, depth, concrete_strength_mpa, density=2400, n_elements=20, damage_type='none', damage_params=None):
        """
        Initialize the Beam FEM model.

        Args:
            length (float): Length of the beam in meters.
            width (float): Width of the beam in meters.
            depth (float): Depth of the beam in meters.
            concrete_strength_mpa (float): Compressive strength of concrete in MPa.
            density (float): Density of reinforced concrete in kg/m^3. Default is 2400.
            n_elements (int): Number of finite elements to discretize the beam. Default is 20.
            damage_type (str): Type of damage ('none', 'corrosion', 'crack', 'random').
            damage_params (dict): Parameters for the damage type.
                - For 'corrosion': {'level': float (0-100)}
                - For 'crack': {'location': float (0-L), 'severity': float (0-1), 'width': float (m)}
                - For 'random': {'count': int, 'severity_range': tuple}
        """
        self.L = length
        self.b = width
        self.h = depth
        self.fc = concrete_strength_mpa
        self.rho = density
        self.n_elements = n_elements
        self.damage_type = damage_type
        self.damage_params = damage_params if damage_params else {}
        
        # Derived properties
        self.E = 4700 * np.sqrt(self.fc) * 1e6  # Young's Modulus in Pa
        self.A = self.b * self.h
        self.I_original = (self.b * self.h**3) / 12
        
        # Discretization
        self.node_coords = np.linspace(0, self.L, self.n_elements + 1)
        self.element_lengths = np.diff(self.node_coords)
        self.n_nodes = len(self.node_coords)
        self.dof_per_node = 2
        self.total_dof = self.n_nodes * self.dof_per_node
        
        # Calculate stiffness profile (I_effective for each element)
        self.I_effective = self._calculate_stiffness_profile()

    def _calculate_stiffness_profile(self):
        """
        Calculates the Effective Moment of Inertia for each element based on the damage scenario.
        Returns:
            np.ndarray: Array of I_effective for each element.
        """
        I_profile = np.full(self.n_elements, self.I_original)
        
        if self.damage_type == 'none':
            return I_profile
            
        elif self.damage_type == 'corrosion':
            # Uniform stiffness reduction
            level = self.damage_params.get('level', 0)
            alpha = min(1.6 * (level / 100.0), 0.9)
            I_profile *= (1 - alpha)
            
        elif self.damage_type == 'crack':
            # Localized damage
            loc = self.damage_params.get('location', self.L / 2)
            severity = self.damage_params.get('severity', 0.5) # 0.5 means 50% stiffness loss
            width = self.damage_params.get('width', self.L / 10) # Width of the cracked zone
            
            # Find elements within the crack zone
            elem_centers = self.node_coords[:-1] + self.element_lengths / 2
            mask = np.abs(elem_centers - loc) <= (width / 2)
            
            I_profile[mask] *= (1 - severity)
            
        elif self.damage_type == 'random':
            # Random cracks
            count = self.damage_params.get('count', 3)
            sev_min, sev_max = self.damage_params.get('severity_range', (0.1, 0.4))
            
            indices = np.random.choice(self.n_elements, count, replace=False)
            severities = np.random.uniform(sev_min, sev_max, count)
            
            for idx, sev in zip(indices, severities):
                I_profile[idx] *= (1 - sev)
                
        return I_profile

    def _element_matrices(self, le, I_elem):
        """
        Computes local stiffness (k) and mass (m) matrices for a beam element.
        """
        E = self.E
        A = self.A
        rho = self.rho
        
        # Stiffness Matrix (k)
        k_factor = (E * I_elem) / (le**3)
        k = k_factor * np.array([
            [12, 6*le, -12, 6*le],
            [6*le, 4*le**2, -6*le, 2*le**2],
            [-12, -6*le, 12, -6*le],
            [6*le, 2*le**2, -6*le, 4*le**2]
        ])
        
        # Mass Matrix (m) - Assumed constant density even if cracked
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
        Assembles local matrices into Global Stiffness [K] and Global Mass [M].
        """
        K = np.zeros((self.total_dof, self.total_dof))
        M = np.zeros((self.total_dof, self.total_dof))
        
        for i in range(self.n_elements):
            le = self.element_lengths[i]
            I_elem = self.I_effective[i]
            
            k_local, m_local = self._element_matrices(le, I_elem)
            
            dofs = [2*i, 2*i+1, 2*i+2, 2*i+3]
            
            for row_local in range(4):
                global_row = dofs[row_local]
                for col_local in range(4):
                    global_col = dofs[col_local]
                    
                    K[global_row, global_col] += k_local[row_local, col_local]
                    M[global_row, global_col] += m_local[row_local, col_local]
                    
        return K, M

    def apply_boundary_conditions(self, K, M):
        """
        Applies Fixed-Fixed boundary conditions.
        Removes DOFs corresponding to fixed ends (Node 0 and Node N).
        """
        # Fixed at Node 0: DOF 0 (v) and 1 (theta) are 0
        # Fixed at Node N: DOF 2*N (v) and 2*N+1 (theta) are 0
        
        # Indices to remove (must be sorted descending to avoid index shift issues if popping, 
        # but for numpy masking/deletion we just need the list)
        fixed_dofs = [0, 1, self.total_dof-2, self.total_dof-1]
        
        # Create a mask of free DOFs
        free_dofs = np.ones(self.total_dof, dtype=bool)
        free_dofs[fixed_dofs] = False
        
        # Filter matrices using open mesh (ix_) or boolean indexing
        K_reduced = K[np.ix_(free_dofs, free_dofs)]
        M_reduced = M[np.ix_(free_dofs, free_dofs)]
        
        return K_reduced, M_reduced, free_dofs

    def solve_eigenvalues(self):
        """
        Solves the generalized eigenvalue problem: [K]{u} = w^2 [M]{u}
        
        Returns:
            SimulationResult object containing frequencies and mode shapes.
        """
        K, M = self.assemble_global_matrices()
        K_red, M_red, free_dofs_mask = self.apply_boundary_conditions(K, M)
        
        # Solve generalized eigenvalue problem
        # eigh is for Hermitian/Symmetric matrices, which K and M are.
        eigenvalues, eigenvectors_red = scipy.linalg.eigh(K_red, M_red)
        
        # Convert eigenvalues to frequencies (Hz)
        # omega^2 = eigenvalue
        # f = omega / (2*pi) = sqrt(eigenvalue) / (2*pi)
        
        # Filter out tiny negative values due to numerical noise
        eigenvalues = np.maximum(eigenvalues, 0)
        
        frequencies = np.sqrt(eigenvalues) / (2 * np.pi)
        
        # Reconstruct full eigenvectors (add zeros for fixed DOFs)
        num_modes = len(frequencies)
        full_eigenvectors = np.zeros((self.total_dof, num_modes))
        
        # Place the computed values back into the free positions
        # We need to map the reduced indices back to global indices
        free_indices = np.where(free_dofs_mask)[0]
        
        for i in range(num_modes):
            full_eigenvectors[free_indices, i] = eigenvectors_red[:, i]
            
        # Extract mode shapes (vertical displacement only, which are the even indices: 0, 2, 4...)
        # DOFs are [v0, theta0, v1, theta1, ...]
        # We want v at each node.
        mode_shapes = full_eigenvectors[0::2, :]
        
        return SimulationResult(
            frequencies=frequencies,
            eigenvectors=full_eigenvectors,
            mode_shapes=mode_shapes,
            nodes=self.node_coords
        )

if __name__ == "__main__":
    # Quick test
    print("--- Normal Beam ---")
    beam = BeamFEM(length=3.0, width=0.3, depth=0.45, concrete_strength_mpa=30, damage_type='none')
    result = beam.solve_eigenvalues()
    print(f"Mode 1 Freq: {result.frequencies[0]:.4f} Hz")
    
    print("\n--- Corroded Beam (10%) ---")
    beam = BeamFEM(length=3.0, width=0.3, depth=0.45, concrete_strength_mpa=30, damage_type='corrosion', damage_params={'level': 10})
    result = beam.solve_eigenvalues()
    print(f"Mode 1 Freq: {result.frequencies[0]:.4f} Hz")
    
    print("\n--- Cracked Beam (Mid-span) ---")
    beam = BeamFEM(length=3.0, width=0.3, depth=0.45, concrete_strength_mpa=30, damage_type='crack', damage_params={'location': 1.5, 'severity': 0.5, 'width': 0.3})
    result = beam.solve_eigenvalues()
    print(f"Mode 1 Freq: {result.frequencies[0]:.4f} Hz")
