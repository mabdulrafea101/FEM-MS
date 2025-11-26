import sys
import os
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

from fem_core import BeamFEM

def theoretical_frequency(L, E, I, A, rho, mode=1):
    """
    Calculates theoretical natural frequency for a Fixed-Fixed beam.
    Formula: f = (beta^2 / (2 * pi * L^2)) * sqrt(EI / (rho * A))
    """
    # Betas for Fixed-Fixed beams
    betas = [4.73004074, 7.85320462, 10.9956079, 14.1371655]
    
    if mode < 1 or mode > 4:
        raise ValueError("Mode must be between 1 and 4")
        
    beta = betas[mode-1]
    
    term1 = (beta**2) / (2 * np.pi * L**2)
    term2 = np.sqrt((E * I) / (rho * A))
    
    return term1 * term2

def run_test_a():
    print("--- Test A: Theoretical Validation (Standard Steel Beam) ---")
    # Parameters for a standard steel beam
    L = 3.0 # m
    b = 0.1 # m
    h = 0.1 # m
    
    # Steel properties
    E_steel = 200e9 # Pa
    rho_steel = 7850 # kg/m^3
    
    # We need to hack BeamFEM to use steel properties instead of concrete
    # BeamFEM calculates E from fc. We will override it.
    
    # Initialize with dummy concrete values
    beam = BeamFEM(length=L, width=b, depth=h, concrete_strength_mpa=30, corrosion_level=0, density=rho_steel)
    
    # Override E
    beam.E = E_steel
    
    # Solve
    result = beam.solve_eigenvalues()
    fem_freq = result.frequencies[0]
    
    # Theoretical
    I = (b * h**3) / 12
    A = b * h
    theo_freq = theoretical_frequency(L, E_steel, I, A, rho_steel, mode=1)
    
    error = abs(fem_freq - theo_freq) / theo_freq * 100
    
    print(f"FEM Frequency: {fem_freq:.4f} Hz")
    print(f"Theoretical Frequency: {theo_freq:.4f} Hz")
    print(f"Error: {error:.4f}%")
    
    if error < 1.0:
        print("Test A PASSED")
    else:
        print("Test A FAILED")
    print("")

def run_test_b():
    print("--- Test B: Experimental Validation (Sivasuriyan) ---")
    print("NOTE: Please update the dimensions below with values from the Sivasuriyan paper.")
    
    # Placeholder values (User to update)
    L = 2.0 
    b = 0.15
    h = 0.20
    fc = 30 # MPa
    
    # Sivasuriyan might use Simply Supported. 
    # If so, we would need to adjust BCs in fem_core or subclass it.
    # For now, we assume Fixed-Fixed as per project plan focus, 
    # or the user needs to modify this test to check Simply Supported if that's what the paper used.
    
    beam = BeamFEM(length=L, width=b, depth=h, concrete_strength_mpa=fc, corrosion_level=0)
    result = beam.solve_eigenvalues()
    
    print(f"FEM Mode 1 Frequency: {result.frequencies[0]:.4f} Hz")
    print("Compare this with the value from the paper.")
    print("")

if __name__ == "__main__":
    run_test_a()
    run_test_b()
