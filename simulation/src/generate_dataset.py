import numpy as np
import pandas as pd
from scipy.stats import qmc
import logging
import os
from fem_core import BeamFEM

# Setup logging
log_dir = os.path.join(os.path.dirname(__file__), '../logs')
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(log_dir, 'generation.log'),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def generate_dataset():
    print("Starting dataset generation...")
    logging.info("Starting dataset generation")
    
    # 1. Latin Hypercube Sampling for Pristine Data
    # Variables: Length, Width, Depth, Conc_Strength
    # Ranges:
    # L: [3.0, 8.0] m
    # b: [0.2, 0.5] m
    # h: [0.3, 0.8] m
    # fc: [25, 50] MPa
    
    # 2. Generate Scenarios
    # We will generate 3000 samples in total
    # 1000 Normal
    # 700 Corroded
    # 700 Cracked (Single Crack)
    # 600 Broken (Multiple Cracks)
    
    scenarios = [
        {'type': 'none', 'count': 1000},
        {'type': 'corrosion', 'count': 700},
        {'type': 'crack', 'count': 700},
        {'type': 'random', 'count': 600}
    ]
    
    total_samples = sum(s['count'] for s in scenarios)
    
    # Generate base parameters for all samples using LHS
    sampler = qmc.LatinHypercube(d=4, seed=42)
    sample = sampler.random(n=total_samples)
    
    l_bounds = [3.0, 8.0]
    b_bounds = [0.2, 0.5]
    h_bounds = [0.3, 0.8]
    fc_bounds = [25, 50]
    
    lengths = qmc.scale(sample[:, 0:1], l_bounds[0], l_bounds[1]).flatten()
    widths = qmc.scale(sample[:, 1:2], b_bounds[0], b_bounds[1]).flatten()
    depths = qmc.scale(sample[:, 2:3], h_bounds[0], h_bounds[1]).flatten()
    strengths = qmc.scale(sample[:, 3:4], fc_bounds[0], fc_bounds[1]).flatten()
    
    data = []
    current_idx = 0
    
    for scenario in scenarios:
        sType = scenario['type']
        count = scenario['count']
        print(f"Simulating {count} {sType} beams...")
        
        for i in range(count):
            idx = current_idx + i
            L = lengths[idx]
            b = widths[idx]
            h = depths[idx]
            fc = strengths[idx]
            
            damage_params = {}
            severity_metric = 0.0
            
            if sType == 'corrosion':
                # Random corrosion 5-30%
                level = np.random.uniform(5, 30)
                damage_params = {'level': level}
                severity_metric = level
                
            elif sType == 'crack':
                # Random location and severity
                loc = np.random.uniform(0.1*L, 0.9*L)
                sev = np.random.uniform(0.1, 0.7)
                width = np.random.uniform(0.1, 0.5)
                damage_params = {'location': loc, 'severity': sev, 'width': width}
                severity_metric = sev * 100 # Scale to 0-100 for consistency
                
            elif sType == 'random':
                # Multiple cracks
                cnt = np.random.randint(2, 5)
                damage_params = {'count': cnt, 'severity_range': (0.1, 0.5)}
                # Metric is average severity * count (rough proxy)
                severity_metric = 0.3 * cnt * 100 
            
            try:
                beam = BeamFEM(L, b, h, fc, damage_type=sType, damage_params=damage_params)
                res = beam.solve_eigenvalues()
                
                row = {
                    'ID': idx,
                    'Length': L,
                    'Width': b,
                    'Depth': h,
                    'Conc_Strength': fc,
                    'Damage_Type': sType,
                    'Damage_Severity': severity_metric, # Normalized-ish metric
                    'Freq_Mode_1': res.frequencies[0],
                    'Freq_Mode_2': res.frequencies[1]
                }
                data.append(row)
                
            except Exception as e:
                logging.error(f"Error in sample {idx}: {e}")
                
        current_idx += count

    # Save to CSV
    df = pd.DataFrame(data)
    output_path = os.path.join(os.path.dirname(__file__), '../data/beam_vibration_dataset.csv')
    df.to_csv(output_path, index=False)
    
    print(f"Dataset saved to {output_path}")
    logging.info("Dataset generation complete")

if __name__ == "__main__":
    generate_dataset()
