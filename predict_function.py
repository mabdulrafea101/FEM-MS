
import numpy as np
import joblib
import os
from pathlib import Path

# Load artifacts
# Assuming this script is run from the project root or we can find paths relative to it
# If run as a script
BASE_DIR = Path(__file__).parent.resolve() if '__file__' in locals() else Path.cwd()
MODELS_DIR = BASE_DIR / 'simulation' / 'models'

# Helper to load models (optimistic loading)
scaler = None
model_m1 = None
model_m2 = None

def load_resources():
    global scaler, model_m1, model_m2
    try:
        if not scaler:
            s_path = MODELS_DIR / 'scaler.pkl'
            if s_path.exists():
                scaler = joblib.load(s_path)
            else:
                print(f"Scaler not found at {s_path}")
                return False
                
        if not model_m1:
            # Try to find the Mode 1 model
            # We look for best_model_CatBoost.pkl based on the notebook logic
            m1_path = MODELS_DIR / 'best_model_CatBoost.pkl'
            if m1_path.exists():
                model_m1 = joblib.load(m1_path)
            else:
                # Fallback search for any best_model that isn't mode2
                candidates = list(MODELS_DIR.glob('best_model_*.pkl'))
                candidates = [c for c in candidates if 'mode2' not in c.name]
                if candidates:
                    model_m1 = joblib.load(candidates[0])
                else:
                    print(f"Mode 1 model not found in {MODELS_DIR}")
                    return False
                    
        if not model_m2:
            m2_path = MODELS_DIR / 'best_model_mode2.pkl'
            if m2_path.exists():
                model_m2 = joblib.load(m2_path)
            else:
                print(f"Mode 2 model not found at {m2_path}")
                return False
            
        return True
    except Exception as e:
        print(f"Error loading models: {e}")
        return False

def predict_frequency(length, width, depth, conc_strength, damage_severity):
    """
    Predict Mode 1 and Mode 2 natural frequencies for given beam parameters.
    
    Parameters:
    -----------
    length : float - Beam length (m)
    width : float - Beam width (m)
    depth : float - Beam depth (m)
    conc_strength : float - Concrete strength (MPa)
    damage_severity : float - Damage severity (0-100%)
    
    Returns:
    --------
    dict - Predicted frequencies: {'Mode_1': float, 'Mode_2': float}
    """
    # Ensure resources are loaded
    if scaler is None or model_m1 is None or model_m2 is None:
        if not load_resources():
            print("Failed to load models. Cannot predict.")
            return {'Mode_1': 0.0, 'Mode_2': 0.0}

    # Create input array
    input_data = np.array([[length, width, depth, conc_strength, damage_severity]])
    
    # Scale input
    input_scaled = scaler.transform(input_data)
    
    # Predict
    pred_m1 = model_m1.predict(input_scaled)[0]
    pred_m2 = model_m2.predict(input_scaled)[0]
    
    return {'Mode_1': pred_m1, 'Mode_2': pred_m2}

# Example Usage Block
if __name__ == "__main__":
    print("Loading models and testing prediction...")
    success = load_resources()
    if success:
        test_length = 4.0
        test_width = 0.3
        test_depth = 0.5
        test_strength = 35
        test_damage = 10
        
        preds = predict_frequency(test_length, test_width, test_depth, test_strength, test_damage)
        
        print("\n" + "="*80)
        print("PREDICTION TEST")
        print("="*80)
        print(f"Input Parameters:")
        print(f"  Length: {test_length} m")
        print(f"  Width: {test_width} m")
        print(f"  Depth: {test_depth} m")
        print(f"  Concrete Strength: {test_strength} MPa")
        print(f"  Damage Severity: {test_damage}%")
        print(f"\nPredicted Frequencies:")
        print(f"  Mode 1: {preds['Mode_1']:.2f} Hz")
        print(f"  Mode 2: {preds['Mode_2']:.2f} Hz")
    else:
        print("Failed to load models. Please ensure training scripts have been run.")
