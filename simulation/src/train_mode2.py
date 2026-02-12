
import pandas as pd
import numpy as np
import joblib
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def train_mode2_model():
    logger.info("Starting Mode 2 Model Training...")
    
    # Paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, '../data/beam_vibration_dataset.csv')
    models_dir = os.path.join(base_dir, '../models')
    
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
        
    # 1. Load Data
    if not os.path.exists(data_path):
        logger.error(f"Dataset not found at {data_path}")
        return

    df = pd.read_csv(data_path)
    logger.info(f"Dataset loaded: {df.shape}")
    
    feature_cols = ['Length', 'Width', 'Depth', 'Conc_Strength', 'Damage_Severity']
    target_col = 'Freq_Mode_2'
    
    X = df[feature_cols]
    y = df[target_col]
    
    # 2. Train/Test Split
    # Must use same random state as Mode 1 training to facilitate fair comparison if needed
    RANDOM_STATE = 42
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, shuffle=True
    )
    
    # 3. Scaling
    # We ideally want to use the EXACT same scaler as Mode 1 to ensure consistency.
    # We can try to load it, or refit it (should be identical if data is same).
    scaler_path = os.path.join(models_dir, 'scaler.pkl')
    if os.path.exists(scaler_path):
        logger.info("Loading existing scaler...")
        scaler = joblib.load(scaler_path)
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)
    else:
        logger.info("Fitting new scaler (scaler.pkl not found)...")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        joblib.dump(scaler, scaler_path)
        
    # 4. Train CatBoost
    logger.info("Training CatBoostRegressor for Mode 2...")
    model = CatBoostRegressor(
        iterations=200,
        depth=8,
        learning_rate=0.1,
        random_state=RANDOM_STATE,
        verbose=False,
        allow_writing_files=False
    )
    
    model.fit(X_train_scaled, y_train)
    
    # 5. Evaluate
    y_pred = model.predict(X_test_scaled)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    logger.info("Mode 2 Performance:")
    logger.info(f"  R2: {r2:.4f}")
    logger.info(f"  MAE: {mae:.4f}")
    logger.info(f"  RMSE: {rmse:.4f}")
    
    # 6. Save Model
    output_path = os.path.join(models_dir, 'best_model_mode2.pkl')
    joblib.dump(model, output_path)
    logger.info(f"Mode 2 model saved to: {output_path}")

if __name__ == "__main__":
    train_mode2_model()
