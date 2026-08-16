from pathlib import Path

SEED = 42
N_FOLDS = 5
N_BOOTSTRAP = 100
DEV_SIZE = 800
APDL_SOLVE_SECONDS = 360.0

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_DIR / "data" / "rc_beam_ansys_dataset.xlsx"
OUTPUT_DIR = PROJECT_DIR / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"
TABLES_DIR = OUTPUT_DIR / "tables"
MODELS_DIR = OUTPUT_DIR / "models"
LOGS_DIR = OUTPUT_DIR / "logs"

FEATURE_COLS = ["L_mm", "b_mm", "h_mm", "fc_MPa", "rho_percent",
                "crack1_depth_mm", "crack2_depth_mm"]
FAMILY_COL = "family"
TARGET_COLS = ["f1_hz", "f2_hz", "f3_hz", "f4_hz", "f5_hz"]

SKIPPED_FIELDS = {
    "case_id": "administrative identifier, not a predictor",
    "dataset_role": "administrative; all rows are PRIMARY",
    "length_class": "derived from L_mm; used only for the extrapolation test",
    "combination_code": "redundant with family (same values)",
    "combination_name": "human-readable form of combination_code",
    "crack1_type": "redundant with family (FF=Flexural, SS=Shear)",
    "crack2_type": "redundant with family (FF=Flexural, SS=Shear)",
    "Ec_MPa": "deterministically derived from fc_MPa via ACI 318-19",
    "As_mm2": "derived from rho_percent, b_mm and h_mm",
    "equivalent_diameter_mm": "derived from As_mm2",
    "Concrete Cover": "constant 40 mm for all cases; zero information",
    "crack1_angle_deg": "constant within family (FF=90, SS=45); no information beyond family",
    "crack2_angle_deg": "constant within family (FF=90, SS=135); no information beyond family",
    "slenderness_L_h": "derived from L_mm and h_mm",
    "width_depth_b_h": "derived from b_mm and h_mm",
    "mesh_size_mm": "solver metadata, not a physical input",
    "length_divisions": "solver metadata",
    "height_divisions": "solver metadata",
    "supports": "constant (all fixed-fixed)",
    "concrete_element": "solver metadata",
    "rebar_element": "solver metadata",
    "modes_extracted": "solver metadata",
    "preanalysis_qc": "quality-control metadata",
    "bend_1_mode": "leakage: solver index derived from solved frequencies",
    "bend_2_mode": "leakage: solver index derived from solved frequencies",
    "bend_3_mode": "leakage: solver index derived from solved frequencies",
    "bend_4_mode": "leakage: solver index derived from solved frequencies",
    "bend_5_mode": "leakage: solver index derived from solved frequencies",
}

# Crack locations are not separate columns in the spreadsheet; they are
# encoded in combination_name and are constant per family:
#   FF family: cracks at 0.45L and 0.55L (flexural)
#   SS family: cracks at 0.1L and 0.9L (shear)
# Location therefore carries no information beyond the family label and is
# represented by the family categorical only (per Ch. 3.10.1).
