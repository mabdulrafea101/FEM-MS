from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pipeline.config import FIGURES_DIR


def save_fig(fig, name, out_dir=FIGURES_DIR):
    out_dir = Path(out_dir) if out_dir is not None else FIGURES_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path
