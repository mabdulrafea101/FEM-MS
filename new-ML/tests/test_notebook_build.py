import json
import subprocess
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent


def test_build_script_generates_notebook():
    result = subprocess.run(["../.venv12/bin/python", "build_notebook.py"],
                            cwd=PROJECT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    nb_path = PROJECT / "model_training_ansys.ipynb"
    assert nb_path.exists()
    nb = json.loads(nb_path.read_text())
    # 1 header + 18 stages x (markdown + code) = 37 cells
    assert len(nb["cells"]) >= 37


def test_notebook_contains_all_stages():
    nb = json.loads((PROJECT / "model_training_ansys.ipynb").read_text())
    text = "\n".join("".join(c["source"]) for c in nb["cells"])
    for marker in ["Feature Selection", "Bootstrap", "Hyperparameter",
                   "Das", "prediction interface", "ANOVA"]:
        assert marker in text, marker
