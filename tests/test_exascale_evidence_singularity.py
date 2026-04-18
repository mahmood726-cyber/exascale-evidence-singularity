import importlib.util
import json
from pathlib import Path


def load_module():
    script_path = Path(__file__).resolve().parents[1] / "simulation.py"
    spec = importlib.util.spec_from_file_location("exascale_evidence_singularity_simulation", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_main_writes_relative_exascale_outputs(tmp_path):
    module = load_module()

    result = module.main(n_simulations=5, seed=42, project_root=tmp_path)

    certification_path = tmp_path / "certification.json"
    results_path = tmp_path / "exascale_audit_results.csv"

    assert certification_path.exists()
    assert results_path.exists()

    certification = json.loads(certification_path.read_text(encoding="utf-8"))
    assert certification["system"] == "Exascale Evidence Singularity (EES-DTA)"
    assert certification["n_simulations"] == 5
    assert len(result["results"]) == 5
