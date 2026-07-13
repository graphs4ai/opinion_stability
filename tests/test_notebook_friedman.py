import json
from pathlib import Path


NOTEBOOK = Path(__file__).resolve().parents[1] / "analises.ipynb"


def _notebook_source() -> str:
    with NOTEBOOK.open(encoding="utf-8") as handle:
        notebook = json.load(handle)
    return "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )


def test_notebook_defines_paired_friedman_helpers():
    source = _notebook_source()

    assert "def friedman_kendall_w(" in source
    assert "def friedman_by_block(" in source
    assert "stats.friedmanchisquare" in source
    assert "kendall_w" in source


def test_notebook_reports_friedman_tables_for_paired_comparisons():
    source = _notebook_source()

    assert "model_stability_friedman_tests" in source
    assert "scenario_variance_friedman_tests" in source
    assert "cumulative_shift_friedman_tests" in source
    assert "Model-level stability Friedman tests" in source
    assert "Scenario-specific variance Friedman tests" in source
    assert "Cumulative-shift Friedman tests" in source
