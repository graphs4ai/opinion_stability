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
    assert "Scenario-specific standard deviation Friedman tests" in source
    assert "Cumulative-deviation Friedman tests" in source


def test_scenario_analysis_uses_standard_deviation_consistently():
    source = _notebook_source()

    assert "value_col='std'" in source
    assert ".agg(mean_standard_deviation=('std', 'mean'))" in source
    assert "Mean within-question standard deviation" in source
    assert ".agg(mean_variance=('variance', 'mean'))" not in source


def test_cumulative_metric_is_described_as_deviation_from_initial_response():
    source = _notebook_source()

    assert "Mean cumulative deviation\\nfrom initial response" in source
    assert "Cumulative-deviation Friedman tests" in source
    assert "Mean cumulative shift score" not in source


def test_notebook_generates_likert_transition_matrix_figure():
    source = _notebook_source()

    assert "def build_likert_transition_matrix(" in source
    assert "def plot_figure_7(" in source
    assert "Transition probability" in source
    assert "figure_7" in source
    assert "plot_figure_7(round_level)" in source


def test_likert_transition_figure_uses_requested_layout():
    source = _notebook_source()

    assert "fig.suptitle('Likert transition matrices across consecutive rounds'" not in source
    assert "cbar_ax = fig.add_subplot(gs[:, 2])" in source
    assert "cbar_ax" in source
    assert "fig.text(0.5" in source
    assert "Next-round Likert category" in source
    assert "fig.text(0.02" in source
    assert "Current-round Likert category" in source
    assert "annot_kws={'fontsize': 13}" in source
    assert "# plot_figure_2(trajectories)" in source
    assert "# plot_figure_6(trajectories)" in source
