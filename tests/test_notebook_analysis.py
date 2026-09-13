"""Verificações estruturais sobre ``analises.ipynb``.

Estes testes são deliberadamente rasos: eles apenas garantem que o notebook
continua sendo o driver esperado — que importa as primitivas compartilhadas em
vez de redefini-las, que todas as células compilam, e que as análises exigidas
pela revisão do ICTAI seguem sendo executadas.

O comportamento estatístico em si é testado em ``test_stats_core.py``, contra
as funções reais. Antes desta separação, as primitivas viviam dentro do JSON do
notebook e só podiam ser "testadas" por comparação de strings, o que travava
qualquer alteração cosmética.
"""

import json
from pathlib import Path

import pytest

NOTEBOOK = Path(__file__).resolve().parents[1] / "analises.ipynb"


def _code_cells() -> list[str]:
    with NOTEBOOK.open(encoding="utf-8") as handle:
        notebook = json.load(handle)
    return [
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    ]


def _notebook_source() -> str:
    return "\n".join(_code_cells())


def test_todas_as_celulas_compilam():
    for i, source in enumerate(_code_cells()):
        compile(source, f"cell{i}", "exec")


def test_notebook_importa_primitivas_em_vez_de_redefinilas():
    source = _notebook_source()
    assert "from src.analysis.stats_core import" in source
    for nome in ("friedman_by_block", "holm_adjust_pvalues",
                 "all_pairs_wilcoxon_holm", "loop_level_movement", "round1_reproducibility"):
        assert nome in source
        assert f"def {nome}(" not in source, f"{nome} deve vir de stats_core, não ser redefinida"


def test_notebook_usa_pipeline_de_dados_compartilhado():
    """load_experiment/build_trajectories vivem em src.analysis.data."""
    source = _notebook_source()
    assert "from src.analysis.data import" in source
    for nome in ("load_experiment", "build_round_level", "build_trajectories"):
        assert f"def {nome}(" not in source, f"{nome} deve vir de src.analysis.data"


def test_notebook_usa_manifesto_unico_de_experimentos():
    source = _notebook_source()
    assert "from src.analysis.manifest import" in source
    assert "EXPERIMENTS = [" not in source, "o manifesto deve vir de src.analysis.manifest"


def test_comparacao_entre_condicoes_esta_presente_e_e_executada():
    """A análise que responde às críticas convergentes do ICTAI."""
    source = _notebook_source()
    for nome in ("condition_comparison_tests", "condition_pairwise_tests",
                 "condition_exact_stability_test", "condition_descriptives"):
        assert f"def {nome}(" in source
        assert f"{nome}(" in source.split(f"def {nome}(")[1], f"{nome} definida mas nunca chamada"


def test_estabilidade_exata_usa_friedman_sobre_a_proporcao_do_loop():
    """Não deve haver teste para resposta binária: a métrica agregada é proporção."""
    source = _notebook_source()
    assert "def condition_exact_stability_test(" in source
    assert "cochran" not in source.lower(), "Cochran Q exigiria dicotomizar a proporção"
    assert "def condition_sensitivity_tests(" in source
    assert "condition_sensitivity_tests(trajectories)" in source


def test_matriz_de_transicoes_usa_apenas_rodadas_consecutivas():
    """Ligar r-1 a r+1 quando r é inválido criaria transições de dois passos."""
    source = _notebook_source()
    assert "rondas[k + 1] - rondas[k] != 1" in source


def test_reprodutibilidade_e_deriva_estao_presentes():
    source = _notebook_source()
    assert "def round1_reproducibility_tests(" in source
    assert "def provider_drift_analysis(" in source
    assert "round1_reproducibility_tests(round_level)" in source


def test_comparacao_entre_condicoes_usa_o_loop_como_unidade():
    """Blocos = statement x cenário, com os dez modelos agregados dentro."""
    source = _notebook_source()
    assert "LOOP_BLOCK_COLS" in source
    assert "LOOP_BLOCK_COLS = " not in source, "deve vir do manifesto, não ser redefinida"
    assert "loop_level_movement(traj" in source


def test_analises_por_modelo_seguem_existindo():
    source = _notebook_source()
    for nome in ("model_stability_friedman_tests", "scenario_variance_friedman_tests",
                 "cumulative_shift_friedman_tests"):
        assert nome in source


def test_analise_de_desvio_padrao_permanece_consistente():
    source = _notebook_source()
    assert "value_col='std'" in source
    assert ".agg(mean_variance=('variance', 'mean'))" not in source


def test_metrica_cumulativa_descrita_como_desvio_do_inicial():
    source = _notebook_source()
    assert "Mean cumulative deviation\\nfrom initial response" in source
    assert "Mean cumulative shift score" not in source


def test_figura_de_transicoes_likert_permanece():
    source = _notebook_source()
    assert "def build_likert_transition_matrix(" in source
    assert "def plot_figure_7(" in source
    assert "plot_figure_7(round_level)" in source


def test_celulas_de_definicao_executam_sem_erro_de_nome():
    """Executa as células que só definem coisas (imports, helpers, funções).

    Pega NameError e ImportError introduzidos pela migração das primitivas para
    src/analysis/*, sem depender dos CSVs da coleta. As células que carregam
    dados (2), plotam (4) e executam a análise (7) ficam de fora.
    """
    import matplotlib
    matplotlib.use("Agg")

    celulas = _code_cells()
    escopo: dict = {"__name__": "__main__"}
    for i in (0, 1, 3, 5, 6):
        exec(compile(celulas[i], f"cell{i}", "exec"), escopo)

    # As funções que o artigo depende precisam estar resolvíveis no escopo.
    for nome in ("plot_figure_1", "plot_figure_2", "plot_figure_7",
                 "condition_comparison_tests", "condition_pairwise_tests",
                 "round1_reproducibility_tests", "provider_drift_analysis",
                 "model_stability_friedman_tests", "build_likert_transition_matrix"):
        assert callable(escopo.get(nome)), f"{nome} não ficou definida"
