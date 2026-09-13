"""As figuras precisam acomodar cinco condições, não três.

Vários layouts estavam dimensionados para o número antigo de condições. O caso
mais grave era plot_figure_7, que criava quatro eixos e usava zip contra a
lista de painéis: com seis painéis, E2 e E3 sumiriam silenciosamente do PDF.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from src.analysis.data import build_trajectories
from src.analysis.manifest import EXPERIMENT_ORDER
from src.analysis.model_size_figure import create_model_size_figures

MODELOS = [
    "google/gemma-3-4b-it", "google/gemma-4-31B-it",
    "meta-llama/Llama-4-Scout-17B-16E-Instruct", "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "Qwen/Qwen3-30B-A3B", "Qwen/Qwen3-14B",
    "mistralai/Mistral-Small-3.2-24B-Instruct-2506", "mistralai/Mistral-Nemo-Instruct-2407",
    "deepseek-ai/DeepSeek-V3.2", "deepseek-ai/DeepSeek-V4-Flash",
]


@pytest.fixture(scope="module")
def dados():
    rng = np.random.default_rng(3)
    linhas = []
    for exp in EXPERIMENT_ORDER:
        for q in range(4):
            for cond in ("Left", "Right", "Neutral"):
                for modelo in MODELOS:
                    atual = int(rng.integers(-2, 3))
                    for rodada in range(1, 6):
                        if rodada > 1 and rng.random() < 0.25:
                            atual = int(np.clip(atual + rng.choice([-1, 1]), -2, 2))
                        linhas.append({
                            "experiment": exp, "experiment_name": exp, "intervention": "i",
                            "modelo": modelo, "condition": cond, "tendencia": cond.lower(),
                            "eixo": "Economia", "question_id": f"q{q}",
                            "rodada": rodada, "likert_valor": float(atual),
                        })
    round_level = pd.DataFrame(linhas)
    return round_level, build_trajectories(round_level)


@pytest.fixture(scope="module")
def escopo_notebook():
    """Escopo com as células de definição do notebook já executadas."""
    nb = json.loads(Path("analises.ipynb").read_text(encoding="utf-8"))
    celulas = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    escopo: dict = {"__name__": "__main__"}
    for i in (0, 1, 3, 5, 6):
        exec(compile(celulas[i], f"cell{i}", "exec"), escopo)
    escopo["SAVE_FIGURES"] = False
    escopo["SHOW_FIGURES_INLINE"] = False
    escopo["CLOSE_FIGURES_AFTER_RENDER"] = False
    return escopo


def _ultima_figura():
    return plt.figure(plt.get_fignums()[-1])


@pytest.mark.parametrize("nome", ["plot_figure_1", "plot_figure_2", "plot_figure_5", "plot_figure_6"])
def test_figuras_de_trajetoria_renderizam_com_cinco_condicoes(escopo_notebook, dados, nome):
    plt.close("all")
    escopo_notebook[nome](dados[1])
    assert plt.get_fignums(), f"{nome} não produziu figura"
    plt.close("all")


def test_figura_4_renderiza(escopo_notebook, dados):
    plt.close("all")
    escopo_notebook["plot_figure_4"](dados[1])
    assert plt.get_fignums()
    plt.close("all")


def test_figura_7_desenha_um_painel_por_condicao_mais_o_agregado(escopo_notebook, dados):
    """O bug original: quatro eixos fixos descartavam painéis em silêncio."""
    plt.close("all")
    escopo_notebook["plot_figure_7"](dados[0])
    fig = _ultima_figura()
    # Só os painéis têm título; o eixo da colorbar também carrega uma
    # collection, então contá-las levaria a um a mais.
    titulos = [ax.get_title() for ax in fig.axes if ax.get_title()]
    assert len(titulos) == len(EXPERIMENT_ORDER) + 1, titulos
    for exp in EXPERIMENT_ORDER:
        assert any(t.startswith(exp) for t in titulos), f"painel de {exp} ausente: {titulos}"
    assert any("Aggregated" in t for t in titulos)
    plt.close("all")


def test_figura_1_contrasta_baselines_e_protocolos(escopo_notebook, dados):
    plt.close("all")
    escopo_notebook["plot_figure_1"](dados[1])
    fig = _ultima_figura()
    assert len(fig.axes) >= 2, "figura 1 deve ter os painéis A e B"
    eixo_a = fig.axes[0]
    assert len(eixo_a.get_xticklabels()) == len(EXPERIMENT_ORDER)
    plt.close("all")


def test_figuras_de_tamanho_de_modelo_aceitam_seis_escopos(tmp_path, dados):
    """Cinco condições mais o agregado; o forest antes fixava quatro."""
    plt.close("all")
    stats_table, loo = create_model_size_figures(dados[1], root=tmp_path)
    assert not stats_table.empty and not loo.empty
    assert (tmp_path / "figures" / "pdf" / "figure_3.pdf").exists()
    escopos = set(stats_table["scope"])
    assert len(escopos) == len(EXPERIMENT_ORDER) + 1, escopos
    plt.close("all")
