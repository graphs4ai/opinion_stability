"""Testes comportamentais das primitivas estatísticas.

Cobrem sobretudo o que sustenta o resultado principal novo do artigo: a
comparação direta entre as condições B0/B1/E1/E2/E3.
"""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from src.analysis.stats_core import (
    all_pairs_wilcoxon_holm,
    friedman_kendall_w,
    holm_adjust_pvalues,
    kruskal_epsilon_squared,
    loop_level_movement,
    paired_wide_by_block,
    round1_reproducibility,
)


# ------------------------------------------------------- all-pairs + Holm

def test_all_pairs_wilcoxon_cobre_todos_os_pares():
    rng = np.random.default_rng(0)
    n = 60
    base = rng.normal(size=n)
    data = pd.concat([
        pd.DataFrame({"bloco": range(n), "cond": c, "mov": base + deslocamento})
        for c, deslocamento in [("B0", 0.0), ("B1", 0.5), ("E1", 1.0), ("E2", 1.5)]
    ])
    out = all_pairs_wilcoxon_holm(data, "mov", "cond", ["bloco"])
    assert len(out) == 6  # C(4,2)
    assert set(zip(out.treatment_a, out.treatment_b)) == {
        ("B0", "B1"), ("B0", "E1"), ("B0", "E2"),
        ("B1", "E1"), ("B1", "E2"), ("E1", "E2"),
    }
    # Deslocamentos constantes e grandes: todos os contrastes sobrevivem a Holm.
    assert out.significant_at_0_05.all()
    # mean_diff deve refletir a direção a - b.
    linha = out[(out.treatment_a == "B0") & (out.treatment_b == "E2")].iloc[0]
    assert linha.mean_diff == pytest.approx(-1.5)


def test_all_pairs_usa_o_mesmo_conjunto_de_blocos_em_todos_os_pares():
    """Um bloco faltante em UMA condição deve sair de TODOS os contrastes."""
    linhas = []
    for b in range(20):
        for c in ["B0", "B1", "E1"]:
            if b == 0 and c == "E1":
                continue  # bloco incompleto
            linhas.append({"bloco": b, "cond": c, "mov": b * 0.1 + (c == "E1") * 0.4})
    out = all_pairs_wilcoxon_holm(pd.DataFrame(linhas), "mov", "cond", ["bloco"])
    assert set(out.n_blocks) == {19}
    assert set(out.total_blocks) == {20}


def test_holm_e_monotono_e_nao_excede_um():
    p = [0.001, 0.02, 0.04, 0.9]
    adj = holm_adjust_pvalues(p)
    assert np.all(np.diff(adj) >= 0)
    assert np.all(adj <= 1.0)
    assert adj[0] == pytest.approx(0.004)


# ------------------------------------------------------- agregação por loop

def test_loop_level_movement_agrega_modelos_dentro_do_loop():
    traj = pd.DataFrame({
        "experiment": ["E1"] * 4 + ["B0"] * 4,
        "condition": ["Left"] * 8,
        "eixo": ["Economy"] * 8,
        "question_id": ["q1", "q1", "q2", "q2"] * 2,
        "modelo": ["m1", "m2"] * 4,
        "std": [0.2, 0.4, 1.0, 2.0, 0.0, 0.1, 0.3, 0.5],
    })
    out = loop_level_movement(traj, "std")
    # 2 condições x 2 questões = 4 loops, cada um com 2 modelos.
    assert len(out) == 4
    assert set(out.n_models) == {2}
    e1q1 = out[(out.experiment == "E1") & (out.question_id == "q1")].iloc[0]
    assert e1q1.movement == pytest.approx(0.3)


def test_loop_level_movement_de_metrica_binaria_produz_proporcao():
    """A estabilidade exata é binária por trajetória, mas vira proporção no loop.

    É por isso que a comparação entre condições usa Friedman sobre essa
    proporção e não um teste para resposta dicotômica: dicotomizá-la de volta
    exigiria um limiar arbitrário e descartaria a magnitude.
    """
    traj = pd.DataFrame({
        "experiment": ["E1"] * 4,
        "condition": ["Left"] * 4,
        "eixo": ["Economy"] * 4,
        "question_id": ["q1"] * 4,
        "exact_stability": [1.0, 1.0, 0.0, 0.0],
    })
    out = loop_level_movement(traj, "exact_stability")
    assert len(out) == 1
    assert out.movement.iloc[0] == pytest.approx(0.5)
    assert out.n_models.iloc[0] == 4


def test_loop_level_movement_preserva_pareamento_entre_condicoes():
    """Cada bloco deve existir uma vez por condição, para o Friedman parear."""
    traj = pd.DataFrame({
        "experiment": np.repeat(["B0", "B1", "E1"], 20),
        "condition": ["Left"] * 60,
        "eixo": ["Economy"] * 60,
        "question_id": [f"q{i}" for i in range(10)] * 6,
        "std": np.arange(60, dtype=float),
    })
    out = loop_level_movement(traj, "std")
    contagem = out.groupby("question_id").size()
    assert set(contagem) == {3}


# ------------------------------------------------- reprodutibilidade rodada 1

def _round_level_sintetico(valores_por_exp):
    linhas = []
    for exp, valores in valores_por_exp.items():
        for i, v in enumerate(valores):
            linhas.append({
                "experiment": exp, "rodada": 1, "modelo": f"m{i % 2}",
                "condition": "Left", "eixo": "Economy",
                "question_id": f"q{i}", "likert_valor": v,
            })
    return pd.DataFrame(linhas)


def test_round1_reproducibility_deteta_concordancia_perfeita():
    rl = _round_level_sintetico({"E1": [1, 2, 0, -1], "E2": [1, 2, 0, -1], "E3": [1, 2, 0, -1]})
    por_modelo, glob = round1_reproducibility(rl, ["E1", "E2", "E3"])
    assert glob.all_identical_rate.iloc[0] == pytest.approx(1.0)
    assert glob.mean_range.iloc[0] == pytest.approx(0.0)
    assert glob.n.iloc[0] == 4 and glob.n_runs.iloc[0] == 3
    assert (por_modelo.all_identical_rate == 1.0).all()


def test_round1_reproducibility_quantifica_discordancia():
    # q0 varia em 2 categorias; q1 varia em 1; q2 e q3 concordam.
    rl = _round_level_sintetico({
        "E1": [0, 0, 1, 1],
        "E2": [2, 1, 1, 1],
        "E3": [0, 0, 1, 1],
    })
    _, glob = round1_reproducibility(rl, ["E1", "E2", "E3"])
    assert glob.all_identical_rate.iloc[0] == pytest.approx(0.5)
    assert glob.share_diff_ge_1.iloc[0] == pytest.approx(0.5)
    assert glob.share_diff_ge_2.iloc[0] == pytest.approx(0.25)
    assert glob.mean_range.iloc[0] == pytest.approx(0.75)


def test_round1_reproducibility_ignora_rodadas_posteriores():
    rl = _round_level_sintetico({"E1": [1, 1], "E2": [1, 1]})
    ruido = rl.copy()
    ruido["rodada"] = 2
    ruido["likert_valor"] = -2
    _, glob = round1_reproducibility(pd.concat([rl, ruido]), ["E1", "E2"])
    assert glob.all_identical_rate.iloc[0] == pytest.approx(1.0)


# ------------------------------------------------------------- primitivas base

def test_paired_wide_descarta_blocos_incompletos_e_reporta_total():
    data = pd.DataFrame({
        "bloco": ["a", "a", "b"],
        "cond": ["E1", "E2", "E1"],
        "v": [1.0, 2.0, 3.0],
    })
    wide, total = paired_wide_by_block(data, "v", "cond", ["bloco"])
    assert total == 2 and len(wide) == 1


def test_kendall_w_e_epsilon_quadrado_ficam_no_intervalo_esperado():
    assert friedman_kendall_w(30.0, 100, 4) == pytest.approx(0.1)
    assert kruskal_epsilon_squared(1.0, 100, 5) == 0.0  # floor em zero
    assert kruskal_epsilon_squared(50.0, 100, 5) == pytest.approx(46 / 95)
