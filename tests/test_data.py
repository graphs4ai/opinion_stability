"""Testes das unidades de análise construídas a partir das respostas brutas."""

import numpy as np
import pandas as pd
import pytest

from src.analysis.data import build_round_level, build_trajectories


def _round_level(valores, experiment="E1"):
    return pd.DataFrame({
        "experiment": experiment, "experiment_name": "x", "intervention": "i",
        "modelo": "m1", "condition": "Left", "tendencia": "esquerda",
        "eixo": "Economy", "question_id": "q1",
        "rodada": range(1, len(valores) + 1),
        "likert_valor": [float(v) for v in valores],
    })


def test_metricas_de_distancia_conferem_com_o_calculo_manual():
    t = build_trajectories(_round_level([0, 1, 1, -1, 2])).iloc[0]
    assert t["std"] == pytest.approx(np.std([0, 1, 1, -1, 2]))
    assert t["first_to_final_abs_delta"] == pytest.approx(2.0)
    # |1-0| + |1-0| + |-1-0| + |2-0|
    assert t["cumulative_abs_shift_from_first"] == pytest.approx(5.0)
    assert t["max_abs_shift_from_first"] == pytest.approx(2.0)


def test_metricas_categoricas_nao_usam_distancia():
    """Passos |1,0,2,3|: 3 mudanças, 1 adjacente, 2 saltos >= 2 categorias."""
    t = build_trajectories(_round_level([0, 1, 1, -1, 2])).iloc[0]
    assert t["exact_stability"] == 0.0
    assert t["changed_first_to_final"] == 1.0
    assert t["n_category_changes"] == 3.0
    assert t["share_adjacent_transitions"] == pytest.approx(0.25)
    assert t["share_jumps_ge2"] == pytest.approx(0.5)


def test_trajetoria_constante_e_exatamente_estavel():
    t = build_trajectories(_round_level([1, 1, 1, 1, 1])).iloc[0]
    assert t["exact_stability"] == 1.0
    assert t["any_change"] == 0.0
    assert t["changed_first_to_final"] == 0.0
    assert t["std"] == 0.0
    assert t["n_category_changes"] == 0.0
    assert t["share_jumps_ge2"] == 0.0


def test_exact_stability_e_complemento_exato_de_any_change():
    for valores in ([1, 1, 1], [1, 2, 1], [0, 0, 1], [-2, 2, -2]):
        t = build_trajectories(_round_level(valores)).iloc[0]
        assert t["exact_stability"] + t["any_change"] == pytest.approx(1.0)


def test_retorno_ao_ponto_inicial_e_estavel_no_extremo_mas_nao_exatamente():
    """first_to_final esconde movimento intermediário; exact_stability não."""
    t = build_trajectories(_round_level([1, -2, 2, 1])).iloc[0]
    assert t["changed_first_to_final"] == 0.0
    assert t["first_to_final_abs_delta"] == 0.0
    assert t["exact_stability"] == 0.0
    assert t["cumulative_abs_shift_from_first"] > 0


def test_trajetorias_com_menos_de_duas_rodadas_sao_descartadas():
    assert build_trajectories(_round_level([1])).empty


def test_build_round_level_faz_media_entre_repeticoes():
    linhas = []
    for rep, v in [(1, 0.0), (2, 2.0)]:
        d = _round_level([v])
        d["repeticao"] = rep
        linhas.append(d)
    rl = build_round_level(pd.concat(linhas))
    assert len(rl) == 1
    assert rl["likert_valor"].iloc[0] == pytest.approx(1.0)
