"""Primitivas estatísticas compartilhadas pelo notebook e pelos testes.

As funções ``kruskal_*``, ``paired_wide_by_block``, ``friedman_*`` e
``holm_adjust_pvalues`` vinham definidas dentro de ``analises.ipynb``, o que
impedia testá-las de outro modo que não por comparação de strings no código-
fonte do notebook. O comportamento aqui é idêntico ao original.

As funções novas — ``all_pairs_wilcoxon_holm``, ``loop_level_movement`` e
``round1_reproducibility`` — sustentam a comparação direta entre as condições
B0/B1/E1/E2/E3.

Todas as métricas, inclusive a estabilidade exata, são comparadas por Friedman
sobre a mesma unidade: a proporção agregada por loop. Uma versão anterior
dicotomizava a estabilidade agregada para aplicar o Q de Cochran, o que exigia
um limiar arbitrário e descartava a magnitude da proporção.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats


# --------------------------------------------------------------------------
# Primitivas originais (comportamento preservado)
# --------------------------------------------------------------------------

def kruskal_epsilon_squared(h_stat, n, k):
    if pd.isna(h_stat) or n <= k:
        return np.nan
    return max(0.0, (h_stat - k + 1) / (n - k))


def kruskal_by_group(data, value_col, group_col):
    groups = [g[value_col].dropna().to_numpy() for _, g in data.groupby(group_col)]
    groups = [g for g in groups if len(g) > 0]
    if len(groups) < 2:
        return np.nan, np.nan, np.nan, sum(len(g) for g in groups), len(groups)
    h_stat, p_value = stats.kruskal(*groups)
    n = sum(len(g) for g in groups)
    k = len(groups)
    return h_stat, p_value, kruskal_epsilon_squared(h_stat, n, k), n, k


def paired_wide_by_block(data, value_col, treatment_col, block_cols):
    """Pivot bloco x tratamento, descartando blocos incompletos (listwise).

    Retorna também o total de blocos antes do descarte, para que o chamador
    possa reportar quantos foram perdidos.
    """
    if isinstance(block_cols, str):
        block_cols = [block_cols]
    wide = data.pivot_table(
        index=block_cols,
        columns=treatment_col,
        values=value_col,
        aggfunc='mean',
    )
    total_blocks = len(wide)
    wide = wide.dropna(axis=0, how='any')
    return wide, total_blocks


def friedman_kendall_w(chi2_stat, n_blocks, n_treatments):
    if pd.isna(chi2_stat) or n_blocks <= 0 or n_treatments <= 1:
        return np.nan
    return chi2_stat / (n_blocks * (n_treatments - 1))


def friedman_by_block(data, value_col, treatment_col, block_cols):
    wide, total_blocks = paired_wide_by_block(data, value_col, treatment_col, block_cols)
    n_blocks, n_treatments = wide.shape
    if n_blocks < 2 or n_treatments < 2:
        return np.nan, np.nan, np.nan, n_blocks, n_treatments, total_blocks
    chi2_stat, p_value = stats.friedmanchisquare(
        *[wide[col].to_numpy(dtype=float) for col in wide.columns]
    )
    kendall_w = friedman_kendall_w(chi2_stat, n_blocks, n_treatments)
    return chi2_stat, p_value, kendall_w, n_blocks, n_treatments, total_blocks


def holm_adjust_pvalues(p_values):
    p_values = np.asarray(p_values, dtype=float)
    adjusted = np.full_like(p_values, np.nan, dtype=float)
    valid = np.where(~np.isnan(p_values))[0]
    if len(valid) == 0:
        return adjusted
    ordered = valid[np.argsort(p_values[valid])]
    running_max = 0.0
    m = len(ordered)
    for rank, idx in enumerate(ordered):
        adj = min(1.0, (m - rank) * p_values[idx])
        running_max = max(running_max, adj)
        adjusted[idx] = running_max
    return adjusted


# --------------------------------------------------------------------------
# Novas primitivas para a comparação entre condições
# --------------------------------------------------------------------------

def all_pairs_wilcoxon_holm(data, value_col, treatment_col, block_cols):
    """Wilcoxon pareado em TODOS os k(k-1)/2 pares, com correção de Holm.

    Difere de ``model_stability_wilcoxon_holm`` do notebook, que compara apenas
    o melhor tratamento contra os demais (k-1 comparações). Para responder ao
    Reviewer 2 é preciso o conjunto completo de contrastes entre condições.

    Todos os pares usam o mesmo conjunto de blocos completos, de modo que os
    contrastes sejam mutuamente comparáveis. Retorna um DataFrame ordenado pelo
    p-valor bruto.
    """
    wide, total_blocks = paired_wide_by_block(data, value_col, treatment_col, block_cols)
    tratamentos = list(wide.columns)
    linhas = []
    for a, b in combinations(tratamentos, 2):
        x = wide[a].to_numpy(dtype=float)
        y = wide[b].to_numpy(dtype=float)
        if len(x) < 2 or np.all(x == y):
            stat, p = np.nan, np.nan
        else:
            stat, p = stats.wilcoxon(x, y, zero_method='zsplit')
        linhas.append({
            'treatment_a': a,
            'treatment_b': b,
            'mean_a': float(np.mean(x)) if len(x) else np.nan,
            'mean_b': float(np.mean(y)) if len(y) else np.nan,
            'mean_diff': float(np.mean(x - y)) if len(x) else np.nan,
            'median_diff': float(np.median(x - y)) if len(x) else np.nan,
            'wilcoxon_stat': stat,
            'p_value': p,
            'n_blocks': len(x),
            'total_blocks': total_blocks,
        })
    out = pd.DataFrame(linhas)
    if out.empty:
        return out
    out['p_holm'] = holm_adjust_pvalues(out['p_value'].to_numpy())
    out['significant_at_0_05'] = out['p_holm'] < 0.05
    return out.sort_values('p_value').reset_index(drop=True)


def loop_level_movement(trajectories, value_col, block_cols=('condition', 'eixo', 'question_id')):
    """Agrega o movimento dos dez modelos dentro de cada loop statement x cenário.

    A unidade passa a ser o loop, não a trajetória individual. Isso responde ao
    Reviewer 4: como todos os modelos recebem o mesmo resumo, as trajetórias
    dentro de um loop não são independentes; agregá-las antes de testar remove
    boa parte da pseudo-independência em vez de fingir que ela não existe.

    Retorna um DataFrame com uma linha por (experiment, bloco) e a coluna
    ``movement``.
    """
    block_cols = list(block_cols)
    return (
        trajectories
        .groupby(['experiment'] + block_cols, as_index=False)
        .agg(movement=(value_col, 'mean'), n_models=(value_col, 'size'))
    )


def round1_reproducibility(round_level, experiments, block_cols=('modelo', 'condition', 'eixo', 'question_id')):
    """Concordância entre execuções independentes da MESMA condição na rodada 1.

    Na rodada 1 nenhum resumo foi produzido ainda e o prompt é byte a byte
    idêntico em todas as condições, de modo que cada condição listada em
    ``experiments`` fornece um sorteio independente da mesma condição. Mede o
    não-determinismo residual a temperatura 0, que o Reviewer 1 apontou como
    não quantificado.

    Retorna ``(resumo_por_modelo, resumo_global)``.
    """
    block_cols = list(block_cols)
    r1 = round_level[round_level['rodada'] == 1]
    r1 = r1[r1['experiment'].isin(experiments)]
    wide = r1.pivot_table(
        index=block_cols,
        columns='experiment',
        values='likert_valor',
        aggfunc='first',
    ).dropna(axis=0, how='any')

    cols = [c for c in experiments if c in wide.columns]
    arr = wide[cols].to_numpy(dtype=float)
    if arr.shape[1] < 2:
        raise ValueError("São necessárias ao menos duas execuções para medir reprodutibilidade.")

    todas_iguais = (arr == arr[:, [0]]).all(axis=1)
    amplitude = arr.max(axis=1) - arr.min(axis=1)
    detalhe = wide.reset_index()[block_cols].copy()
    detalhe['all_identical'] = todas_iguais
    detalhe['range'] = amplitude
    detalhe['diff_ge_1'] = amplitude >= 1
    detalhe['diff_ge_2'] = amplitude >= 2

    por_modelo = (
        detalhe.groupby('modelo', as_index=False)
        .agg(
            n=('all_identical', 'size'),
            all_identical_rate=('all_identical', 'mean'),
            mean_range=('range', 'mean'),
            share_diff_ge_1=('diff_ge_1', 'mean'),
            share_diff_ge_2=('diff_ge_2', 'mean'),
        )
        .sort_values('all_identical_rate')
        .reset_index(drop=True)
    )
    global_ = pd.DataFrame([{
        'n': len(detalhe),
        'n_runs': len(cols),
        'runs': '/'.join(cols),
        'all_identical_rate': detalhe['all_identical'].mean(),
        'mean_range': detalhe['range'].mean(),
        'share_diff_ge_1': detalhe['diff_ge_1'].mean(),
        'share_diff_ge_2': detalhe['diff_ge_2'].mean(),
    }])
    return por_modelo, global_
