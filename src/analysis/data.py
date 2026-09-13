"""Carregamento e construção das unidades de análise.

Estas funções viviam dentro de ``analises.ipynb``. Extraí-las permite que o
notebook e o script de exportação de resultados para o LaTeX partam
exatamente do mesmo pipeline, em vez de reimplementá-lo — o que era a origem
mais provável de números divergentes entre a análise e o texto do artigo.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis.manifest import EXPERIMENTS, TENDENCY_LABELS

ROOT = Path(__file__).resolve().parents[2]


def load_experiment(meta, root: Path | None = None) -> pd.DataFrame:
    root = ROOT if root is None else root
    df = pd.read_csv(root / meta['path'])
    df['experiment'] = meta['id']
    df['experiment_name'] = meta['name']
    df['intervention'] = meta['intervention']
    df['rodada'] = pd.to_numeric(df.get('rodada', 1), errors='coerce').fillna(1).astype(int)
    df['likert_valor'] = pd.to_numeric(df.get('likert_valor'), errors='coerce')
    for col in ['modelo', 'provider', 'eixo', 'tipo_pergunta', 'pergunta',
                'tendencia', 'pair_id', 'temperatura', 'repeticao']:
        if col not in df.columns:
            df[col] = np.nan
    df['condition'] = df['tendencia'].map(TENDENCY_LABELS).fillna(df['tendencia'])
    df['question_id'] = (
        df['pair_id'].astype(str) + '|' + df['tipo_pergunta'].astype(str)
        + '|' + df['pergunta'].astype(str)
    )
    return df


def build_round_level(valid: pd.DataFrame) -> pd.DataFrame:
    return (
        valid.groupby(
            ['experiment', 'experiment_name', 'intervention', 'modelo', 'condition',
             'tendencia', 'eixo', 'question_id', 'rodada'],
            dropna=False,
        )['likert_valor']
        .mean()
        .reset_index()
        .sort_values(['experiment', 'modelo', 'condition', 'eixo', 'question_id', 'rodada'])
    )


def build_trajectories(round_level: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_cols = ['experiment', 'experiment_name', 'intervention', 'modelo',
                  'condition', 'tendencia', 'eixo', 'question_id']
    for keys, group in round_level.groupby(group_cols, dropna=False):
        ordered = group.sort_values('rodada')
        values = ordered['likert_valor'].to_numpy(dtype=float)
        rounds = ordered['rodada'].to_numpy(dtype=int)
        if len(values) == 0:
            continue
        diffs = np.diff(values)
        first = values[0]
        final = values[-1]
        # Métricas categóricas: não dependem de as categorias Likert serem
        # equidistantes, ao contrário de std/displacement/cumulative.
        mudou = float(len(np.unique(values)) > 1)
        passos = np.abs(diffs)
        rows.append({
            'experiment': keys[0],
            'experiment_name': keys[1],
            'intervention': keys[2],
            'modelo': keys[3],
            'condition': keys[4],
            'tendencia': keys[5],
            'eixo': keys[6],
            'question_id': keys[7],
            'n_rounds': len(np.unique(rounds)),
            'first_score': first,
            'final_score': final,
            'std': np.std(values, ddof=0),
            'variance': np.var(values, ddof=0),
            'any_change': mudou,
            'exact_stability': 1.0 - mudou,
            'changed_first_to_final': float(first != final),
            'n_category_changes': float(np.count_nonzero(diffs)) if len(diffs) else 0.0,
            'share_adjacent_transitions': float(np.mean(passos == 1)) if len(passos) else 0.0,
            'share_jumps_ge2': float(np.mean(passos >= 2)) if len(passos) else 0.0,
            'first_to_final_delta': final - first,
            'first_to_final_abs_delta': abs(final - first),
            'max_abs_shift_from_first': float(np.max(np.abs(values - first))),
            'cumulative_abs_shift_from_first': (
                float(np.sum(np.abs(values[1:] - first))) if len(values) > 1 else 0.0
            ),
            'mean_abs_step_change': float(np.mean(np.abs(diffs))) if len(diffs) else 0.0,
        })
    traj = pd.DataFrame(rows)
    return traj[traj['n_rounds'] > 1].copy()


def load_all(root: Path | None = None):
    """Retorna (responses, valid, round_level, trajectories) para as cinco condições."""
    responses = pd.concat(
        [load_experiment(meta, root) for meta in EXPERIMENTS], ignore_index=True
    )
    valid = responses.dropna(subset=['likert_valor']).copy()
    round_level = build_round_level(valid)
    trajectories = build_trajectories(round_level)
    return responses, valid, round_level, trajectories
