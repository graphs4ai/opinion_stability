"""Gera ``paper/template_icaart/results.tex`` a partir da análise.

Todo número que aparece no artigo é produzido aqui e escrito como um comando
LaTeX. Nenhum valor é transcrito à mão, o que elimina a classe de erro mais
provável nesta revisão: deixar no texto um número da coleta antiga.

Uso:
    pixi run export-results
"""

from __future__ import annotations

import json

from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis.data import ROOT, load_all
from src.analysis.manifest import (
    BASELINE_IDS,
    COLETA_JULHO,
    EXPERIMENTS,
    EXPERIMENT_ORDER,
    PROTOCOL_IDS,
    SHORT_ID_TO_MACRO,
)
from src.analysis.model_size_figure import (
    SHORT_MODEL_NAMES,
    compute_model_size_statistics,
)
from src.analysis.stats_core import (
    all_pairs_wilcoxon_holm,
    friedman_by_block,
    kruskal_by_group,
    loop_level_movement,
    round1_reproducibility,
)

# Diretório do manuscrito: os arquivos oficiais do template SCITEPRESS
# vivem aqui junto com icaart.tex e results.tex.
PAPER_DIR = ROOT / 'paper' / 'template_icaart'
# O manuscrito vive no Example.tex do template oficial do SCITEPRESS.
MANUSCRITO = PAPER_DIR / 'Example.tex'
SAIDA = PAPER_DIR / 'results.tex'
LOOP_BLOCKS = ['condition', 'eixo', 'question_id']
AXIS_EN = {
    'Corrupção e Justiça': 'Corruption and justice',
    'Economia': 'Economy',
    'Educação e Cultura': 'Education and culture',
    'Instituições Democráticas': 'Democratic institutions',
    'Meio Ambiente': 'Environment',
    'Políticas Sociais': 'Social policy',
    'Segurança Pública': 'Public security',
}


# ----------------------------------------------------------------- formatação

def num(valor, casas=3):
    if valor is None or (isinstance(valor, float) and not np.isfinite(valor)):
        return r'\textbf{??n/a??}'
    return f'{valor:.{casas}f}'


def pct(valor, casas=1):
    if valor is None or not np.isfinite(valor):
        return r'\textbf{??n/a??}'
    return f'{100 * valor:.{casas}f}\\%'


def milhar(valor):
    return f'{int(valor):,}'.replace(',', '{,}')


def pval(p):
    """p-valores em notação adequada ao texto corrido."""
    if p is None or not np.isfinite(p):
        return r'\textbf{??p??}'
    if p < 1e-10:
        return r'p < 10^{-10}'
    if p < 0.001:
        expo = int(np.floor(np.log10(p)))
        mant = p / 10 ** expo
        return f'p = {mant:.2f} \\times 10^{{{expo}}}'
    return f'p = {p:.4f}'


def holm_resumo(pares, alvo_a, alvo_b):
    """Descreve um contraste pareado já corrigido por Holm."""
    linha = pares[
        ((pares.treatment_a == alvo_a) & (pares.treatment_b == alvo_b))
        | ((pares.treatment_a == alvo_b) & (pares.treatment_b == alvo_a))
    ]
    if linha.empty:
        return r'\textbf{??par??}'
    linha = linha.iloc[0]
    p = linha.p_holm
    marca = '' if p < 0.05 else ', not significant'
    return f'${pval(p)}$ after Holm correction{marca}'


def nome_curto(modelo):
    return SHORT_MODEL_NAMES.get(modelo, str(modelo).split('/')[-1])


# ----------------------------------------------------------------- análise

def coletar(root: Path | None = None) -> dict[str, str]:
    responses, valid, round_level, traj = load_all(root)
    v: dict[str, str] = {}

    # --- contagens -------------------------------------------------------
    n_modelos = responses['modelo'].nunique()
    n_stmt = responses['question_id'].nunique()
    n_cen = responses['condition'].nunique()
    n_rod = int(responses['rodada'].max())
    v['NPlanned'] = milhar(len(EXPERIMENT_ORDER) * n_modelos * n_stmt * n_cen * n_rod)
    v['NRecorded'] = milhar(len(responses))
    v['NValid'] = milhar(len(valid))
    v['NInvalid'] = milhar(len(responses) - len(valid))
    v['NTrajectories'] = milhar(len(traj))
    # As inválidas concentram-se num único modelo; dizer qual evita que o
    # leitor suponha um problema distribuído pelo desenho.
    inval = responses[responses['opcao_extraida'] == 'resposta_invalida']
    if len(inval):
        pior = inval['modelo'].value_counts()
        v['InvalidModel'] = nome_curto(pior.index[0])
        v['InvalidModelShare'] = pct(pior.iloc[0] / len(inval))

    # --- reprodutibilidade da rodada 1 -----------------------------------
    por_modelo, glob = round1_reproducibility(round_level, EXPERIMENT_ORDER)
    g = glob.iloc[0]
    v['RoneIdentical'] = pct(g.all_identical_rate)
    v['RoneRange'] = num(g.mean_range)
    v['RoneGeOne'] = pct(g.share_diff_ge_1)
    v['RoneGeTwo'] = pct(g.share_diff_ge_2)
    pior, melhor = por_modelo.iloc[0], por_modelo.iloc[-1]
    v['RoneWorstModel'] = nome_curto(pior.modelo)
    v['RoneWorstRate'] = pct(pior.all_identical_rate)
    v['RoneBestModel'] = nome_curto(melhor.modelo)
    v['RoneBestRate'] = pct(melhor.all_identical_rate)

    # --- comparação entre condições --------------------------------------
    loops = loop_level_movement(traj, 'std', LOOP_BLOCKS)
    chi2, p, w, n_blocks, _, _ = friedman_by_block(loops, 'movement', 'experiment', LOOP_BLOCKS)
    v['CondBlocks'] = milhar(n_blocks)
    v['CondChi'] = num(chi2, 1)
    # Fragmento de math sem cifrões: o texto o envolve em $...$.
    v['CondP'] = pval(p)
    v['CondW'] = num(w)

    medias = loops.groupby('experiment')['movement'].mean()
    for exp, macro in SHORT_ID_TO_MACRO.items():
        v[f'Mov{macro}'] = num(medias.get(exp, np.nan))

    pares = all_pairs_wilcoxon_holm(loops, 'movement', 'experiment', LOOP_BLOCKS)
    v['PairBzeroEone'] = holm_resumo(pares, 'B0', 'E1')
    v['PairBoneEone'] = holm_resumo(pares, 'B1', 'E1')
    v['PairBzeroBone'] = holm_resumo(pares, 'B0', 'B1')
    v['PairEoneEthree'] = holm_resumo(pares, 'E1', 'E3')
    v['PairEtwoEthree'] = holm_resumo(pares, 'E2', 'E3')
    v['PairEoneEtwo'] = holm_resumo(pares, 'E1', 'E2')

    # Amplitude entre protocolos vs. distância do B1 até eles: a comparação que
    # mostra se a estrutura do resumo importa perto de haver resumo.
    med_prot = [medias[e] for e in PROTOCOL_IDS if e in medias]
    if med_prot:
        v['ProtocolSpread'] = num(max(med_prot) - min(med_prot))
        if 'B1' in medias:
            v['BaselineToProtocolGap'] = num(float(np.mean(med_prot)) - medias['B1'])

    entre_prot = pares[
        pares.treatment_a.isin(PROTOCOL_IDS) & pares.treatment_b.isin(PROTOCOL_IDS)
    ]
    n_sig = int(entre_prot.significant_at_0_05.sum())
    if n_sig == 0:
        v['ProtocolContrasts'] = (
            'none of the three pairwise contrasts survives Holm correction, '
            'indicating that protocol structure had limited effect.'
        )
    elif n_sig == len(entre_prot):
        v['ProtocolContrasts'] = 'all three pairwise contrasts survive Holm correction.'
    else:
        sobrevivem = ', '.join(
            f'{r.treatment_a}--{r.treatment_b}'
            for r in entre_prot[entre_prot.significant_at_0_05].itertuples()
        )
        v['ProtocolContrasts'] = (
            f'{n_sig} of {len(entre_prot)} pairwise contrasts survive Holm '
            f'correction ({sobrevivem}).'
        )

    # --- estabilidade exata ----------------------------------------------
    # Testada por Friedman sobre a proporção agregada por loop, a mesma unidade
    # das métricas de distância. Dicotomizá-la para usar um teste de resposta
    # binária exigiria um limiar arbitrário.
    estab = traj.groupby('experiment')['exact_stability'].mean()
    for exp, macro in SHORT_ID_TO_MACRO.items():
        v[f'Stab{macro}'] = pct(estab.get(exp, np.nan))
    loops_e = loop_level_movement(traj, 'exact_stability', LOOP_BLOCKS)
    chi_e, p_e, w_e, *_ = friedman_by_block(loops_e, 'movement', 'experiment', LOOP_BLOCKS)
    v['StabChi'] = num(chi_e, 1)
    v['StabP'] = pval(p_e)
    v['StabW'] = num(w_e)

    # --- completude e análises de sensibilidade ----------------------------
    completas = traj['n_rounds'] == 5
    v['CompleteShare'] = pct(completas.mean())
    inc = traj.assign(i=~completas).groupby('experiment')['i'].mean()
    v['IncompleteLow'] = pct(inc.min())
    v['IncompleteHigh'] = pct(inc.max())

    comp = traj[completas]
    lo = loop_level_movement(comp, 'std', LOOP_BLOCKS)
    _, _, w_cc, _, _, _ = friedman_by_block(lo, 'movement', 'experiment', LOOP_BLOCKS)
    v['CompleteCaseW'] = num(w_cc)
    pw_cc = all_pairs_wilcoxon_holm(lo, 'movement', 'experiment', LOOP_BLOCKS)
    v['CompleteCaseSameContrasts'] = (
        'the same set of contrasts survives Holm correction'
        if set(map(tuple, pw_cc[pw_cc.significant_at_0_05][['treatment_a', 'treatment_b']].values))
        == set(map(tuple, pares[pares.significant_at_0_05][['treatment_a', 'treatment_b']].values))
        else 'the set of surviving contrasts changes'
    )

    # Subconjunto estrito: loops em que os dez modelos têm trajetória completa
    # nas cinco condições. Custa muitos blocos, mas remove qualquer diferença
    # de composição entre condições.
    cont = comp.groupby(LOOP_BLOCKS + ['experiment']).size().unstack()
    cheios = cont[(cont == comp['modelo'].nunique()).all(axis=1)].index
    if len(cheios) > 2:
        estrito = comp.set_index(LOOP_BLOCKS).loc[cheios].reset_index()
        lo_s = loop_level_movement(estrito, 'std', LOOP_BLOCKS)
        _, _, w_s, nb_s, _, _ = friedman_by_block(lo_s, 'movement', 'experiment', LOOP_BLOCKS)
        v['StrictBlocks'] = milhar(nb_s)
        v['StrictW'] = num(w_s)
        pw_s = all_pairs_wilcoxon_holm(lo_s, 'movement', 'experiment', LOOP_BLOCKS)
        base_prot = pw_s[
            (pw_s.treatment_a.isin(BASELINE_IDS) & pw_s.treatment_b.isin(PROTOCOL_IDS))
            | (pw_s.treatment_a.isin(PROTOCOL_IDS) & pw_s.treatment_b.isin(BASELINE_IDS))
        ]
        entre_p = pw_s[pw_s.treatment_a.isin(PROTOCOL_IDS) & pw_s.treatment_b.isin(PROTOCOL_IDS)]
        v['StrictBaselineHolds'] = ('all' if base_prot.significant_at_0_05.all() else 'not all')
        v['StrictProtocolHolds'] = (
            'no contrast among the protocols survives'
            if not entre_p.significant_at_0_05.any()
            else 'the protocol contrast survives'
        )

    # --- tabela descritiva por condição -----------------------------------
    # Todas as métricas na mesma tabela: as três baseadas em distância e a
    # categórica, para que o leitor veja que concordam sem precisar cruzar
    # números espalhados pelo texto.
    desc = (
        traj.groupby('experiment')
        .agg(sd=('std', 'mean'),
             d1r=('first_to_final_abs_delta', 'mean'),
             cum=('cumulative_abs_shift_from_first', 'mean'),
             est=('exact_stability', 'mean'))
        .reindex(EXPERIMENT_ORDER)
    )
    # Sem a coluna de nome por extenso: ela estourava a largura da coluna, e os
    # nomes já aparecem na tabela de desenho e na caption da figura.
    linhas_tab = [
        f"{exp} & {num(r.sd)} & {num(r.d1r)} & {num(r.cum)} & {pct(r.est)} \\\\"
        for exp, r in desc.iterrows()
    ]
    v['CondTableRows'] = '\n'.join(linhas_tab)

    # --- heterogeneidade entre modelos -----------------------------------
    for exp, macro in SHORT_ID_TO_MACRO.items():
        sub = traj[traj.experiment == exp]
        _, _, w_m, *_ = friedman_by_block(sub, 'std', 'modelo', ['condition', 'eixo', 'question_id'])
        v[f'ModelW{macro}'] = num(w_m, 4)
    por_mod = traj.groupby('modelo')['std'].mean().sort_values()
    v['HighestMovementModel'] = nome_curto(por_mod.index[-1])
    v['LowestMovementModel'] = nome_curto(por_mod.index[0])

    # --- eixos temáticos ---------------------------------------------------
    eixo = traj.groupby('eixo')['first_to_final_abs_delta'].mean().sort_values()
    v['AxisHighest'] = AXIS_EN.get(eixo.index[-1], eixo.index[-1])
    v['AxisHighestValue'] = num(eixo.iloc[-1])
    v['AxisLowest'] = AXIS_EN.get(eixo.index[0], eixo.index[0])
    v['AxisLowestValue'] = num(eixo.iloc[0])
    _, _, eps, _, _ = kruskal_by_group(traj, 'first_to_final_abs_delta', 'eixo')
    v['AxisEpsilon'] = num(eps, 5)

    # --- persistência Likert -----------------------------------------------
    # Apenas pares de rodadas consecutivas: quando uma rodada intermediária é
    # inválida, ligar r-1 a r+1 criaria uma transição que atravessa duas rodadas.
    rl = round_level.sort_values('rodada')
    pares_tr = []
    for _, g_ in rl.groupby(['experiment', 'modelo', 'condition', 'eixo', 'question_id']):
        vals = g_['likert_valor'].to_numpy()
        rds = g_['rodada'].to_numpy()
        pares_tr.extend(
            (vals[k], vals[k + 1]) for k in range(len(vals) - 1) if rds[k + 1] - rds[k] == 1
        )
    tr = pd.DataFrame(pares_tr, columns=['de', 'para']).dropna()
    tr = tr[tr.de.isin([-2, -1, 0, 1, 2])]
    # Quanto do movimento é local (permanecer ou ir a uma categoria vizinha),
    # por condição. Sustenta a afirmação de que o loop ajusta em vez de reverter.
    tr_exp = []
    for _, g_ in rl.groupby(['experiment', 'modelo', 'condition', 'eixo', 'question_id']):
        vals = g_['likert_valor'].to_numpy()
        rds = g_['rodada'].to_numpy()
        exp_id = g_['experiment'].iloc[0]
        tr_exp.extend(
            (exp_id, vals[k], vals[k + 1])
            for k in range(len(vals) - 1) if rds[k + 1] - rds[k] == 1
        )
    te = pd.DataFrame(tr_exp, columns=['exp', 'de', 'para']).dropna()
    te = te[te.de.isin([-2, -1, 0, 1, 2]) & te.para.isin([-2, -1, 0, 1, 2])]
    passo = (te.para - te.de).abs()
    local = te.assign(p=passo).groupby('exp')['p'].apply(lambda s_: (s_ <= 1).mean())
    salto = te.assign(p=passo).groupby('exp')['p'].apply(lambda s_: (s_ >= 2).mean())
    v['LocalMovesLow'] = pct(local.min())
    v['LocalMovesHigh'] = pct(local.max())
    v['JumpShareHigh'] = pct(salto.max())

    persist = tr.assign(igual=tr.de == tr.para).groupby('de')['igual'].mean()
    rot = {-2: 'strongly disagree', -1: 'disagree', 0: 'neutral', 1: 'agree', 2: 'strongly agree'}
    alto, baixo = persist.idxmax(), persist.idxmin()
    v['PersistHighest'] = f'\\emph{{{rot[alto]}}} ({pct(persist[alto])})'
    v['PersistLowest'] = f'\\emph{{{rot[baixo]}}} ({pct(persist[baixo])})'

    # --- tamanho dos modelos (exploratório) --------------------------------
    tabela, loo = compute_model_size_statistics(traj)
    agg = tabela[(tabela.scope.str.startswith('Aggregated')) & (tabela.size_metric == 'active_params_b')]
    if not agg.empty:
        r = agg.iloc[0]
        v['SizeRho'] = num(r.spearman_rho)
        v['SizeP'] = num(r.p_value, 4)
        # Sem cifrões: o texto envolve em $...$.
        v["SizeCI"] = f"[{num(r.ci95_low)}, {num(r.ci95_high)}]"
        v['SizeLooRange'] = (f'$\\rho$ between {num(loo.spearman_rho.min())} '
                             f'and {num(loo.spearman_rho.max())}')

    # --- comprimento dos resumos -------------------------------------------
    # Os resumos ficam ao lado das respostas de cada protocolo; resolvemos o
    # caminho a partir do manifesto para respeitar um `root` alternativo (usado
    # pelos testes) em vez de assumir o diretório do repositório.
    base = ROOT if root is None else Path(root)
    razao_token_palavra = None
    for meta in EXPERIMENTS:
        exp = meta['id']
        if meta['kind'] != 'protocol':
            continue
        caminho = (base / meta['path']).parent / f'resumos_{exp.lower()}.csv'
        if not caminho.exists():
            continue
        d = pd.read_csv(caminho, usecols=['usage_raw', 'resumo_produzido'])
        toks = [
            json.loads(s).get('completion_tokens')
            for s in d['usage_raw'].dropna()
            if isinstance(s, str) and s.strip().startswith('{')
        ]
        toks = [t for t in toks if t is not None]
        if toks:
            v[f'Len{SHORT_ID_TO_MACRO[exp]}'] = num(float(np.mean(toks)), 1)
        if exp == 'E1':
            pal = d.get('resumo_produzido')
            if pal is not None:
                pal = pal.dropna().astype(str).str.split().str.len()
                if len(pal) and toks:
                    razao_token_palavra = float(np.mean(toks)) / float(pal.mean())

    # --- comprimento do contexto do B1 --------------------------------------
    # O artigo afirma que o contexto do B1 é comparável em comprimento aos
    # resumos de E1; medimos em vez de assumir. Os resumos são contados em
    # tokens de completion do sumarizador, então estimamos os tokens do
    # contexto do B1 pela mesma razão tokens/palavra observada nos resumos.
    # A razão tokens/palavra é derivada dos próprios resumos, onde temos texto e
    # completion_tokens lado a lado, em vez de assumir um valor para português.
    b1 = responses[(responses.experiment == 'B1') & responses.contexto_usado.notna()] \
        if 'contexto_usado' in responses.columns else pd.DataFrame()
    if not b1.empty and razao_token_palavra:
        palavras = b1['contexto_usado'].astype(str).str.split().str.len()
        v['LenBone'] = num(float(palavras.mean()) * razao_token_palavra, 1)

    # --- deriva do provedor -------------------------------------------------
    # Comparar SORTEIOS INDIVIDUAIS. Tirar a média das cinco execuções de
    # setembro antes de testar igualdade produziria valores fracionários que
    # quase nunca coincidem com o inteiro de julho, subestimando o acordo.
    chave = ['modelo', 'pair_id', 'tipo_pergunta', 'tendencia']
    julho = {}
    for exp_id, caminho in COLETA_JULHO.items():
        arq = ROOT / caminho
        if not arq.exists():
            continue
        d = pd.read_csv(arq, usecols=chave + ['rodada', 'likert_valor'])
        julho[exp_id] = d[d.rodada == 1].set_index(chave)['likert_valor'].rename(f'jul_{exp_id}')
    setembro = {
        exp: (responses[(responses.experiment == exp) & (responses.rodada == 1)]
              .set_index(chave)['likert_valor'].rename(f'set_{exp}'))
        for exp in EXPERIMENT_ORDER
    }
    if len(julho) >= 2 and len(setembro) >= 2:
        m = pd.concat(list(julho.values()) + list(setembro.values()), axis=1).dropna()
        cj = [c for c in m.columns if c.startswith('jul_')]
        cs = [c for c in m.columns if c.startswith('set_')]
        dentro_jul = [(m[a] == m[b]).mean() for i, a in enumerate(cj) for b in cj[i + 1:]]
        dentro_set = [(m[a] == m[b]).mean() for i, a in enumerate(cs) for b in cs[i + 1:]]
        entre = [(m[a] == m[b]).mean() for a in cs for b in cj]
        v['DriftWithinJuly'] = pct(np.mean(dentro_jul))
        v['DriftWithinSept'] = pct(np.mean(dentro_set))
        v['DriftAcross'] = pct(np.mean(entre))

        unan = m[cj].nunique(axis=1) == 1
        cons = m[unan]
        ref = cons[cj[0]]
        v['DriftUnanimousMatch'] = pct(np.mean([(cons[c] == ref).mean() for c in cs]))
        det = cons.reset_index()[['modelo']].copy()
        det['dif'] = np.mean([(cons[c] != ref).to_numpy() for c in cs], axis=0)
        pm = det.groupby('modelo')['dif'].mean().sort_values()
        v['DriftWorstModel'] = nome_curto(pm.index[-1])
        v['DriftWorstRate'] = pct(pm.iloc[-1])

    return v


def escrever(valores: dict[str, str], destino: Path = SAIDA) -> None:
    linhas = [
        '% GERADO POR src/analysis/export_results.py -- NÃO EDITAR À MÃO.',
        '% Reexecutar com: pixi run export-results',
        '%',
        '% Qualquer macro não preenchida aparece no PDF como ??nome?? em negrito,',
        '% de modo que seja impossível submeter com um número faltando.',
        r'\providecommand{\pending}[1]{\textbf{??#1??}}',
        '',
    ]
    texto = destino.read_text(encoding='utf-8') if destino.exists() else ''
    import re
    conhecidas = re.findall(r'\\newcommand\{\\(\w+)\}', texto)
    for nome in conhecidas:
        valor = valores.get(nome)
        if valor is None:
            linhas.append(f'\\newcommand{{\\{nome}}}{{\\pending{{{nome}}}}}')
        else:
            linhas.append(f'\\newcommand{{\\{nome}}}{{{valor}}}')
    faltando = [n for n in conhecidas if n not in valores]
    destino.write_text('\n'.join(linhas) + '\n', encoding='utf-8')
    print(f'Escrito {destino} com {len(conhecidas) - len(faltando)}/{len(conhecidas)} valores.')
    if faltando:
        print('AINDA PENDENTES:', ', '.join(faltando))


if __name__ == '__main__':
    escrever(coletar())
