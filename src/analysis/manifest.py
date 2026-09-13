"""Fonte única de verdade sobre as condições experimentais.

Antes desta consolidação a lista de experimentos aparecia em quatro lugares
(célula 1 de ``analises.ipynb``, ``EXPERIMENTS`` em ``model_size_figure.py``,
``scope_labels`` em ``_prepare_size_data`` e ``order`` em
``plot_active_size_forest``). Acrescentar uma condição exigia editar todos, e
esquecer um deles produzia ``NaN`` silenciosos que só estouravam no guard
``Missing values after metadata merge``.

Sobre a coleta de setembro de 2026
----------------------------------
As execuções originais de E1/E2/E3 são de julho de 2026. Uma sonda de rodada 1
sobre 3.360 células (prompt byte a byte idêntico) mostrou que os modelos
servidos pelo provedor derivaram no intervalo: o acordo entre as três execuções
de julho é de 92,6-93,6%, mas o acordo entre setembro e julho cai para
84,4-84,9%, e as recusas passam de ~31 para 199. A deriva concentra-se em
Qwen3 30B (37,5% de discordância) e Llama 3.1 8B (33,8%) — justamente os
modelos que sustentam os achados principais.

Comparar baselines coletados em setembro contra protocolos coletados em julho
confundiria o contraste central com dois meses de mudança do provedor. Por isso
as cinco condições foram recoletadas contemporaneamente em
``dados/coleta_2026_09/``. Os dados de julho seguem preservados em
``dados/experimento_{1,2,3}/`` e são usados apenas na análise de deriva.
"""

from __future__ import annotations

# Diretório da coleta contemporânea das cinco condições.
COLETA = "dados/coleta_2026_09"

# Diretórios da coleta original de julho de 2026, preservados para a análise de
# deriva. Não usar nas análises principais.
COLETA_JULHO = {
    "E1": "dados/experimento_1/respostas_iterativas.csv",
    "E2": "dados/experimento_2/respostas_iterativas_v2.csv",
    "E3": "dados/experimento_3/respostas_iterativas_v3.csv",
}


# ``kind`` separa condições de controle (sem sumarizador) dos protocolos de
# sumarização. As análises que agregam "os protocolos" filtram por kind.
EXPERIMENTS = [
    {
        "id": "B0",
        "name": "No-context repetition",
        "path": f"{COLETA}/b0/respostas_b0.csv",
        "intervention": "none",
        "kind": "baseline",
    },
    {
        "id": "B1",
        "name": "Self-explanation feedback",
        "path": f"{COLETA}/b1/respostas_b1.csv",
        "intervention": "self explanation",
        "kind": "baseline",
    },
    {
        "id": "E1",
        "name": "Free summary",
        "path": f"{COLETA}/e1/respostas_e1.csv",
        "intervention": "summary",
        "kind": "protocol",
    },
    {
        "id": "E2",
        "name": "Negative and positive points",
        "path": f"{COLETA}/e2/respostas_e2.csv",
        "intervention": "summary v2",
        "kind": "protocol",
    },
    {
        "id": "E3",
        "name": "Trade-offs and uncertainty",
        "path": f"{COLETA}/e3/respostas_e3.csv",
        "intervention": "structured summary",
        "kind": "protocol",
    },
]

EXPERIMENT_BY_ID = {meta["id"]: meta for meta in EXPERIMENTS}
EXPERIMENT_ORDER = [meta["id"] for meta in EXPERIMENTS]
BASELINE_IDS = [m["id"] for m in EXPERIMENTS if m["kind"] == "baseline"]
PROTOCOL_IDS = [m["id"] for m in EXPERIMENTS if m["kind"] == "protocol"]

# Rótulos de painel: quebras de linha só onde o texto não cabe.
FIGURE_EXPERIMENT_LABELS = {
    "B0": "B0 — No-context\nrepetition",
    "B1": "B1 — Self-explanation\nfeedback",
    "E1": "E1 — Free summary",
    "E2": "E2 — Negative and\npositive points",
    "E3": "E3 — Trade-offs and\nuncertainty",
}

# Escopos das correlações de tamanho de modelo. A agregação cobre apenas os
# protocolos: misturar baselines num "movimento médio" não teria sentido.
AGGREGATED_SCOPE = "Aggregated E1/E2/E3"
SCOPE_LABELS = {meta["id"]: meta["name"] for meta in EXPERIMENTS}
SCOPE_ORDER = [SCOPE_LABELS[i] for i in EXPERIMENT_ORDER] + [AGGREGATED_SCOPE]

TENDENCY_LABELS = {"esquerda": "Left", "neutro": "Neutral", "direita": "Right"}

AXIS_SHORT_NAMES = {
    "Corrupção e Justiça": "Corruption/justice",
    "Economia": "Economy",
    "Educação e Cultura": "Education/culture",
    "Instituições Democráticas": "Democratic inst.",
    "Meio Ambiente": "Environment",
    "Políticas Sociais": "Social policy",
    "Segurança Pública": "Public security",
}

# Nomes de macro LaTeX por condição. Comandos LaTeX não podem conter dígitos,
# de modo que "B0" vira "Bzero" em \MovBzero, \StabBzero etc.
SHORT_ID_TO_MACRO = {
    "B0": "Bzero",
    "B1": "Bone",
    "E1": "Eone",
    "E2": "Etwo",
    "E3": "Ethree",
}

# Unidade da comparação principal entre condições: o loop statement x cenário.
# Dentro de um loop os dez modelos recebem o mesmo resumo, então suas
# trajetórias não são independentes; agregá-las antes de testar é o que
# responde à objeção de pseudo-independência.
LOOP_BLOCK_COLS = ['condition', 'eixo', 'question_id']
