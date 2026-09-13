"""Teste de integração do exportador de resultados para o LaTeX.

Monta uma coleta sintética com as cinco condições e verifica que
``coletar`` produz todos os valores que ``results.tex`` declara. O objetivo é
garantir que, quando os dados reais chegarem, nenhum macro fique pendente por
um erro de encanamento.
"""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.analysis.export_results import MANUSCRITO, PAPER_DIR, coletar, escrever
from src.analysis.manifest import EXPERIMENTS

MODELOS = [
    "google/gemma-3-4b-it", "google/gemma-4-31B-it",
    "meta-llama/Llama-4-Scout-17B-16E-Instruct", "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "Qwen/Qwen3-30B-A3B", "Qwen/Qwen3-14B",
    "mistralai/Mistral-Small-3.2-24B-Instruct-2506", "mistralai/Mistral-Nemo-Instruct-2407",
    "deepseek-ai/DeepSeek-V3.2", "deepseek-ai/DeepSeek-V4-Flash",
]
EIXOS = ["Economia", "Meio Ambiente", "Instituições Democráticas"]
TENDENCIAS = ["esquerda", "direita", "neutro"]


def _coleta_sintetica(root: Path, n_pares=6, n_rodadas=5):
    """Baselines quase estáveis, protocolos com mais movimento."""
    rng = np.random.default_rng(7)
    movimento = {"B0": 0.02, "B1": 0.08, "E1": 0.30, "E2": 0.28, "E3": 0.36}

    for meta in EXPERIMENTS:
        exp = meta["id"]
        linhas = []
        for pid in range(n_pares):
            for tipo in ("P+", "P-"):
                for tend in TENDENCIAS:
                    for modelo in MODELOS:
                        atual = int(rng.integers(-2, 3))
                        # Como nos dados reais, as recusas concentram-se num
                        # único modelo em vez de se espalharem pelo desenho.
                        recusa = modelo == "meta-llama/Meta-Llama-3.1-8B-Instruct" and pid == 0
                        for rodada in range(1, n_rodadas + 1):
                            if rodada > 1 and rng.random() < movimento[exp]:
                                atual = int(np.clip(atual + rng.choice([-1, 1]), -2, 2))
                            linhas.append({
                                "modelo": modelo, "provider": "deepinfra",
                                "eixo": EIXOS[pid % len(EIXOS)], "tipo_pergunta": tipo,
                                "pergunta": f"afirmacao {pid} {tipo}",
                                "temperatura": 0.0, "repeticao": 1, "tendencia": tend,
                                "pair_id": pid, "modelo_gerador": "g", "rodada": rodada,
                                "likert_valor": None if recusa else atual,
                                "opcao_extraida": "resposta_invalida" if recusa else "Concordo",
                                "explicacao_extraida": "x",
                                "erro_parse": "json_invalido" if recusa else None,
                                # B1 injeta a explicação própria da rodada anterior;
                                # as demais condições não têm contexto por modelo.
                                "modo_contexto": {"B0": "none", "B1": "self"}.get(exp),
                                "contexto_usado": (
                                    " ".join(["palavra"] * 78)
                                    if exp == "B1" and rodada > 1 else None
                                ),
                                "contexto_fallback": False,
                                "usage_raw": None, "custo_estimado": 0.0,
                                "timestamp": "2026-09-04T00:00:00",
                            })
        destino = root / meta["path"]
        destino.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(linhas).to_csv(destino, index=False)

        # resumos, apenas para os protocolos
        if meta["kind"] == "protocol":
            tokens = {"E1": 136, "E2": 214, "E3": 327}[exp]
            # 100 palavras por resumo: com 136 tokens em E1 isso fixa a razão
            # tokens/palavra em 1.36, que o exportador deriva daqui em vez de
            # assumir um valor.
            resumos = [{
                "resumo_produzido": " ".join(["palavra"] * 100),
                "usage_raw": json.dumps({"completion_tokens": tokens, "prompt_tokens": 100}),
            } for _ in range(20)]
            pd.DataFrame(resumos).to_csv(
                destino.parent / f"resumos_{exp.lower()}.csv", index=False
            )


@pytest.fixture(scope="module")
def valores(tmp_path_factory):
    root = tmp_path_factory.mktemp("coleta")
    _coleta_sintetica(root)
    return coletar(root)


def test_contagens_batem_com_o_desenho(valores):
    # 10 modelos x 12 statements x 3 cenários x 5 rodadas x 5 condições
    assert valores["NRecorded"] == "9{,}000"
    # Um modelo recusa em 2 statements x 3 cenários x 5 rodadas x 5 condições.
    assert valores["NInvalid"] == "150"
    assert valores["NValid"] == "8{,}850"


def test_inválidas_sao_atribuidas_ao_modelo_responsavel(valores):
    assert valores["InvalidModel"] == "Llama 3.1 8B"
    assert valores["InvalidModelShare"] == "100.0\\%"


def test_todos_os_macros_do_paper_sao_preenchidos(valores):
    """Nenhum \\pending deve sobrar quando os dados estão completos."""
    declarados = re.findall(
        r"\\newcommand\{\\(\w+)\}",
        (PAPER_DIR / "results.tex").read_text(encoding="utf-8"),
    )
    faltando = [n for n in declarados if n not in valores]
    # Os macros de deriva dependem dos CSVs de julho, ausentes no fixture.
    faltando = [n for n in faltando if not n.startswith("Drift")]
    assert not faltando, f"macros sem valor: {faltando}"


def test_baselines_movem_menos_que_os_protocolos(valores):
    b0, b1 = float(valores["MovBzero"]), float(valores["MovBone"])
    protocolos = [float(valores[f"Mov{m}"]) for m in ("Eone", "Etwo", "Ethree")]
    assert b0 < b1 < min(protocolos)


def test_estabilidade_exata_ordena_ao_contrario_do_movimento(valores):
    """Mais movimento implica menos trajetórias inalteradas."""
    ler = lambda k: float(valores[k].replace("\\%", ""))
    assert ler("StabBzero") > ler("StabBone") > max(
        ler("StabEone"), ler("StabEtwo"), ler("StabEthree")
    )


def test_comprimento_do_contexto_do_b1_e_medido(valores):
    """O artigo compara o contexto do B1 com os resumos de E1; ambos precisam existir."""
    # 78 palavras de contexto x razão 1.36 derivada dos resumos de E1.
    assert float(valores["LenBone"]) == pytest.approx(78 * 1.36, rel=1e-3)
    assert float(valores["LenEone"]) == pytest.approx(136.0)


def test_macros_usados_dentro_de_math_nao_carregam_cifroes(valores):
    """Um macro com $ próprio, usado dentro de $...$, gera $$ e quebra o LaTeX.

    Foi exatamente o que aconteceu com \\SizeCI e \\CondP. O teste varre o
    manuscrito, encontra os macros usados em modo matemático e exige que seus
    valores não tragam cifrões.
    """
    tex = MANUSCRITO.read_text(encoding="utf-8")
    dentro_de_math = set()
    for trecho in re.findall(r"(?<!\\)\$([^$]+)\$", tex):
        dentro_de_math.update(re.findall(r"\\([A-Za-z]+)", trecho))
    verificados = [n for n in dentro_de_math if n in valores]
    assert verificados, "nenhum macro de resultado encontrado em modo matemático"
    culpados = [n for n in verificados if "$" in valores[n]]
    assert not culpados, f"macros com cifrões próprios usados dentro de math: {culpados}"


def test_pvalores_trazem_a_relacao_e_a_correcao(valores):
    # Fragmento de math sem cifrões, já contendo "p =" ou "p <".
    assert re.match(r"^p [<=]", valores["CondP"]), valores["CondP"]
    assert "$" not in valores["CondP"]
    # Contrastes pareados são usados em texto corrido, então trazem os cifrões.
    assert valores["PairBzeroEone"].startswith("$")
    assert "Holm" in valores["PairBzeroEone"]


def test_escrever_gera_arquivo_sem_pendencias(valores, tmp_path):
    destino = tmp_path / "results.tex"
    destino.write_text(
        (PAPER_DIR / "results.tex").read_text(encoding="utf-8"), encoding="utf-8"
    )
    escrever(valores, destino)
    conteudo = destino.read_text(encoding="utf-8")
    pendentes = re.findall(r"\\pending\{(\w+)\}", conteudo)
    pendentes = [p for p in pendentes if not p.startswith("Drift") and p != "#1"]
    assert not pendentes, f"pendentes após escrita: {pendentes}"
