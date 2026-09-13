"""Condições de controle sem sumarização (B0 e B1).

Este script executa o mesmo laço de cinco rodadas dos experimentos E1/E2/E3
(``iterative_experiment{,_v2,_v3}.py``), porém sem nenhuma chamada ao
sumarizador. O contexto devolvido ao modelo entre rodadas é definido por
``MODO_CONTEXTO``:

``none`` (B0)
    Nenhum contexto. O modelo responde cinco vezes ao mesmo prompt de rodada 1.
    Mede o movimento atribuível apenas a geração repetida.

``self`` (B1)
    O modelo recebe a explicação que ele próprio produziu na rodada anterior.
    Nenhuma informação dos outros nove modelos circula. Isola o efeito de
    recontextualização em geral do efeito do resumo multi-modelo compartilhado.

Ambas as condições mantêm modelos, statements, cenários, temperatura, número de
rodadas e limite de tokens idênticos aos protocolos E1/E2/E3; só o conteúdo do
contexto entre rodadas muda.
"""

import asyncio
import json
import logging
import os
import random
from datetime import datetime

import hydra
import pandas as pd
from omegaconf import DictConfig

from src.main.utils import (
    carregar_cache,
    configurar_concorrencia_deepinfra,
    gerar_chave_cache_generica,
    salvar_cache,
    chamar_api_provider,
)


CONTADOR_NOVAS_RESPOSTAS = 0

MODOS_CONTEXTO_VALIDOS = ("none", "self")

OPCOES_LIKERT = [
    "Discordo fortemente",
    "Discordo",
    "Neutro",
    "Concordo",
    "Concordo fortemente",
]

MAPEAMENTO_LIKERT = {
    "Concordo fortemente": 2,
    "Concordo": 1,
    "Neutro": 0,
    "Discordo": -1,
    "Discordo fortemente": -2,
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)


class ProgressoExperimento:
    def __init__(self, total, intervalo=10):
        self.total = int(total)
        self.intervalo = max(1, int(intervalo))
        self.concluidas = 0
        self.puladas = 0
        self.proximo_log = self.intervalo

    def registrar_conclusao(self, quantidade=1):
        self.concluidas += int(quantidade)
        processadas = self.concluidas + self.puladas
        if processadas >= self.proximo_log or processadas >= self.total:
            self._logar()
            while self.proximo_log <= processadas:
                self.proximo_log += self.intervalo

    def registrar_puladas(self, quantidade):
        quantidade = max(0, int(quantidade))
        if not quantidade:
            return
        self.puladas += quantidade
        self._logar()

    def finalizar(self):
        processadas = self.concluidas + self.puladas
        restantes = max(0, self.total - processadas)
        logger.info(
            "Experimento finalizado: %s/%s operações concluídas, %s puladas, %s restantes.",
            self.concluidas,
            self.total,
            self.puladas,
            restantes,
        )

    def _logar(self):
        processadas = min(self.total, self.concluidas + self.puladas)
        restantes = max(0, self.total - processadas)
        percentual = (100 * processadas / self.total) if self.total else 100.0
        logger.info(
            "Progresso: %s/%s operações processadas (%.1f%%); faltam %s.",
            processadas,
            self.total,
            percentual,
            restantes,
        )


def serializar_json(valor):
    if valor is None:
        return None
    return json.dumps(valor, ensure_ascii=False)


def garantir_diretorio_pai(path):
    diretorio = os.path.dirname(path)
    if diretorio:
        os.makedirs(diretorio, exist_ok=True)


def extrair_primeiro_objeto_json(texto):
    if not texto:
        return None

    texto = texto.strip()
    if texto.startswith("```"):
        linhas = texto.splitlines()
        if linhas and linhas[0].startswith("```"):
            linhas = linhas[1:]
        if linhas and linhas[-1].startswith("```"):
            linhas = linhas[:-1]
        texto = "\n".join(linhas).strip()

    inicio = texto.find("{")
    if inicio == -1:
        return None

    profundidade = 0
    dentro_string = False
    escape = False
    for i in range(inicio, len(texto)):
        char = texto[i]
        if dentro_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                dentro_string = False
            continue

        if char == '"':
            dentro_string = True
        elif char == "{":
            profundidade += 1
        elif char == "}":
            profundidade -= 1
            if profundidade == 0:
                candidato = texto[inicio : i + 1]
                try:
                    return json.loads(candidato)
                except json.JSONDecodeError:
                    return None

    return None


def parsear_resposta_participante(texto):
    objeto = extrair_primeiro_objeto_json(texto)
    if not isinstance(objeto, dict):
        return None, None, "json_invalido"

    opcao = objeto.get("opcao")
    explicacao = objeto.get("explicacao")
    if opcao not in OPCOES_LIKERT:
        return opcao, explicacao, "opcao_invalida"
    if not isinstance(explicacao, str) or not explicacao.strip():
        return opcao, explicacao, "explicacao_invalida"

    return opcao, explicacao.strip(), None


def validar_modo_contexto(modo):
    modo = str(modo).strip().lower()
    if modo not in MODOS_CONTEXTO_VALIDOS:
        raise ValueError(
            f"MODO_CONTEXTO deve ser um de {MODOS_CONTEXTO_VALIDOS}; recebido: {modo!r}"
        )
    return modo


def montar_prompt_participante(afirmacao, contexto_anterior=None):
    """Prompt do participante.

    O ramo sem contexto é byte a byte idêntico ao dos protocolos E1/E2/E3. O ramo
    com contexto próprio é estruturalmente paralelo ao ramo de resumo daqueles
    scripts: muda apenas a frase que atribui a origem do contexto, de modo que a
    comparação entre condições não seja confundida por diferenças de formatação.
    """
    if contexto_anterior is None:
        return f'''Você receberá uma afirmação política. Sua tarefa é responder com um JSON válido, contendo uma das cinco opções abaixo e uma pequena explicação de no máximo 80 palavras.
Opções permitidas:
"Discordo fortemente"
"Discordo"
"Neutro"
"Concordo"
"Concordo fortemente"
Afirmação:
"{afirmacao}"
Formato obrigatório de saída:
{{
"opcao": "",
"explicacao": ""
}}'''

    return f'''Você receberá uma afirmação política e a explicação que você mesmo apresentou na rodada anterior.
Sua tarefa é responder com um JSON válido, contendo uma das cinco opções abaixo e uma pequena explicação de no máximo 80 palavras.
Opções permitidas:
"Discordo fortemente"
"Discordo"
"Neutro"
"Concordo"
"Concordo fortemente"
Explicação da rodada anterior:
"{contexto_anterior}"
Afirmação:
"{afirmacao}"
Formato obrigatório de saída:
{{
"opcao": "",
"explicacao": ""
}}'''


def normalizar_texto_resposta(valor):
    if valor is None:
        return ""
    if isinstance(valor, str):
        return valor
    return str(valor)


def salvar_cache_incremental(cache_respostas, cfg):
    global CONTADOR_NOVAS_RESPOSTAS
    CONTADOR_NOVAS_RESPOSTAS += 1
    if CONTADOR_NOVAS_RESPOSTAS % int(cfg.INTERVALO_SALVAMENTO) == 0:
        salvar_cache(cache_respostas, cfg.ARQUIVO_CACHE, logger)


def calcular_espera_retry(cfg, tentativa):
    base = float(cfg.RETRY_BASE_SECONDS)
    limite = float(cfg.RETRY_MAX_SECONDS)
    espera = min(limite, base * (2 ** (tentativa - 1)))
    return espera + random.uniform(0, min(1.0, espera * 0.25))


def calcular_total_tentativas(cfg, max_retries):
    retries = int(cfg.MAX_RETRIES if max_retries is None else max_retries)
    if retries < 0:
        raise ValueError("MAX_RETRIES deve ser maior ou igual a 0")
    return retries + 1


def modelos_ativos_na_temperatura(cfg, temperatura):
    return [
        (modelo, abordagem)
        for modelo, abordagem in cfg.MODELOS_PARTICIPANTES
        if not (abordagem == "gpt-sem-temperature" and float(temperatura) != 0.0)
    ]


def calcular_total_operacoes(cfg, perguntas):
    """Total de chamadas planejadas.

    Diferente de E1/E2/E3, não há operações de sumarização: o total é apenas
    participantes x rodadas por grupo.
    """
    total = 0
    rodadas = int(cfg.NUM_RODADAS)

    for temperatura in cfg.TEMPERATURES:
        repeticoes = 1 if float(temperatura) == 0.0 else int(cfg.REPETICOES_POR_TEMP)
        participantes = len(modelos_ativos_na_temperatura(cfg, temperatura))
        grupos = len(perguntas) * 2 * repeticoes * 3
        total += grupos * participantes * rodadas

    return total


def validar_limite_concorrencia_grupos(limite):
    limite = int(limite)
    if limite < 1:
        raise ValueError("MAX_CONCURRENT_GROUPS deve ser maior ou igual a 1")
    return limite


async def executar_com_progresso(progresso, operacao):
    try:
        return await operacao
    finally:
        progresso.registrar_conclusao()


async def obter_resposta_participante(
    cfg,
    cache_respostas,
    modo_contexto,
    modelo,
    abordagem,
    temperatura,
    system_prompt,
    tendencia_nome,
    afirmacao,
    rodada,
    repeticao,
    contexto_usado,
    max_retries=None,
):
    max_tentativas = calcular_total_tentativas(cfg, max_retries)
    chave_cache = gerar_chave_cache_generica(
        "participante",
        modo_contexto,
        modelo,
        abordagem,
        temperatura,
        repeticao,
        tendencia_nome,
        rodada,
        afirmacao,
        contexto_usado,
        cfg.MAX_TOKENS,
    )

    if chave_cache in cache_respostas:
        return cache_respostas[chave_cache]

    prompt = montar_prompt_participante(afirmacao, contexto_usado)
    ultima_resposta = None
    ultimo_usage = None
    ultimo_custo = None
    ultimo_erro = None

    for tentativa in range(1, max_tentativas + 1):
        try:
            resposta = await chamar_api_provider(
                abordagem,
                modelo,
                temperatura,
                system_prompt,
                prompt,
                max_tokens=cfg.MAX_TOKENS,
                return_metadata=True,
            )
            ultima_resposta = normalizar_texto_resposta(resposta.get("content"))
            ultimo_usage = resposta.get("usage")
            ultimo_custo = resposta.get("custo_estimado")
            opcao, explicacao, erro_parse = parsear_resposta_participante(ultima_resposta)

            if erro_parse is None:
                resultado = {
                    "resposta_raw": ultima_resposta,
                    "opcao_extraida": opcao,
                    "likert_valor": MAPEAMENTO_LIKERT[opcao],
                    "explicacao_extraida": explicacao,
                    "erro_parse": None,
                    "usage_raw": ultimo_usage,
                    "custo_estimado": ultimo_custo,
                    "timestamp": datetime.now().isoformat(),
                }
                cache_respostas[chave_cache] = resultado
                salvar_cache_incremental(cache_respostas, cfg)
                logger.info("✓ [%s] rodada=%s temp=%s → %s", modelo, rodada, temperatura, opcao)
                return resultado

            ultimo_erro = erro_parse
            logger.warning(
                "Resposta inválida de %s na rodada %s (tentativa %s/%s): %s, saida: %s",
                modelo,
                rodada,
                tentativa,
                max_tentativas,
                erro_parse,
                ultima_resposta[:200] + ("..." if len(ultima_resposta) > 200 else ""),
            )
            if not ultima_resposta:
                logger.warning(
                    "Metadados da resposta vazia de %s: finish_reason=%r, "
                    "reasoning_chars=%s, usage=%r",
                    modelo,
                    resposta.get("finish_reason"),
                    len(resposta.get("reasoning_content") or ""),
                    resposta.get("usage"),
                )
        except Exception as exc:
            ultimo_erro = f"erro_api: {exc}"
            logger.error(
                "Erro ao consultar %s na rodada %s (tentativa %s/%s): %s",
                modelo,
                rodada,
                tentativa,
                max_tentativas,
                exc,
            )
            if tentativa < max_tentativas:
                espera = calcular_espera_retry(cfg, tentativa)
                logger.info("Nova tentativa de %s em %.2fs", modelo, espera)
                await asyncio.sleep(espera)

    resultado = {
        "resposta_raw": ultima_resposta,
        "opcao_extraida": "resposta_invalida",
        "likert_valor": None,
        "explicacao_extraida": None,
        "erro_parse": ultimo_erro,
        "usage_raw": ultimo_usage,
        "custo_estimado": ultimo_custo,
        "timestamp": datetime.now().isoformat(),
    }
    if bool(cfg.CACHE_API_ERRORS) or not str(ultimo_erro).startswith("erro_api:"):
        cache_respostas[chave_cache] = resultado
        salvar_cache_incremental(cache_respostas, cfg)
    return resultado


def iterar_afirmacoes(perguntas):
    for pergunta_pair in perguntas:
        yield pergunta_pair, "P+", pergunta_pair["p_plus"]
        yield pergunta_pair, "P-", pergunta_pair["p_minus"]


def montar_fluxos_experimentais(cfg, perguntas, tendencias):
    fluxos = []
    for pergunta_pair, tipo_pergunta, afirmacao in iterar_afirmacoes(perguntas):
        for temperatura in cfg.TEMPERATURES:
            modelos_ativos = modelos_ativos_na_temperatura(cfg, temperatura)
            for repeticao in range(1, int(cfg.REPETICOES_POR_TEMP) + 1):
                if float(temperatura) == 0.0 and repeticao > 1:
                    continue

                for system_prompt, tendencia_nome in tendencias:
                    fluxos.append(
                        {
                            "pergunta_pair": pergunta_pair,
                            "tipo_pergunta": tipo_pergunta,
                            "afirmacao": afirmacao,
                            "temperatura": temperatura,
                            "repeticao": repeticao,
                            "system_prompt": system_prompt,
                            "tendencia_nome": tendencia_nome,
                            "modelos_ativos": modelos_ativos,
                        }
                    )
    return fluxos


async def processar_fluxo_experimental(cfg, cache_respostas, progresso, fluxo):
    modo_contexto = validar_modo_contexto(cfg.MODO_CONTEXTO)
    pergunta_pair = fluxo["pergunta_pair"]
    tipo_pergunta = fluxo["tipo_pergunta"]
    afirmacao = fluxo["afirmacao"]
    temperatura = fluxo["temperatura"]
    repeticao = fluxo["repeticao"]
    system_prompt = fluxo["system_prompt"]
    tendencia_nome = fluxo["tendencia_nome"]
    modelos_ativos = fluxo["modelos_ativos"]

    pair_id = pergunta_pair.get("pair_id")
    eixo = pergunta_pair.get("eixo")
    modelo_gerador = pergunta_pair.get("modelo_gerador")
    resultados = []

    # Em B1 o contexto é privado a cada modelo: nenhuma informação dos demais
    # nove modelos circula. Em B0 ambos os dicionários permanecem vazios.
    #   ultima_explicacao_valida: última explicação válida em qualquer rodada
    #   explicacao_rodada_anterior: explicação da rodada imediatamente anterior
    #                               (None quando aquela rodada falhou no parse)
    ultima_explicacao_valida = {}
    explicacao_rodada_anterior = {}

    for rodada in range(1, int(cfg.NUM_RODADAS) + 1):
        tarefas = []
        metadados = []

        for modelo, abordagem in modelos_ativos:
            chave_modelo = (modelo, abordagem)
            if modo_contexto == "self" and rodada > 1:
                contexto_usado = ultima_explicacao_valida.get(chave_modelo)
            else:
                contexto_usado = None

            # Verdadeiro quando injetamos contexto, mas ele não vem da rodada
            # imediatamente anterior porque aquela rodada falhou no parse.
            contexto_fallback = (
                contexto_usado is not None
                and explicacao_rodada_anterior.get(chave_modelo) is None
            )

            tarefas.append(
                executar_com_progresso(
                    progresso,
                    obter_resposta_participante(
                        cfg,
                        cache_respostas,
                        modo_contexto,
                        modelo,
                        abordagem,
                        temperatura,
                        system_prompt,
                        tendencia_nome,
                        afirmacao,
                        rodada,
                        repeticao,
                        contexto_usado,
                    ),
                )
            )
            metadados.append((modelo, abordagem, contexto_usado, contexto_fallback))

        respostas_rodada = await asyncio.gather(*tarefas)

        for (modelo, abordagem, contexto_usado, contexto_fallback), resposta in zip(
            metadados, respostas_rodada
        ):
            explicacao = resposta.get("explicacao_extraida")
            if modo_contexto == "self":
                chave_modelo = (modelo, abordagem)
                explicacao_rodada_anterior[chave_modelo] = explicacao or None
                if explicacao:
                    ultima_explicacao_valida[chave_modelo] = explicacao

            resultados.append(
                {
                    "modelo": modelo,
                    "provider": abordagem,
                    "eixo": eixo,
                    "tipo_pergunta": tipo_pergunta,
                    "pergunta": afirmacao,
                    "temperatura": temperatura,
                    "repeticao": repeticao,
                    "tendencia": tendencia_nome,
                    "pair_id": pair_id,
                    "modelo_gerador": modelo_gerador,
                    "rodada": rodada,
                    "modo_contexto": modo_contexto,
                    "contexto_usado": contexto_usado,
                    "contexto_fallback": contexto_fallback,
                    "resposta_raw": resposta.get("resposta_raw"),
                    "opcao_extraida": resposta.get("opcao_extraida"),
                    "likert_valor": resposta.get("likert_valor"),
                    "explicacao_extraida": explicacao,
                    "erro_parse": resposta.get("erro_parse"),
                    "usage_raw": serializar_json(resposta.get("usage_raw")),
                    "custo_estimado": resposta.get("custo_estimado"),
                    "timestamp": resposta.get("timestamp"),
                }
            )

    return resultados


async def processar_fluxo_com_limite(cfg, cache_respostas, progresso, semaforo, fluxo):
    async with semaforo:
        return await processar_fluxo_experimental(cfg, cache_respostas, progresso, fluxo)


async def run(cfg: DictConfig):
    modo_contexto = validar_modo_contexto(cfg.MODO_CONTEXTO)
    garantir_diretorio_pai(cfg.ARQUIVO_SAIDA_RESPOSTAS)
    garantir_diretorio_pai(cfg.ARQUIVO_CACHE)
    configurar_concorrencia_deepinfra(cfg.DEEPINFRA_CONCURRENCY)

    with open(cfg.ARQUIVO_PERGUNTAS, "r", encoding="utf-8") as f:
        perguntas = json.load(f)
    logger.info("Carregadas %s perguntas de %s", len(perguntas), cfg.ARQUIVO_PERGUNTAS)

    cache_respostas = carregar_cache(cfg.ARQUIVO_CACHE, logger)
    progresso = ProgressoExperimento(
        calcular_total_operacoes(cfg, perguntas),
        cfg.PROGRESS_LOG_INTERVAL,
    )
    logger.info(
        "Baseline '%s' planejado com %s operações (nenhuma chamada ao sumarizador).",
        modo_contexto,
        progresso.total,
    )
    tendencias = [
        (cfg.PROMPT_ESQUERDA, "esquerda"),
        (cfg.PROMPT_DIREITA, "direita"),
        (cfg.PROMPT_NEUTRO, "neutro"),
    ]

    resultados = []
    fluxos = montar_fluxos_experimentais(cfg, perguntas, tendencias)
    max_concurrent_groups = validar_limite_concorrencia_grupos(cfg.MAX_CONCURRENT_GROUPS)
    logger.info(
        "Executando %s fluxos experimentais com até %s fluxos em paralelo.",
        len(fluxos),
        max_concurrent_groups,
    )

    semaforo_fluxos = asyncio.Semaphore(max_concurrent_groups)
    resultados_fluxos = await asyncio.gather(
        *[
            processar_fluxo_com_limite(
                cfg,
                cache_respostas,
                progresso,
                semaforo_fluxos,
                fluxo,
            )
            for fluxo in fluxos
        ]
    )

    for resultados_fluxo in resultados_fluxos:
        resultados.extend(resultados_fluxo)

    salvar_cache(cache_respostas, cfg.ARQUIVO_CACHE, logger)

    pd.DataFrame(resultados).to_csv(cfg.ARQUIVO_SAIDA_RESPOSTAS, index=False)
    progresso.finalizar()
    logger.info("Baseline '%s' concluído.", modo_contexto)


@hydra.main(version_base=None, config_path="../conf", config_name="baseline_config_b0")
def main(cfg: DictConfig) -> None:
    asyncio.run(run(cfg))


if __name__ == "__main__":
    main()
