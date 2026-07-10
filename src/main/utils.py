import hashlib
import os
import pickle
from collections.abc import Mapping
from openai import OpenAI, AsyncOpenAI
from xai_sdk import Client
from xai_sdk.chat import user, system
from dotenv import load_dotenv
from google import genai
import ollama
import asyncio

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OPEN_AI_API_KEY = os.getenv("OPEN_AI_API_KEY")
GROK_API_KEY = os.getenv("GROK_API_KEY")
DEEPINFRA_API_KEY = os.getenv("DEEPINFRA_API_KEY")
MARITACA_API_KEY = os.getenv("MARITACA_API_KEY")
OLLAMA_HOST = os.getenv("OLLAMA_HOST")

# Inicializar clientes de API
client_deepinfra = AsyncOpenAI(
    api_key=DEEPINFRA_API_KEY,
    base_url="https://api.deepinfra.com/v1/openai",
    max_retries=0,
)
client_grok = Client(api_key=GROK_API_KEY)
client_genai = genai.Client(api_key=GEMINI_API_KEY)
client_ollama = ollama.Client(host=OLLAMA_HOST)
client_openai = OpenAI(api_key=OPEN_AI_API_KEY)
cliente_maritaca = OpenAI(api_key=MARITACA_API_KEY, base_url="https://chat.maritaca.ai/api")
    

gemini_semaphore = asyncio.Semaphore(10)
deepinfra_semaphore = asyncio.Semaphore(10)

def configurar_concorrencia_deepinfra(limite):
    """Atualiza o limite de chamadas simultâneas à DeepInfra."""
    global deepinfra_semaphore
    limite = int(limite)
    if limite < 1:
        raise ValueError("DEEPINFRA_CONCURRENCY deve ser maior ou igual a 1")
    deepinfra_semaphore = asyncio.Semaphore(limite)

def gerar_chave_cache(modelo, afirmacao, temperatura, repeticao, tendencia_prompt):
    """Gera uma chave única incluindo a repetição."""
    chave = f"{modelo}|{afirmacao}|{temperatura}|{repeticao}|{tendencia_prompt}"
    return hashlib.md5(chave.encode()).hexdigest()

def gerar_chave_cache_generica(*partes):
    """Gera uma chave estável para fluxos que precisam de mais dimensões."""
    chave = "|".join("" if parte is None else str(parte) for parte in partes)
    return hashlib.md5(chave.encode()).hexdigest()

def _to_plain_data(obj):
    if obj is None:
        return None
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    if isinstance(obj, Mapping):
        return {k: _to_plain_data(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_plain_data(v) for v in obj]
    return obj

def _get_attr_or_key(obj, name, default=None):
    if obj is None:
        return default
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)

def _extrair_primeira_mensagem_chat(raw_response):
    choices = _get_attr_or_key(raw_response, "choices", []) or []
    first_choice = choices[0] if choices else None
    return first_choice, _get_attr_or_key(first_choice, "message")

def _normalizar_conteudo(content):
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        partes = []
        for item in content:
            texto = _get_attr_or_key(item, "text")
            if texto is None:
                texto = _get_attr_or_key(item, "content")
            if texto is not None:
                partes.append(str(texto))
        return "".join(partes) if partes else str(content)
    return str(content)

def extrair_custo_estimado(usage):
    """Procura campos de custo reportados diretamente pelo provider."""
    usage_plain = _to_plain_data(usage)
    if not isinstance(usage_plain, Mapping):
        return None

    nomes_custo = {
        "estimated_cost",
        "estimated_cost_usd",
        "cost",
        "cost_usd",
        "total_cost",
        "total_cost_usd",
    }

    def buscar(valor):
        if isinstance(valor, Mapping):
            for chave, conteudo in valor.items():
                if str(chave).lower() in nomes_custo:
                    return conteudo
                encontrado = buscar(conteudo)
                if encontrado is not None:
                    return encontrado
        elif isinstance(valor, list):
            for item in valor:
                encontrado = buscar(item)
                if encontrado is not None:
                    return encontrado
        return None

    return buscar(usage_plain)

def carregar_cache(ARQUIVO_CACHE, logger):
    """Carrega o cache de respostas anteriores."""
    if os.path.exists(ARQUIVO_CACHE):
        try:
            with open(ARQUIVO_CACHE, 'rb') as f:
                cache = pickle.load(f)
                logger.info(f"Cache carregado com {len(cache)} entradas")
                return cache
        except Exception as e:
            logger.warning(f"Erro ao carregar cache: {e}")
            return {}
    return {}

def salvar_cache(cache, ARQUIVO_CACHE, logger):
    """Salva o cache de respostas."""
    try:
        with open(ARQUIVO_CACHE, 'wb') as f:
            pickle.dump(cache, f)
        logger.info(f"Cache salvo com {len(cache)} entradas")
    except Exception as e:
        logger.error(f"Erro ao salvar cache: {e}")

def validar_resposta(resposta_limpa, OPCOES_VALIDAS):
    """
    Valida se a resposta está entre as opções permitidas.
    Retorna a resposta válida ou None se inválida.
    """
    # Verificação exata
    if resposta_limpa in OPCOES_VALIDAS:
        return resposta_limpa
    
    # Tentar encontrar uma opção válida na resposta (fallback)
    for opcao in OPCOES_VALIDAS:
        if opcao in resposta_limpa:
            return opcao
    
    return None

def atualizar_cache_e_salvar_se_necessario(CONTADOR_NOVAS_RESPOSTAS, chave, valor, cache_respostas, ARQUIVO_CACHE, INTERVALO_SALVAMENTO, logger):
    cache_respostas[chave] = valor
    CONTADOR_NOVAS_RESPOSTAS += 1
    if CONTADOR_NOVAS_RESPOSTAS % INTERVALO_SALVAMENTO == 0:
        logger.info(f"Salvamento incremental ({CONTADOR_NOVAS_RESPOSTAS} novas respostas)...")
        try:
            salvar_cache(cache_respostas, ARQUIVO_CACHE, logger) 
        except Exception as e:
            logger.error(f"Erro no salvamento incremental: {e}")
    return CONTADOR_NOVAS_RESPOSTAS


async def chamar_api_provider(
    abordagem,
    modelo,
    temperatura,
    system_prompt,
    user_prompt,
    max_tokens=None,
    return_metadata=False,
    reasoning_effort=None,
):
    response_content = ""
    raw_response = None
    if abordagem == 'ollama':
        options = {'temperature': temperatura}
        if max_tokens is not None:
            options['num_predict'] = max_tokens
        response = client_ollama.chat(
            model=modelo,
            messages=[{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': user_prompt}],
            options=options
        )
        raw_response = response
        response_content = response['message']['content']

    elif abordagem == 'gemini':
        async with gemini_semaphore:
            config_kwargs = {"temperature": temperatura}
            if system_prompt:
                config_kwargs["system_instruction"] = system_prompt
            if max_tokens is not None:
                config_kwargs["max_output_tokens"] = max_tokens
            gemini_resposta = await client_genai.aio.models.generate_content(
                model=modelo,
                contents=user_prompt,
                config=genai.types.GenerateContentConfig(**config_kwargs)
            )
            raw_response = gemini_resposta
            response_content = gemini_resposta.text

    elif abordagem in ['gpt', 'gpt-sem-temperature']:
        kwargs = {
            "model": modelo,
            "input": [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        }
        if max_tokens is not None:
            kwargs["max_output_tokens"] = max_tokens
        if abordagem == 'gpt':
             kwargs["temperature"] = temperatura
        
        gpt_resposta = client_openai.responses.create(**kwargs)
        raw_response = gpt_resposta
        response_content = gpt_resposta.output_text
    elif abordagem == 'deepinfra':
        kwargs = {
            "model": modelo,
            "temperature": temperatura,
            "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        }
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        if reasoning_effort is not None:
            kwargs["reasoning_effort"] = reasoning_effort
        async with deepinfra_semaphore:
            deepinfra_resposta = await client_deepinfra.chat.completions.create(**kwargs)
            raw_response = deepinfra_resposta
            _, message = _extrair_primeira_mensagem_chat(deepinfra_resposta)
            response_content = _normalizar_conteudo(
                _get_attr_or_key(message, "content")
            )
    elif abordagem == 'maritaca':
        kwargs = {
            "model": modelo,
            "temperature": temperatura,
            "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        }
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        maritaca_resposta = cliente_maritaca.chat.completions.create(**kwargs)
        raw_response = maritaca_resposta
        response_content = maritaca_resposta.choices[0].message.content
    elif abordagem == 'grok':
        kwargs = {"model": modelo, "temperature": temperatura}
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        chat = client_grok.chat.create(**kwargs)
        chat.append(system(system_prompt))
        chat.append(user(user_prompt))
        grok_resposta = chat.sample()
        raw_response = grok_resposta
        response_content = grok_resposta.content

    if return_metadata:
        usage = _get_attr_or_key(raw_response, "usage")
        first_choice, message = _extrair_primeira_mensagem_chat(raw_response)
        return {
            "content": _normalizar_conteudo(response_content),
            "usage": _to_plain_data(usage),
            "custo_estimado": extrair_custo_estimado(usage),
            "response_id": _get_attr_or_key(raw_response, "id"),
            "finish_reason": _get_attr_or_key(first_choice, "finish_reason"),
            "reasoning_content": _get_attr_or_key(message, "reasoning_content"),
        }

    return response_content
