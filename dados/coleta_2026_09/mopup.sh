#!/bin/zsh
# Passe de limpeza: reexecuta cada braço com o MESMO cache.
# Respostas válidas já estão cacheadas e não geram chamada; apenas as células
# que falharam com erro de API (não cacheadas, CACHE_API_ERRORS: false) são
# retentadas. Concorrência baixa, porque a causa das falhas foi 429.
set -u
cd "/Users/anonymous/Documents/[ICAART]opinion_stability"
D=dados/coleta_2026_09

while ps -p "$1" > /dev/null 2>&1; do sleep 60; done
echo "[$(date)] coleta principal terminou; iniciando passe de limpeza."

mop() {
  local task=$1 slug=$2; shift 2
  echo "[$(date)] === limpeza $slug ==="
  pixi run "$task" \
    MAX_RETRIES=8 RETRY_MAX_SECONDS=60 MAX_CONCURRENT_GROUPS=4 DEEPINFRA_CONCURRENCY=25 \
    ARQUIVO_CACHE=$D/$slug/cache.pkl \
    ARQUIVO_SAIDA_RESPOSTAS=$D/$slug/respostas_$slug.csv \
    "$@" \
    hydra.run.dir=$D/$slug/hydra_mop \
    > $D/$slug/mopup.log 2>&1
  echo "[$(date)] === limpeza $slug status $? ==="
}

mop baseline-b0 b0
mop baseline-b1 b1
mop iterative    e1 ARQUIVO_SAIDA_RESUMOS=$D/e1/resumos_e1.csv
mop iterative-v2 e2 ARQUIVO_SAIDA_RESUMOS=$D/e2/resumos_e2.csv
mop iterative-v3 e3 ARQUIVO_SAIDA_RESUMOS=$D/e3/resumos_e3.csv

echo "[$(date)] LIMPEZA COMPLETA"
