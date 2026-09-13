#!/bin/zsh
# Recoleta contemporânea das condições restantes (B1, E1, E2, E3).
# B0 já está em execução; aguardamos seu término antes de começar.
# Caches novos são obrigatórios: reusar os de julho faria tudo bater em cache
# e nada seria efetivamente recoletado.
set -u
cd "/Users/anonymous/Documents/[ICAART]opinion_stability"
D=dados/coleta_2026_09

while ps -p "$1" > /dev/null 2>&1; do sleep 20; done
echo "[$(date)] B0 terminou; iniciando os demais braços."

run() {  # run <task> <slug> <extra-overrides...>
  local task=$1 slug=$2; shift 2
  echo "[$(date)] === iniciando $slug ==="
  pixi run "$task" \
    MAX_RETRIES=5 RETRY_MAX_SECONDS=30 \
    ARQUIVO_CACHE=$D/$slug/cache.pkl \
    ARQUIVO_SAIDA_RESPOSTAS=$D/$slug/respostas_$slug.csv \
    "$@" \
    hydra.run.dir=$D/$slug/hydra \
    > $D/$slug/run.log 2>&1
  echo "[$(date)] === $slug terminou com status $? ==="
}

run baseline-b1 b1
run iterative    e1 ARQUIVO_SAIDA_RESUMOS=$D/e1/resumos_e1.csv
run iterative-v2 e2 ARQUIVO_SAIDA_RESUMOS=$D/e2/resumos_e2.csv
run iterative-v3 e3 ARQUIVO_SAIDA_RESUMOS=$D/e3/resumos_e3.csv

echo "[$(date)] COLETA COMPLETA"
wc -l $D/*/respostas_*.csv
