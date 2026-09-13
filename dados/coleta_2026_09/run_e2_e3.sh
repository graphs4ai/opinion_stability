#!/bin/zsh
# E2 e E3 em paralelo, com concorrência maior.
#
# Diagnóstico que motivou isto: comparando B0 (24 grupos, sem sumarizador,
# 24 min) com B1 (8 grupos, sem sumarizador, 90 min), o gargalo é
# MAX_CONCURRENT_GROUPS, não o sumarizador — o E1 COM sumarizador levou 81 min,
# menos que o B1 sem. Os 429 do DeepSeek são praticamente constantes entre 8 e
# 24 grupos, então subir a concorrência não os agrava.
#
# 16 grupos por braço, dois braços = 32 em agregado, um pouco acima dos 24 do
# B0. Os caches existentes são reaproveitados: o E2 já tem ~40% feito.
set -u
cd "/Users/anonymous/Documents/[ICAART]opinion_stability"
D=dados/coleta_2026_09
GRUPOS=16

run() {
  local task=$1 slug=$2
  echo "[$(date)] === iniciando $slug (grupos=$GRUPOS) ==="
  pixi run "$task" \
    MAX_RETRIES=5 RETRY_MAX_SECONDS=30 MAX_CONCURRENT_GROUPS=$GRUPOS \
    ARQUIVO_CACHE=$D/$slug/cache.pkl \
    ARQUIVO_SAIDA_RESPOSTAS=$D/$slug/respostas_$slug.csv \
    ARQUIVO_SAIDA_RESUMOS=$D/$slug/resumos_$slug.csv \
    hydra.run.dir=$D/$slug/hydra_v2 \
    > $D/$slug/run.log 2>&1
  echo "[$(date)] === $slug terminou com status $? ==="
}

run iterative-v2 e2 &
PID_E2=$!
run iterative-v3 e3 &
PID_E3=$!
wait $PID_E2 $PID_E3
echo "[$(date)] E2 E E3 COMPLETOS"

# Passe de limpeza: só recupera células que falharam com erro de API (não
# cacheadas). Concorrência baixa, já que a causa foi 429.
for par in "baseline-b0 b0" "baseline-b1 b1" "iterative e1" "iterative-v2 e2" "iterative-v3 e3"; do
  set -- ${=par}
  task=$1; slug=$2
  echo "[$(date)] === limpeza $slug ==="
  extra=()
  [[ $slug == e* ]] && extra=(ARQUIVO_SAIDA_RESUMOS=$D/$slug/resumos_$slug.csv)
  pixi run "$task" \
    MAX_RETRIES=8 RETRY_MAX_SECONDS=60 MAX_CONCURRENT_GROUPS=4 DEEPINFRA_CONCURRENCY=25 \
    ARQUIVO_CACHE=$D/$slug/cache.pkl \
    ARQUIVO_SAIDA_RESPOSTAS=$D/$slug/respostas_$slug.csv \
    "${extra[@]}" \
    hydra.run.dir=$D/$slug/hydra_mop \
    > $D/$slug/mopup.log 2>&1
  echo "[$(date)] === limpeza $slug status $? ==="
done
echo "[$(date)] TUDO COMPLETO"
wc -l $D/*/respostas_*.csv
