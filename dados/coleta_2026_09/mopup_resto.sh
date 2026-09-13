#!/bin/zsh
# Limpeza dos braços que faltam (b0 já concluída em 04/09).
# Só recupera células que falharam com erro de API — respostas válidas vêm do
# cache sem chamada. Concorrência baixa porque a causa foi 429 do DeepSeek.
set -u
cd "/Users/anonymous/Documents/[ICAART]opinion_stability"
D=dados/coleta_2026_09
for par in "baseline-b1 b1" "iterative e1" "iterative-v2 e2" "iterative-v3 e3"; do
  set -- ${=par}; task=$1; slug=$2
  echo "[$(date)] === limpeza $slug ==="
  extra=()
  [[ $slug == e* ]] && extra=(ARQUIVO_SAIDA_RESUMOS=$D/$slug/resumos_$slug.csv)
  pixi run "$task" \
    MAX_RETRIES=8 RETRY_MAX_SECONDS=60 MAX_CONCURRENT_GROUPS=8 DEEPINFRA_CONCURRENCY=40 \
    ARQUIVO_CACHE=$D/$slug/cache.pkl \
    ARQUIVO_SAIDA_RESPOSTAS=$D/$slug/respostas_$slug.csv \
    "${extra[@]}" \
    hydra.run.dir=$D/$slug/hydra_mop \
    > $D/$slug/mopup.log 2>&1
  echo "[$(date)] === limpeza $slug status $? ==="
done
echo "[$(date)] LIMPEZA COMPLETA"
