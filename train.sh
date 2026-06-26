#!/usr/bin/env bash
# Entrena el modelo climbing.
# Uso:
#   bash train.sh                     -> 500 iteraciones, datos en $HOME/climbmix
#   bash train.sh --iter 1000         -> 1000 iteraciones
#   bash train.sh --iter 500 --batch 8 --data /ruta/climbmix
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

# Valores por defecto
NUM_ITERS=500
BATCH=8
DATA_DIR="${DATA_DIR:-$HOME/climbmix}"

# Parseo de argumentos
while [ $# -gt 0 ]; do
    case "$1" in
        --iter)  NUM_ITERS="$2"; shift 2 ;;
        --batch) BATCH="$2"; shift 2 ;;
        --data)  DATA_DIR="$2"; shift 2 ;;
        *) echo "Argumento desconocido: $1"; exit 1 ;;
    esac
done

# Asegurar que uv esta en el PATH (maquinas remotas)
export PATH="$HOME/.local/bin:$PATH"
[ -f "$HOME/.local/bin/env" ] && source "$HOME/.local/bin/env"

echo ">> Entrenando: NUM_ITERS=$NUM_ITERS BATCH=$BATCH DATA_DIR=$DATA_DIR"
DATA_DIR="$DATA_DIR" BATCH="$BATCH" NUM_ITERS="$NUM_ITERS" \
    PYTHONPATH=. uv run python scripts/run_climbing.py

echo ">> Entrenamiento terminado. Checkpoint en checkpoints/tinyllm_climbing.pth"
