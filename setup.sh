#!/usr/bin/env bash
# Instala todo de cero: uv + dependencias del proyecto (uv sync).
# Uso:
#   bash setup.sh            -> instala uv y hace uv sync
#   bash setup.sh --data     -> ademas descarga 10 shards de climbmix
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

# 1) Instalar uv si no esta disponible
if ! command -v uv >/dev/null 2>&1; then
    echo ">> Instalando uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi

# 2) Asegurar que uv esta en el PATH (necesario en maquinas remotas)
export PATH="$HOME/.local/bin:$PATH"
[ -f "$HOME/.local/bin/env" ] && source "$HOME/.local/bin/env"

# 3) Instalar dependencias del proyecto
echo ">> Sincronizando dependencias (uv sync)..."
uv sync
source .venv/bin/activate

# 4) (Opcional) Descargar dataset climbmix con --data
if [ "${1:-}" = "--data" ]; then
    DATA_DIR="${DATA_DIR:-$HOME/climbmix}"
    echo ">> Descargando 10 shards de climbmix en $DATA_DIR ..."
    DATA_DIR="$DATA_DIR" uv run python -c "
import os
from huggingface_hub import hf_hub_download
data_dir = os.environ['DATA_DIR']
os.makedirs(data_dir, exist_ok=True)
for i in range(10):
    hf_hub_download(repo_id='karpathy/climbmix-400b-shuffle', repo_type='dataset',
                    filename=f'shard_{i:05d}.parquet', local_dir=data_dir)
print('dataset listo en', data_dir)
"
fi


