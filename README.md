# LLM-rg

LLM-rg is a small Python workspace for training and evaluating tiny
decoder-only language models. The project includes a Shakespeare training setup
and larger training configurations that can be selected from the training entry
point.

## Current Status

- `tinyllm/train.py` contains the main training loop and experiment
  configurations.
- The default training entry point runs `train_shakespeare()`.
- Training checkpoints are written under `checkpoints/` by the training
  functions in `tinyllm/train.py`.
- Larger training runs can be configured in `tinyllm/train.py` by replacing the
  call to `train_shakespeare()` with another training function, for example a
  `train_something()` configuration.
- The project uses `torch`, `numpy`, `tiktoken`, `datasets`, `transformers`,
  `tqdm`, and `pytest`.

## Requirements

- Python 3.10 or newer
- `uv` for dependency and virtual environment management

The checked-in `.python-version` file pins local development to Python 3.10.

## Setup

Clone the repository and install the project dependencies:

Install uv
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Get all the needed dependencies
```
uv sync
```

Then run commands inside the managed environment:

```bash
uv run python --version
```

If you prefer to activate the environment manually:

```bash
source .venv/bin/activate
```

## Running

Prepare the Shakespeare dataset first:

```bash
uv run python scripts/create_shakespeare_dataset.py
```

Then run the default training job from the repository root:

```bash
uv run python -m tinyllm.train
```

By default this launches the Shakespeare training configuration. To run a larger
project, edit `tinyllm/train.py` and change the final call from
`train_shakespeare()` to the training configuration you want to use, such as
`train_something()`.

## Evaluation

After training has produced a checkpoint, evaluate it on HellaSwag with:

```bash
uv run python -m scripts.eval_hellaswag --checkpoint checkpoints/tinyllm_shakespeare.pth --tokenizer shakespeare
```

For a model trained with the GPT-2 tokenizer, use:

```bash
uv run python -m scripts.eval_hellaswag --checkpoint checkpoints/tinyllm_climbing.pth --tokenizer gpt2
```

The script evaluates the `validation` split by default. For a quick smoke test,
limit the number of examples:

```bash
uv run python -m scripts.eval_hellaswag --checkpoint checkpoints/tinyllm_shakespeare.pth --tokenizer shakespeare --max-examples 100
```

## Project Layout

```text
.
|-- tinyllm/
|   |-- train.py         # Training loop and experiment configurations
|   |-- models.py        # Decoder transformer model
|   |-- load_data.py     # Dataset loading utilities
|   `-- hellaswag.py     # HellaSwag evaluation utilities
|-- datasets/            # Prepared local datasets
|-- scripts/             # Dataset and evaluation scripts
|-- checkpoints/         # Local model checkpoints
|-- pyproject.toml       # Project metadata and dependencies
|-- uv.lock              # Locked dependency versions
`-- README.md
```

## License

This project is licensed under the BSD 3-Clause License. See
[`LICENSE`](LICENSE) for details.



source $HOME/.local/bin/env so that it works in remote machine before doing de uv sync


-------------climb
export PATH="$HOME/.local/bin:$PATH"
cd /root/LLM-rg

# 1) cargar dataset = comprobar que los parquets están
ls /home/criteo/.cache/nanochat/base_data_climbmix/*.parquet | head

# 2) entrenar (ajusta BATCH a tu GPU; 64 necesita ~A100 80GB)
DATA_DIR=/home/criteo/.cache/nanochat/base_data_climbmix BATCH=8 NUM_ITERS=500 \
  PYTHONPATH=. uv run python /tmp/run_climbing.py

# 3) evaluar en HellaSwag
uv run python -m scripts.eval_hellaswag --checkpoint checkpoints/tinyllm_climbing.pth --tokenizer gpt2 --max-examples 200


--------------------climb



export PATH="$HOME/.local/bin:$PATH"
cd /root/LLM-rg

# Descarga 10 shards (~0.9 GB, ~4B tokens) — cuidado: solo te quedan ~5GB de disco
uv run python -c "
from huggingface_hub import hf_hub_download
import os; os.makedirs('/root/climbmix', exist_ok=True)
for i in range(10):
    hf_hub_download(repo_id='karpathy/climbmix-400b-shuffle', repo_type='dataset',
                    filename=f'shard_{i:05d}.parquet', local_dir='/root/climbmix')
print('listo')
"

NUM_ITERS=20000 PYTHONPATH=. uv run python /tmp/run_climbing.py



uv run python -m scripts.eval_hellaswag --checkpoint checkpoints/tinyllm_climbing.pth --tokenizer gpt2 --max-examples 200

