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

Clone the repository and run the setup script. It installs `uv`, makes sure it
is on the `PATH` (also on remote machines) and installs all dependencies with
`uv sync`:

```bash
bash setup.sh
```

To also download 10 climbmix shards (~0.9 GB) into `$HOME/climbmix` while
setting up, pass `--data` (override the target with `DATA_DIR`):

```bash
bash setup.sh --data
# or pick another directory:
DATA_DIR=/path/to/climbmix bash setup.sh --data
```

Remember to pass the same `DATA_DIR` when training, e.g.
`DATA_DIR=$HOME/climbmix ... uv run python scripts/run_climbing.py`.

<details>
<summary>Manual setup (if you prefer step by step)</summary>

Install uv:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

On remote machines make uv available in the current shell:
```bash
source $HOME/.local/bin/env
```

Get all the needed dependencies:
```bash
uv sync
```
</details>

Then run commands inside the managed environment:

```bash
uv run python --version
```

If you prefer to activate the environment manually:

```bash
source .venv/bin/activate
```

## Running

### Climbing model

Download the dataset (see Setup with `--data`) and train with the helper script:

```bash
bash train.sh                          # 500 iterations, data in $HOME/climbmix
bash train.sh --iter 1000              # more iterations
bash train.sh --iter 500 --batch 8 --data /path/to/climbmix
```

The checkpoint is written to `checkpoints/tinyllm_climbing.pth`.

### Shakespeare model

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
uv run python -m scripts.eval_hellaswag --checkpoint checkpoints/tinyllm_climbing.pth --tokenizer gpt2 --max-examples 100


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
|-- setup.sh             # Install uv + dependencies (and optionally data)
|-- train.sh             # Train the climbing model
|-- pyproject.toml       # Project metadata and dependencies
|-- uv.lock              # Locked dependency versions
`-- README.md
```

## License

This project is licensed under the BSD 3-Clause License. See
[`LICENSE`](LICENSE) for details.



---------to resume
RESUME=1 DATA_DIR=$HOME/climbmix BATCH=64 NUM_ITERS=600000 \
  PYTHONPATH=. uv run python scripts/run_climbing.py



------- TO ANALYZE -----------

Cómo usarlo

# todo (por defecto en CPU, para no molestar al entrenamiento en GPU)
PYTHONPATH=. uv run python tinyllm/analyzer.py --all

# solo evolución, con tu prompt
PYTHONPATH=. uv run python tinyllm/analyzer.py --evolution --prompt "Climbing is"

# atención con capas/heads concretos
PYTHONPATH=. uv run python tinyllm/analyzer.py --attention --layers 0,5,11 --heads 0,3,7,11





 s
-----------projector

PYTHONPATH=. uv run python tinyllm/analyzer.py \
    --projector \ 
    --ckpt checkpoints/tinyllm_climbing.pth \
    --data-dir $HOME/climbmix \
    --projector-count 1000 \
    --projector-layer -1 \
    --device cpu

    Abre https://projector.tensorflow.org
Botón "Load" → sube los dos .tsv (Step 1 = vectors.tsv, Step 2 = metadata.tsv). Todo corre en tu navegador, no se sube nada a ningún servidor.
Abajo a la izquierda elige UMAP o t-SNE y marca 3D.
Pasa el ratón por los puntos para ver la frase; usa el buscador y "isolate N points" para inspeccionar zonas / vecinos.



------------ to resume
RESUME=1 DATA_DIR=/root/climbmix BATCH=64 NUM_ITERS=600000 PYTHONPATH=. uv run python scripts/run_climbing.py