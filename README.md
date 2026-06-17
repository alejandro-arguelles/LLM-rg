# LLM-rg

LLM-rg is an early-stage Python workspace for language-model research
experiments. The repository is currently scaffolded for data preparation and
pretraining work, with project dependencies managed through `uv`.

## Current Status

This project is still in its initial setup phase.

- `data/prepare.py` exists as the expected entry point for dataset preparation.
- `pretraining/` is reserved for model training and experiment code.
- The dependency set already includes common LLM research tools such as
  `datasets`, `torch`, `transformers`, `numpy`, `tqdm`, and `pytest`.

## Requirements

- Python 3.10 or newer
- `uv` for dependency and virtual environment management

The checked-in `.python-version` file pins local development to Python 3.10.

## Setup

Clone the repository and install the project dependencies:

```bash
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

## Project Layout

```text
.
|-- data/
|   `-- prepare.py       # Dataset preparation entry point
|-- pretraining/         # Pretraining experiments and training code
|-- pyproject.toml       # Project metadata and dependencies
|-- uv.lock              # Locked dependency versions
`-- README.md
```

## Development

Run tests with:

```bash
uv run pytest
```

As the project grows, keep experiment scripts reproducible by documenting:

- input datasets and preprocessing steps
- model and tokenizer choices
- training configuration
- hardware assumptions
- output artifacts and evaluation metrics

## License

This project is licensed under the BSD 3-Clause License. See
[`LICENSE`](LICENSE) for details.
