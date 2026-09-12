# PyTorch DL Lab 🧪

[![Python](https://img.shields.io/badge/Python-3.13+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.11+-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-notebooks-F37626?logo=jupyter&logoColor=white)](https://jupyter.org/)
[![uv](https://img.shields.io/badge/uv-managed-DE5FE9?logo=uv&logoColor=white)](https://docs.astral.sh/uv/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](./LICENSE.md)

**English** · [Русский](./README.ru.md)

This is my deep learning playground built around PyTorch. I use it to move from
basic tensor operations to complete neural networks, implementing the important
parts myself instead of treating the framework as a black box.

The project is organized as ten Jupyter notebook chapters. It starts with a
linear regression training loop, continues through MLPs, CNNs and recurrent
networks, and ends with machine translation and Russian news summarization.

The long-term goal is to bring all of these pieces together and build a small
language model from scratch — from tokenization and Transformer blocks to the
training loop and text generation.

## 📚 What's inside

- tensor basics and data preparation;
- linear regression trained with a handwritten loop;
- MLPs for MNIST and Fashion-MNIST;
- a CNN and a direct comparison with a fully connected network;
- simple RNN, LSTM and GRU models;
- sequence-to-sequence models with teacher forcing and Bahdanau attention;
- a Transformer encoder-decoder for abstractive summarization;
- Optuna experiments for tuning the summarizer.

The later chapters reuse code from `src/dl_roadmap/`, so the repository also
contains a small training framework rather than only isolated notebooks.

## 🧱 Built from scratch

The main reusable pieces include:

- multi-head and Bahdanau attention;
- sinusoidal positional encoding;
- pre-norm Transformer encoder and decoder blocks;
- beam search with length, repetition and no-repeat n-gram penalties;
- a trainer with gradient accumulation, mixed precision, gradient clipping,
  checkpoints and early stopping;
- learning-rate warmup schedulers;
- helpers for metrics, losses, visualizations and model checkpoints.

These implementations are meant for learning and experimentation. PyTorch still
handles tensors, automatic differentiation and low-level operations, while the
model architecture and training logic remain visible and easy to inspect.

## 📊 Results

The values below come from saved notebook outputs.

| Chapter | Model | Dataset | Result |
| --- | --- | --- | --- |
| 04 | MLP | MNIST | accuracy 0.9810 |
| 05 | MLP | Fashion-MNIST | accuracy 0.8795, 235,146 parameters |
| 05 | CNN | Fashion-MNIST | accuracy 0.9040, 14,274 parameters |
| 07 | LSTM | IMDb | accuracy 0.8891, F1 0.8909 |
| 09 | GRU + Bahdanau attention | OPUS-100 en–ru | validation loss 4.123 |
| 10 | Transformer | Gazeta | ROUGE-1 0.1249, BLEU 0.1082 |

The chapter 05 comparison is a good example of what this project is about: the
CNN performs better than the MLP on the same data while using roughly 16 times
fewer parameters.

The summarizer is still an experiment in progress. Its current checkpoint was
tuned with Optuna and trained with beam-search decoding, but there is still room
to improve the context window and generation strategy.

## 🚀 Getting started

You will need Python 3.13+, [uv](https://docs.astral.sh/uv/), and optionally
[Task](https://taskfile.dev/).

```sh
git clone https://github.com/NKTKLN/pytorch-dl-lab.git
cd pytorch-dl-lab
task init
uv run jupyter lab
```

If you do not use Task, replace `task init` with:

```sh
uv sync --all-groups
uv run pre-commit install --install-hooks
```

Start with `notebooks/01_tensor_practice.ipynb` and follow the chapter numbers.
Datasets from torchvision and Hugging Face are downloaded automatically on the
first run.

## 🏋️ Running training from the terminal

Three chapters also have command-line entry points:

```sh
task train:linear  # Linear regression
task train:mlp     # MLP on MNIST
task train:cnn     # CNN on Fashion-MNIST
```

Their settings live in `configs/`. For example, the CNN can load an existing
model instead of training again:

```sh
uv run python -m dl_roadmap.chapters.fashion_cnn.train --skip-training
```

Long-running jobs can optionally send a Telegram notification when training
finishes or fails. Copy `.env.example` to `.env` and provide
`TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` to enable it.

## 🛠️ Useful commands

| Command | What it does |
| --- | --- |
| `task init` | Installs dependencies and Git hooks |
| `task sync` | Synchronizes dependencies |
| `task sync-frozen` | Installs the versions pinned in `uv.lock` |
| `task fmt` | Formats code and applies safe Ruff fixes |
| `task lint` | Runs Ruff, the formatting check and mypy |
| `task audit` | Checks dependencies for known vulnerabilities |
| `task precommit-run` | Runs all pre-commit hooks |

Run `task --list` to see every available command.

## 📁 Project structure

```text
.
├── notebooks/      # Ten learning chapters and a notebook template
├── experiments/    # Hyperparameter-search experiments
├── configs/        # YAML configuration for CLI training
├── src/dl_roadmap/ # Models, layers, training engine and utilities
├── data/            # Downloaded and example datasets
└── reports/figures/ # Generated plots
```

## 📜 License

This project is available under the [MIT License](./LICENSE.md).
