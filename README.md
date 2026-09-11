# 🧪 PyTorch DL Lab

[![Python](https://img.shields.io/badge/Python-3.13+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.11+-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-notebooks-F37626?logo=jupyter&logoColor=white)](https://jupyter.org/)
[![uv](https://img.shields.io/badge/uv-managed-DE5FE9?logo=uv&logoColor=white)](https://docs.astral.sh/uv/)
[![Ruff](https://img.shields.io/badge/linting-ruff-D7FF64?logo=ruff&logoColor=black)](https://docs.astral.sh/ruff/)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-FAB040?logo=pre-commit&logoColor=black)](https://pre-commit.com/)
[![Task](https://img.shields.io/badge/Task-29BEB0?logo=task&logoColor=white)](https://taskfile.dev/)
[![Conventional Commits](https://img.shields.io/badge/Conventional%20Commits-1.0.0-FE5196?logo=conventionalcommits&logoColor=white)](https://www.conventionalcommits.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](./LICENSE.md)

**PyTorch DL Lab** is a deep learning codebase whose Transformer stack and training engine are written by hand — multi-head attention, pre-norm encoder and decoder blocks, beam search with repetition and n-gram penalties, and a `Trainer` with gradient accumulation, mixed precision and learning-rate warmup — driving ten chapters that run from a linear regression loop up to an abstractive Russian news summarizer.

## 🧱 Written from scratch

| Component | Where | What it does |
| --- | --- | --- |
| `MultiHeadAttention` | `layers/attention.py` | Scaled dot-product attention with head split/combine, causal masking, 2/3/4-dim additive or boolean attention masks, key padding masks, and dropout on the attention weights |
| `BahdanauAttention` | `layers/attention.py` | Additive attention scoring the decoder state against encoder states, used by the chapter 09 translator |
| `TransformerEncoderLayer`, `TransformerDecoderLayer` | `layers/transformer.py` | Pre-norm blocks: self-attention, cross-attention in the decoder, and a GELU feed-forward bottleneck defaulting to `4 * model_dim` |
| `TransformerEncoder`, `TransformerDecoder` | `layers/transformer.py` | N-block stacks with a final `LayerNorm` |
| `PositionalEncoding` | `layers/positional.py` | Sinusoidal position table, precomputed to `max_length` and registered as a buffer; raises rather than truncating when a sequence overruns it |
| `Trainer` | `engine/trainer.py` | Training loop with gradient accumulation, autocast, gradient clipping, checkpointing, early stopping, LR scheduling and in-loop metrics |
| `beam_search` | `engine/beam_search.py` | Beam decoding with length penalty, minimum length, repetition penalty and no-repeat-n-gram blocking |
| `WarmupScheduler`, `EpochWarmupScheduler` | `engine/schedulers.py` | Linear LR warmup by optimizer step or by epoch, handing over to a wrapped torch scheduler and restoring its rate on `load_state_dict` |
| Early stopping | `engine/early_stopping.py` | Five strategies — absolute threshold, validation loss, tracked metric, generalization gap and gap threshold — plus a combiner that fires on any or all of them |
| Loss and metric tracking | `engine/loss_tracker.py`, `engine/metric.py` | Mean and per-token loss accumulation, token accuracy, and ROUGE over teacher-forced or beam-generated text |

Two details follow the original papers rather than the convenient default: the output projection is tied to the input embedding, and embeddings are initialised at `std = model_dim ** -0.5` and scaled by `sqrt(model_dim)` before the positional signal is added. An untied projection is drawn from the same distribution, so flipping `use_weight_tying` changes the sharing and nothing else.

Mixed precision resolves its own dtype. On `amp="auto"` autocast stays off outside CUDA, picks bfloat16 where the device supports it in hardware, and falls back to float16 otherwise; `amp="bf16"` refuses a CUDA device that would emulate bfloat16 in software rather than silently running slower than the fp32 path it replaced. The `GradScaler` is enabled only for float16.

ROUGE and BLEU come from `torchmetrics`; what is written here is the accumulation around them — batch-by-batch updates, a decode hook, and a cap on how many validation batches get generated per epoch.

## 📊 Results

Every figure below is a value printed by that chapter's notebook.

| Chapter | Model | Data | Result |
| --- | --- | --- | --- |
| 04 | MLP | MNIST | accuracy 0.9810 |
| 05 | MLP | Fashion-MNIST | accuracy 0.8795, 235,146 parameters |
| 05 | CNN | Fashion-MNIST | accuracy 0.9040, 14,274 parameters |
| 07 | LSTM | `stanfordnlp/imdb` | accuracy 0.8891, F1 0.8909 |
| 09 | GRU seq2seq + Bahdanau attention | `Helsinki-NLP/opus-100` en–ru | val loss 4.123, early stopped at epoch 30/50 |
| 10 | Transformer encoder–decoder | `IlyaGusev/gazeta` | ROUGE-1 F1 0.1249, ROUGE-2 F1 0.0490, ROUGE-L F1 0.1216, BLEU 0.1082 |

Chapter 05 is the comparison worth reading twice: the convolutional model beats the fully connected one on the same data with 16 times fewer parameters.

## 🔬 Chapter 10 is still open

The summarizer numbers above are a checkpoint in an ongoing series of runs, not a result. The current configuration — `model_dim` 256, 4 heads, 6 encoder and 3 decoder layers, an FFN ratio of 2, a 16,000-piece SentencePiece unigram vocabulary, AdamW at lr 3.17e-4 with dropout 0.163 and weight decay 3.2e-4, one warmup epoch then `ReduceLROnPlateau` — came out of the staged Optuna search in `experiments/10_transformer_summarizer_tuning.ipynb`, which sweeps an architecture grid, re-runs the top candidates across noise seeds, and then runs a TPE stage over the survivors.

The run stopped on early stopping at epoch 44 of 100 with val loss 3.982, triggered by a combination of ROUGE-mean patience and a generalization-gap threshold. Decoding uses beam width 5 with a minimum length of 35 tokens, repetition penalty 1.2 and 3-grams blocked from repeating, which is what the penalties in `beam_search` were written for. Open work is the source context window and the decoding strategy.

## 📦 Dependencies

* [Python 3.13+](https://www.python.org/downloads/)
* [uv](https://docs.astral.sh/uv/getting-started/installation/)
* [Task](https://taskfile.dev/) — for the commands below; each one is a thin wrapper over a `uv run` invocation in `Taskfile.yml`

Everything else is resolved by uv from `uv.lock` and needs no separate install: torch 2.11 or newer (the lock currently pins 2.12.1), torchvision, torchaudio, torchmetrics, torcheval, Optuna, `datasets`, sentencepiece, JupyterLab widgets, pandas, matplotlib, seaborn and loguru.

Install the dependency groups and the git hooks:

```sh
task init
```

Dependencies only:

```sh
task sync
```

Pinned versions only, without touching the lock file:

```sh
task sync-frozen
```

## 🚀 Running

Three chapters have command-line training entry points. Run them from the repository root — the config paths inside them are relative to it.

Chapter 03, linear regression on synthetic data:

```sh
task train:linear
```

Chapter 04, MLP on MNIST:

```sh
task train:mlp
```

Chapter 05, CNN on Fashion-MNIST:

```sh
task train:cnn
```

The CNN entry point is a Typer command and takes `--config-file` to point at another YAML and `--skip-training` to load the model at `training.model_path` instead of training it:

```sh
uv run python -m dl_roadmap.chapters.fashion_cnn.train --skip-training
```

Everything else lives in notebooks, because those chapters are about reading intermediate tensors and attention maps rather than a single scalar at the end:

```sh
uv run jupyter lab
```

Start at `notebooks/01_tensor_practice.ipynb` and go in order — later chapters import the layers and engine that earlier ones introduce. The datasets download themselves on first run: torchvision into `data/raw`, and the Hugging Face corpora of chapters 07, 09 and 10 through `datasets`.

## 🔧 Configuration

The three CLI chapters read a YAML file from `configs/`. The file names carry the experiment number, not the notebook number: `01_linear_regression.yaml` drives notebook 03, `02_mlp_mnist.yaml` drives notebook 04, and `03_cnn_fashion_mnist.yaml` drives notebook 05. Defaults below are from `02_mlp_mnist.yaml`.

| Key | Default | What it is |
| --- | --- | --- |
| `experiment_name` | `02_mlp_mnist` | Names the run in logs and figure filenames |
| `seed` | `42` | Seeds the `random`, torch, CUDA and NumPy RNGs |
| `data.root` | `data/raw` | Where torchvision downloads and caches the dataset |
| `data.batch_size` | `64` | Batch size for both loaders |
| `training.epochs` | `10` | Epoch budget |
| `training.learning_rate` | `0.001` | Initial learning rate |
| `training.checkpoint_dir` | `checkpoints/02_mlp_mnist` | Checkpoint directory; an empty value disables checkpointing entirely |
| `training.checkpoint_every` | `1` | Save every N epochs |
| `training.show_progress` | `true` | Whether the trainer draws its progress bar |
| `visualization.figures_dir` | `reports/figures` | Where training curves and confusion matrices are written |
| `visualization.show_fig` | `true` | Whether figures are also displayed, not just saved |
| `logging.disable_logging` | `false` | Silences loguru completely, which is what the notebooks do |
| `logging.log_level` | `INFO` | loguru level |
| `logging.log_path` | `logs/02_mlp_mnist/train.log` | Log file for the run |

The other two files have the same shape. `01_linear_regression.yaml` describes generated data instead of a download, with `data.n_samples` 1000, `data.n_features` 2, `data.noise` 0.1 and `data.batch_size` 32, and runs for 100 epochs at lr 0.01. `03_cnn_fashion_mnist.yaml` adds `training.model_path` (`model/cnn_fashion_mnist.pt`), the file `--skip-training` reads back.

Training runs here are long enough to walk away from — chapter 10 ran for 44 epochs — so `dl_roadmap.utils.notify` can send a Telegram message when one finishes or dies. It is optional and silent when unconfigured. Copy `.env.example` to `.env` to switch it on.

| Variable | Default | What it is |
| --- | --- | --- |
| `TELEGRAM_BOT_TOKEN` | unset | Bot token from [@BotFather](https://t.me/BotFather) |
| `TELEGRAM_CHAT_ID` | unset | Chat that receives the notifications |
| `TELEGRAM_PROXY` | unset | Optional `http://` or `socks5://` proxy for `api.telegram.org`; `socks5` needs `pysocks` |

## 📁 Source layout

```
src/dl_roadmap/
├── layers/         # multi-head and Bahdanau attention, sinusoidal positional encoding, pre-norm transformer blocks
├── engine/         # Trainer, beam search, warmup schedulers, early stopping, loss and metric trackers, class predictor
├── chapters/       # one package per chapter: its model and, for three of them, a train entry point
├── data/           # synthetic regression data, word-level vocabulary, train/test splitting
├── metrics/        # binary and multiclass classification metrics
├── visualization/  # attention matrices, confusion matrices, training history, palettes
└── utils/          # YAML config loading, loguru setup, checkpoint io, seeding, Telegram notifications
```

`notebooks/` holds the ten chapters in order plus a model template, `experiments/` the hyperparameter searches, `configs/` the YAML for the CLI chapters, and `reports/figures/` the plots they produce.

## 🧰 Development

Format and auto-fix:

```sh
task fmt
```

Ruff, the format check and mypy together:

```sh
task lint
```

Everything the pre-commit hooks would run, across all files:

```sh
task precommit-run
```

Dependency security audit, and the unused-dependency check:

```sh
task audit
```

```sh
task unused-libs
```

Commit through commitizen, which enforces Conventional Commits and drives versioning from `pyproject.toml`:

```sh
task cz-commit
```

Ruff runs 16 rule groups including `S` for security and `D` for Google-style docstrings, over notebooks as well as `src`, and mypy is configured over `src` with the strict-adjacent flags (`disallow_untyped_defs`, `disallow_untyped_calls`, `warn_return_any`, `warn_unreachable`, `disallow_any_unimported` and the rest). Ruff and `ruff-format` run on every commit through pre-commit, alongside gitleaks and a `uv.lock` consistency check; commitizen validates the commit message and the branch on push.

Two things a reader should know before trusting the gate. **There is no test suite** — pytest and pytest-cov are installed and coverage is configured with `fail_under = 90`, but no tests have been written, so what this repository currently guarantees is static, not dynamic. And `task check` chains a `test-cov` and a `build` task that `Taskfile.yml` does not define, so it aborts partway; run `task lint`, `task audit` and `task unused-libs` individually until those are added.

## 📜 License

This project is licensed under the MIT License. See the [LICENSE](./LICENSE.md) file for details.
