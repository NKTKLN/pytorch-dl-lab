# 🧠 Deep Learning Roadmap with PyTorch

[![Python](https://img.shields.io/badge/python-3.13+-3776AB?logo=python\&logoColor=white)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](./LICENSE.md)
[![uv](https://img.shields.io/badge/uv-managed-261230?logo=uv\&logoColor=white)](https://docs.astral.sh/uv/)
[![Ruff](https://img.shields.io/badge/linting-ruff-D7FF64?logo=ruff\&logoColor=black)](https://docs.astral.sh/ruff/)
[![Jupyter](https://img.shields.io/badge/notebooks-jupyter-F37626?logo=jupyter\&logoColor=white)](https://jupyter.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-learning-EE4C2C?logo=pytorch\&logoColor=white)](https://pytorch.org/)

**deep-learning-roadmap** is a structured learning repository for studying deep learning with PyTorch, starting from tensors and training loops and progressing toward neural networks for sequences, attention mechanisms, Transformers, and small LLM-style models.

The project is designed as a practical companion to a deep learning roadmap based on **Deep Learning with PyTorch**, the **Yandex ML Handbook**, and hands-on LLM experiments.
Each learning stage ends with a working artifact: a model, a training script, an experiment, a plot, or a mini-project.

## 📦 Dependencies

* [Python 3.13+](https://www.python.org/downloads/)
* [uv](https://docs.astral.sh/uv/getting-started/installation/)
* [PyTorch](https://pytorch.org/)
* [Task](https://taskfile.dev/)
* [Jupyter](https://jupyter.org/)
* [Ruff](https://docs.astral.sh/ruff/)

## 🧭 Learning Goals

This repository covers the full practical path from basic PyTorch to modern deep learning models:

* PyTorch tensors, devices, shapes, and broadcasting
* Autograd, loss functions, optimizers, and training loops
* `nn.Module`, datasets, dataloaders, and model checkpoints
* MLPs and CNNs for supervised learning
* Experiment tracking, metrics, validation, and reproducibility
* Embeddings, RNNs, LSTMs, GRUs, and sequence models
* Seq2seq models and attention mechanisms
* Self-attention and Transformer blocks
* Tiny GPT-style language models
* Fine-tuning, embeddings, and RAG experiments

## 🛠️ Installation & Usage

### 💻 Local Setup

1. Make sure you have **Python 3.13 or newer** installed.

2. Sync dependencies:

```bash
task sync
```

<!-- 3. Run the first training experiment:

```bash
task train:linear
``` -->

## 🧪 Development Commands

Format code:

```bash
task fmt
```

Run linting:

```bash
task lint
```

Run linting and tests together:

```bash
task check
```

<!-- Train the first baseline model:

```bash
task train:linear
``` -->

## 📜 License

This project is licensed under the MIT License. See the [LICENSE](./LICENSE.md) file for details.
