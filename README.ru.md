# PyTorch DL Lab 🧪

[![Python](https://img.shields.io/badge/Python-3.13+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.11+-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-notebooks-F37626?logo=jupyter&logoColor=white)](https://jupyter.org/)
[![uv](https://img.shields.io/badge/uv-managed-DE5FE9?logo=uv&logoColor=white)](https://docs.astral.sh/uv/)
[![Ruff](https://img.shields.io/badge/linting-ruff-D7FF64?logo=ruff&logoColor=black)](https://docs.astral.sh/ruff/)
[![Tested with pytest](https://img.shields.io/badge/testing-pytest-0A9EDC?logo=pytest&logoColor=white)](https://docs.pytest.org/)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-FAB040?logo=pre-commit&logoColor=black)](https://pre-commit.com/)
[![Conventional Commits](https://img.shields.io/badge/Conventional%20Commits-1.0.0-FE5196?logo=conventionalcommits&logoColor=white)](https://www.conventionalcommits.org/)
[![CI](https://github.com/NKTKLN/pytorch-dl-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/NKTKLN/pytorch-dl-lab/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](./LICENSE.md)

[English](./README.md) · **Русский**

Лаборатория по глубокому обучению на PyTorch: от операций с тензорами
до законченных нейросетей, где ключевые части написаны руками, а не взяты
у фреймворка.

Репозиторий устроен как десять глав в Jupyter-ноутбуках. Начинается с цикла
обучения линейной регрессии, дальше MLP, CNN и рекуррентные сети, в конце
машинный перевод и суммаризация русских новостей.

Дальняя цель — собрать изученное вместе и обучить небольшую языковую модель
целиком: токенизация, блоки Transformer, цикл обучения, генерация.

## 📚 Что есть в репозитории

- основы работы с тензорами и подготовка данных;
- линейная регрессия с собственным циклом обучения;
- MLP для MNIST и Fashion-MNIST;
- CNN и её прямое сравнение с полносвязной сетью;
- модели на основе RNN, LSTM и GRU;
- seq2seq с teacher forcing и Bahdanau attention;
- Transformer encoder-decoder для абстрактивной суммаризации;
- эксперименты с Optuna для настройки суммаризатора.

Поздние главы переиспользуют код из `src/dl_roadmap/`, поэтому внутри есть не
только отдельные ноутбуки, но и небольшой общий движок обучения.

## 🧱 Написано с нуля

Основные переиспользуемые компоненты:

- multi-head attention и Bahdanau attention;
- синусоидальное позиционное кодирование;
- pre-norm блоки энкодера и декодера Transformer;
- beam search со штрафами за длину, повторы и повторяющиеся n-граммы;
- тренировочный движок: накопление градиента, mixed precision, обрезка
  градиентов, чекпоинты, early stopping с возвратом лучших весов; обучение
  отмеряется эпохами, шагами оптимизатора или бюджетом токенов;
- расписания learning rate, которые шагают на каждом шаге оптимизатора или
  на каждом интервале, с линейным прогревом перед ними;
- вспомогательные функции для метрик, лоссов, графиков и сохранения моделей.

Механизм внимания собран руками: четыре линейные проекции, маска и softmax.
`nn.Transformer` и `nn.MultiheadAttention` нигде в `src` не используются,
тренировочный цикл тоже написан здесь, а не взят у фреймворка.

В рекуррентных главах `nn.RNN`, `nn.LSTM` и `nn.GRU` взяты готовыми намеренно:
там предмет — устройство seq2seq, а не внутренности ячейки.

Тензоры, автодифференцирование и низкоуровневые операции остаются за PyTorch.
Переписывать их смысла нет: пониманию того, что происходит уровнем выше, это
ничего не добавляет.

## 📊 Результаты

Значения ниже взяты из сохранённых выводов ноутбуков.

| Глава | Модель | Датасет | Результат |
| --- | --- | --- | --- |
| 04 | MLP | MNIST | accuracy 0.9810 |
| 05 | MLP | Fashion-MNIST | accuracy 0.8795, 235 146 параметров |
| 05 | CNN | Fashion-MNIST | accuracy 0.9040, 14 274 параметра |
| 07 | LSTM | IMDb | accuracy 0.8891, F1 0.8909 |
| 09 | GRU + Bahdanau attention | OPUS-100 en–ru | validation loss 4.123 |
| 10 | Transformer | Gazeta | ROUGE-1 0.1249, BLEU 0.1082 |

Сравнение из пятой главы хорошо передаёт идею проекта: CNN показывает более
высокую точность на тех же данных и при этом использует примерно в 16 раз меньше
параметров, чем MLP.

Суммаризатор — незакрытый эксперимент. Текущий вариант подобран через Optuna
и декодируется beam search, но окно контекста и стратегию генерации ещё есть
куда двигать.

## 🚀 Быстрый старт

Понадобятся Python 3.13+, [uv](https://docs.astral.sh/uv/) и, по желанию,
[Task](https://taskfile.dev/).

```sh
git clone https://github.com/NKTKLN/pytorch-dl-lab.git
cd pytorch-dl-lab
task init
uv run jupyter lab
```

Если вы не используете Task, замените `task init` двумя командами:

```sh
uv sync --all-groups
uv run pre-commit install --install-hooks
```

Начинать лучше с `notebooks/01_tensor_practice.ipynb`, а дальше идти по номерам
глав. Датасеты torchvision и Hugging Face скачаются автоматически при первом
запуске.

## 🏋️ Обучение из терминала

У трёх глав есть отдельные команды для запуска обучения:

```sh
task train:linear  # Линейная регрессия
task train:mlp     # MLP на MNIST
task train:cnn     # CNN на Fashion-MNIST
```

Настройки находятся в `configs/`. Например, CNN можно запустить с уже
сохранённой моделью, не обучая её заново:

```sh
uv run python -m dl_roadmap.chapters.fashion_cnn.train --skip-training
```

Долгие запуски могут присылать уведомления в Telegram после завершения или
ошибки. Чтобы включить их, скопируйте `.env.example` в `.env` и укажите
`TELEGRAM_BOT_TOKEN` и `TELEGRAM_CHAT_ID`.

## 🧪 Тесты

У движка обучения есть свой набор тестов в `tests/trainer/`. Он работает на CPU
примерно за десять секунд:

```sh
task test
```

Кроме обычных модульных проверок, он фиксирует то, в чём цикл обучения
ошибается незаметно: градиенты, накопленные по нескольким микробатчам,
совпадают с градиентами одного большого батча, и для усреднённого лосса,
и для суммы по токенам; продолжение из сохранённого состояния повторяет
непрерывный запуск один в один; early stopping возвращает лучшие веса, даже
если колбэк упал с исключением.

GitHub Actions запускает `task ci-frozen` (линтер, проверку типов, тесты и
сборку пакета на зависимостях из `uv.lock`) на каждый push и pull request
в `develop` и `main`.

`task test-cov` проверяет порог покрытия 90% из `pyproject.toml`, который
считается по всему пакету `dl_roadmap`. Тесты пока есть только у движка,
поэтому порог не проходит, и CI запускает `task test`.

## 🛠️ Полезные команды

| Команда | Что делает |
| --- | --- |
| `task init` | Устанавливает зависимости и Git-хуки |
| `task sync` | Синхронизирует зависимости |
| `task sync-frozen` | Устанавливает версии из `uv.lock` |
| `task fmt` | Форматирует код и применяет безопасные исправления Ruff |
| `task lint` | Запускает Ruff, проверку форматирования и mypy |
| `task test` | Запускает тесты |
| `task test-cov` | Запускает тесты с отчётом о покрытии |
| `task ci` | Запускает линтер, тесты и сборку пакета, как в CI |
| `task audit` | Проверяет зависимости на известные уязвимости |
| `task precommit-run` | Запускает все pre-commit-хуки |

Полный список доступен по команде `task --list`.

## 📁 Структура проекта

```text
.
├── notebooks/      # Десять учебных глав и шаблон ноутбука
├── experiments/    # Эксперименты с подбором гиперпараметров
├── configs/        # YAML-конфигурация для обучения из терминала
├── src/dl_roadmap/ # Модели, слои, движок обучения и утилиты
├── tests/          # Тесты движка обучения
├── data/            # Загруженные и демонстрационные датасеты
└── reports/figures/ # Сгенерированные графики
```

## 📜 Лицензия

Проект распространяется по лицензии [MIT](./LICENSE.md).
