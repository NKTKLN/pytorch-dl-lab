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

[English](./README.md) · **Русский**

**PyTorch DL Lab** — репозиторий по глубокому обучению, в котором стек трансформера и движок обучения написаны вручную: multi-head attention, pre-norm блоки энкодера и декодера, beam search со штрафами за повторы и n-граммы, а также `Trainer` с накоплением градиента, смешанной точностью и прогревом learning rate. На этом держатся десять глав — от цикла линейной регрессии до абстрактивной суммаризации русских новостей.

## 🧱 Написано с нуля

| Компонент | Где | Что делает |
| --- | --- | --- |
| `MultiHeadAttention` | `layers/attention.py` | Scaled dot-product attention с разбиением и сборкой голов, каузальной маской, аддитивными или булевыми масками размерности 2/3/4, масками паддинга ключей и дропаутом на весах внимания |
| `BahdanauAttention` | `layers/attention.py` | Аддитивное внимание, сопоставляющее состояние декодера с состояниями энкодера; используется переводчиком главы 09 |
| `TransformerEncoderLayer`, `TransformerDecoderLayer` | `layers/transformer.py` | Pre-norm блоки: self-attention, cross-attention в декодере и GELU-бутылочное горлышко feed-forward со значением по умолчанию `4 * model_dim` |
| `TransformerEncoder`, `TransformerDecoder` | `layers/transformer.py` | Стеки из N блоков с завершающим `LayerNorm` |
| `PositionalEncoding` | `layers/positional.py` | Синусоидальная таблица позиций, предпосчитанная до `max_length` и зарегистрированная как буфер; при выходе последовательности за границу бросает ошибку, а не обрезает |
| `Trainer` | `engine/trainer.py` | Цикл обучения с накоплением градиента, autocast, клиппингом градиента, чекпоинтами, ранней остановкой, планировщиком LR и метриками внутри цикла |
| `beam_search` | `engine/beam_search.py` | Лучевое декодирование со штрафом за длину, минимальной длиной, штрафом за повторы и запретом повторяющихся n-грамм |
| `WarmupScheduler`, `EpochWarmupScheduler` | `engine/schedulers.py` | Линейный прогрев LR по шагам оптимизатора или по эпохам с передачей управления обёрнутому планировщику torch и восстановлением его скорости в `load_state_dict` |
| Ранняя остановка | `engine/early_stopping.py` | Пять стратегий — абсолютный порог, val loss, отслеживаемая метрика, gap обобщения и порог по gap — плюс комбинатор, срабатывающий по любой из них или по всем сразу |
| Учёт лосса и метрик | `engine/loss_tracker.py`, `engine/metric.py` | Накопление среднего и потокенного лосса, точность по токенам и ROUGE по тексту из teacher forcing или из beam search |

Две детали сделаны как в оригинальных статьях, а не как удобнее: выходная проекция связана с входным эмбеддингом, а сами эмбеддинги инициализируются с `std = model_dim ** -0.5` и умножаются на `sqrt(model_dim)` до добавления позиционного сигнала. Несвязанная проекция берётся из того же распределения, поэтому переключение `use_weight_tying` меняет только разделение весов и ничего больше.

Смешанная точность сама определяет свой dtype. При `amp="auto"` autocast выключен вне CUDA, выбирает bfloat16 там, где устройство поддерживает его аппаратно, и откатывается к float16 в остальных случаях; `amp="bf16"` отказывается работать на CUDA-устройстве, которое эмулировало бы bfloat16 программно, вместо того чтобы молча стать медленнее заменяемого fp32-пути. `GradScaler` включается только для float16.

ROUGE и BLEU берутся из `torchmetrics`; здесь написана обвязка вокруг них — обновление по батчам, хук декодирования и ограничение на число валидационных батчей, генерируемых за эпоху.

## 📊 Результаты

Каждое число ниже напечатано ноутбуком соответствующей главы.

| Глава | Модель | Данные | Результат |
| --- | --- | --- | --- |
| 04 | MLP | MNIST | accuracy 0.9810 |
| 05 | MLP | Fashion-MNIST | accuracy 0.8795, 235 146 параметров |
| 05 | CNN | Fashion-MNIST | accuracy 0.9040, 14 274 параметра |
| 07 | LSTM | `stanfordnlp/imdb` | accuracy 0.8891, F1 0.8909 |
| 09 | GRU seq2seq + внимание Богданова | `Helsinki-NLP/opus-100` en–ru | val loss 4.123, ранняя остановка на эпохе 30/50 |
| 10 | Трансформер энкодер–декодер | `IlyaGusev/gazeta` | ROUGE-1 F1 0.1249, ROUGE-2 F1 0.0490, ROUGE-L F1 0.1216, BLEU 0.1082 |

Главу 05 стоит перечитать дважды ради сравнения: свёрточная модель обходит полносвязную на тех же данных, имея в 16 раз меньше параметров.

## 🔬 Глава 10 ещё не закрыта

Числа суммаризатора выше — это срез текущей серии запусков, а не итог. Нынешняя конфигурация — `model_dim` 256, 4 головы, 6 слоёв энкодера и 3 декодера, FFN-отношение 2, словарь SentencePiece unigram на 16 000 фрагментов, AdamW с lr 3.17e-4, dropout 0.163 и weight decay 3.2e-4, одна эпоха прогрева и затем `ReduceLROnPlateau` — получена поэтапным поиском Optuna в `experiments/10_transformer_summarizer_tuning.ipynb`, где сначала перебирается сетка архитектур, затем лучшие кандидаты перезапускаются на разных сидах шума, а по выжившим прогоняется стадия TPE.

Запуск остановился ранней остановкой на эпохе 44 из 100 с val loss 3.982 — по сочетанию терпения к среднему ROUGE и порога по gap обобщения. Декодирование идёт лучом шириной 5 с минимальной длиной 35 токенов, штрафом за повторы 1.2 и запретом повторяющихся 3-грамм — ради этого штрафы в `beam_search` и писались. Открытые вопросы — окно контекста исходного текста и стратегия декодирования.

## 📦 Зависимости

* [Python 3.13+](https://www.python.org/downloads/)
* [uv](https://docs.astral.sh/uv/getting-started/installation/)
* [Task](https://taskfile.dev/) — для команд ниже; каждая из них тонкая обёртка над вызовом `uv run` в `Taskfile.yml`

Всё остальное uv ставит из `uv.lock`, отдельная установка не нужна: torch 2.11 или новее (в локе сейчас закреплён 2.12.1), torchvision, torchaudio, torchmetrics, torcheval, Optuna, `datasets`, sentencepiece, виджеты JupyterLab, pandas, matplotlib, seaborn и loguru.

Установить группы зависимостей и git-хуки:

```sh
task init
```

Только зависимости:

```sh
task sync
```

Только закреплённые версии, не трогая lock-файл:

```sh
task sync-frozen
```

## 🚀 Запуск

У трёх глав есть точки входа для обучения из командной строки. Запускать их нужно из корня репозитория — пути к конфигам внутри заданы относительно него.

Глава 03, линейная регрессия на синтетических данных:

```sh
task train:linear
```

Глава 04, MLP на MNIST:

```sh
task train:mlp
```

Глава 05, CNN на Fashion-MNIST:

```sh
task train:cnn
```

Точка входа CNN — команда Typer: `--config-file` указывает на другой YAML, а `--skip-training` загружает модель из `training.model_path` вместо обучения:

```sh
uv run python -m dl_roadmap.chapters.fashion_cnn.train --skip-training
```

Всё остальное живёт в ноутбуках, потому что те главы про чтение промежуточных тензоров и карт внимания, а не про одно число в конце:

```sh
uv run jupyter lab
```

Начинать стоит с `notebooks/01_tensor_practice.ipynb` и идти по порядку — поздние главы импортируют слои и движок, которые вводят ранние. Датасеты скачиваются сами при первом запуске: torchvision в `data/raw`, а корпуса Hugging Face глав 07, 09 и 10 — через `datasets`.

## 🔧 Конфигурация

Три CLI-главы читают YAML-файл из `configs/`. В именах файлов стоит номер эксперимента, а не ноутбука: `01_linear_regression.yaml` относится к ноутбуку 03, `02_mlp_mnist.yaml` — к ноутбуку 04, `03_cnn_fashion_mnist.yaml` — к ноутбуку 05. Значения по умолчанию ниже взяты из `02_mlp_mnist.yaml`.

| Ключ | По умолчанию | Что это |
| --- | --- | --- |
| `experiment_name` | `02_mlp_mnist` | Имя запуска в логах и в именах файлов графиков |
| `seed` | `42` | Сид для генераторов `random`, torch, CUDA и NumPy |
| `data.root` | `data/raw` | Куда torchvision скачивает и кэширует датасет |
| `data.batch_size` | `64` | Размер батча для обоих загрузчиков |
| `training.epochs` | `10` | Бюджет эпох |
| `training.learning_rate` | `0.001` | Начальный learning rate |
| `training.checkpoint_dir` | `checkpoints/02_mlp_mnist` | Каталог чекпоинтов; пустое значение полностью отключает их сохранение |
| `training.checkpoint_every` | `1` | Сохранять каждые N эпох |
| `training.show_progress` | `true` | Рисует ли тренер прогресс-бар |
| `visualization.figures_dir` | `reports/figures` | Куда пишутся кривые обучения и матрицы ошибок |
| `visualization.show_fig` | `true` | Показывать ли графики, а не только сохранять |
| `logging.disable_logging` | `false` | Полностью заглушает loguru — так делают ноутбуки |
| `logging.log_level` | `INFO` | Уровень логирования loguru |
| `logging.log_path` | `logs/02_mlp_mnist/train.log` | Файл лога запуска |

Два других файла устроены так же. `01_linear_regression.yaml` описывает генерацию данных вместо скачивания — `data.n_samples` 1000, `data.n_features` 2, `data.noise` 0.1 и `data.batch_size` 32 — и учится 100 эпох при lr 0.01. `03_cnn_fashion_mnist.yaml` добавляет `training.model_path` (`model/cnn_fashion_mnist.pt`), файл, который читает `--skip-training`.

Запуски здесь достаточно долгие, чтобы от них отойти — глава 10 училась 44 эпохи, — поэтому `dl_roadmap.utils.notify` умеет прислать сообщение в Telegram, когда обучение завершилось или упало. Это необязательно и молчит, пока не настроено. Чтобы включить, скопируйте `.env.example` в `.env`.

| Переменная | По умолчанию | Что это |
| --- | --- | --- |
| `TELEGRAM_BOT_TOKEN` | не задана | Токен бота от [@BotFather](https://t.me/BotFather) |
| `TELEGRAM_CHAT_ID` | не задана | Чат, куда приходят уведомления |
| `TELEGRAM_PROXY` | не задана | Необязательный прокси `http://` или `socks5://` для `api.telegram.org`; для `socks5` нужен `pysocks` |

## 📁 Структура исходников

```
src/dl_roadmap/
├── layers/         # multi-head и Bahdanau attention, синусоидальные позиционные кодировки, pre-norm блоки трансформера
├── engine/         # Trainer, beam search, планировщики прогрева, ранняя остановка, трекеры лосса и метрик, предсказатель классов
├── chapters/       # по пакету на главу: её модель и, для трёх из них, точка входа обучения
├── data/           # синтетические данные для регрессии, словарь на уровне слов, разбиение на train/test
├── metrics/        # метрики бинарной и многоклассовой классификации
├── visualization/  # матрицы внимания, матрицы ошибок, история обучения, палитры
└── utils/          # загрузка YAML-конфигов, настройка loguru, io чекпоинтов, сиды, уведомления в Telegram
```

В `notebooks/` лежат десять глав по порядку и шаблон модели, в `experiments/` — поиски гиперпараметров, в `configs/` — YAML для CLI-глав, а в `reports/figures/` — построенные ими графики.

## 🧰 Разработка

Форматирование и автоисправления:

```sh
task fmt
```

Ruff, проверка форматирования и mypy разом:

```sh
task lint
```

Всё, что запускают pre-commit хуки, по всем файлам:

```sh
task precommit-run
```

Аудит безопасности зависимостей и проверка на неиспользуемые:

```sh
task audit
```

```sh
task unused-libs
```

Коммит через commitizen, который следит за Conventional Commits и ведёт версионирование из `pyproject.toml`:

```sh
task cz-commit
```

Ruff работает с 16 группами правил, включая `S` для безопасности и `D` для docstring в стиле Google, по ноутбукам наравне с `src`, а mypy настроен по `src` с околострогими флагами (`disallow_untyped_defs`, `disallow_untyped_calls`, `warn_return_any`, `warn_unreachable`, `disallow_any_unimported` и прочими). Ruff и `ruff-format` запускаются на каждом коммите через pre-commit вместе с gitleaks и проверкой согласованности `uv.lock`; commitizen проверяет сообщение коммита и ветку при push.

Две вещи, которые стоит знать, прежде чем доверять этому контуру. **Тестов нет** — pytest и pytest-cov установлены, покрытие настроено с `fail_under = 90`, но ни одного теста не написано, так что репозиторий пока гарантирует статику, а не динамику. И `task check` вызывает задачи `test-cov` и `build`, которых в `Taskfile.yml` нет, поэтому обрывается на середине; пока их не добавили, запускайте `task lint`, `task audit` и `task unused-libs` по отдельности.

## 📜 Лицензия

Проект распространяется по лицензии MIT. Подробности в файле [LICENSE](./LICENSE.md).
