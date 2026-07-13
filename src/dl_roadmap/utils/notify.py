"""Telegram notification utilities.

Credentials are read from the `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID`
environment variables (a local `.env` file is loaded automatically via
python-dotenv, see `.env.example`). By default, missing credentials are
treated as "notifications are off": a warning is logged and the call is
a silent no-op, so notebooks and scripts don't crash just because Telegram
isn't configured. Pass `strict=True` to raise instead.
"""

import os

import requests
from dotenv import load_dotenv
from loguru import logger

TELEGRAM_API_URL = "https://api.telegram.org/bot{token}/sendMessage"

MODEL_TRAINED_MESSAGE_TEMPLATE = """✅ Model trained: {model_name}
{details}"""


def _get_credentials(*, strict: bool = False) -> tuple[str, str] | None:
    """Read Telegram credentials from the environment.

    Loads variables from a local `.env` file (if present) and returns the
    bot token and chat id.

    Args:
        strict: If True, raise when credentials are missing instead of
            returning None.

    Returns:
        A `(bot_token, chat_id)` tuple, or None if either variable is
        unset and `strict` is False.

    Raises:
        RuntimeError: If `strict` is True and `TELEGRAM_BOT_TOKEN` or
            `TELEGRAM_CHAT_ID` is not set in the environment or the
            `.env` file.
    """
    load_dotenv()

    bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")

    if not bot_token or not chat_id:
        message = (
            "Telegram credentials are not configured: set TELEGRAM_BOT_TOKEN "
            "and TELEGRAM_CHAT_ID in the environment or in a .env file "
            "(see .env.example)."
        )
        if strict:
            raise RuntimeError(message)

        logger.warning(f"{message} Skipping notification.")
        return None

    return bot_token, chat_id


def tg_notify(message: str, timeout: float = 10.0, *, strict: bool = False) -> None:
    """Send a text message to the configured Telegram chat.

    If Telegram credentials are not configured, this logs a warning and
    returns without sending anything, unless `strict` is set.

    Args:
        message: Text of the message to send.
        timeout: Request timeout in seconds.
        strict: If True, raise instead of silently skipping when
            credentials are missing.

    Raises:
        RuntimeError: If `strict` is True and Telegram credentials are
            not configured.
        requests.HTTPError: If the Telegram API responds with an error status.
        requests.RequestException: If the request fails (e.g. network error).
    """
    credentials = _get_credentials(strict=strict)
    if credentials is None:
        return

    bot_token, chat_id = credentials

    response = requests.post(
        TELEGRAM_API_URL.format(token=bot_token),
        json={"chat_id": chat_id, "text": message},
        timeout=timeout,
    )
    response.raise_for_status()


def notify_model_trained(
    model_name: str,
    metrics: dict[str, float] | None = None,
    *,
    strict: bool = False,
) -> None:
    """Send a Telegram notification that a model has finished training.

    If Telegram credentials are not configured, this logs a warning and
    returns without sending anything, unless `strict` is set.

    Args:
        model_name: Human-readable name of the trained model.
        metrics: Optional final metrics to include in the message,
            e.g. `{"val_loss": 1.23, "bleu": 0.31}`.
        strict: If True, raise instead of silently skipping when
            credentials are missing.

    Raises:
        RuntimeError: If `strict` is True and Telegram credentials are
            not configured.
        requests.HTTPError: If the Telegram API responds with an error status.
        requests.RequestException: If the request fails (e.g. network error).
    """
    details = ""
    if metrics:
        details = "\n".join(f"* {name}: {value:.4f}" for name, value in metrics.items())

    message = MODEL_TRAINED_MESSAGE_TEMPLATE.format(
        model_name=model_name, details=details
    ).strip()

    tg_notify(message, strict=strict)
