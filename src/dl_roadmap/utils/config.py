"""YAML configuration loading module."""

from pathlib import Path
from typing import Any

import yaml


class LoadConfig:
    """Lazily load and cache a YAML configuration file."""

    def __init__(self, filepath: str) -> None:
        """Initialize the loader for the given config file path.

        Args:
            filepath: Path to the YAML config file to load.

        Raises:
            FileNotFoundError: If no file exists at `filepath`.
        """
        self.filepath = Path(filepath)

        if not self.filepath.is_file():
            raise FileNotFoundError(f"Config file not found: {self.filepath}")

        self._config: dict[str, Any] | None = None

    def load(self) -> dict[str, Any]:
        """Read the YAML config file from disk and cache its content.

        Returns:
            dict[str, Any]: The parsed config content.

        Raises:
            yaml.YAMLError: If the file content is not valid YAML.
            ValueError: If the parsed content is not a mapping (dict).
        """
        with self.filepath.open(mode="r", encoding="utf8") as file:
            config = yaml.safe_load(file)

        if not isinstance(config, dict):
            raise ValueError(
                "Config file must contain a YAML mapping, got "
                f"{type(config).__name__}: {self.filepath}"
            )

        self._config = config

        return config

    @property
    def config(self) -> dict[str, Any]:
        """Return the config content, loading it from disk if needed."""
        if self._config is None:
            return self.load()

        return self._config
