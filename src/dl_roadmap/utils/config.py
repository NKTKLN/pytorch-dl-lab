"""YAML configuration loading module.

This module provides the LoadConfig class for lazily loading and validating
experiment configuration files stored in YAML format.
"""

from pathlib import Path
from typing import Any

import yaml


class LoadConfig:
    """Lazily load and cache a YAML configuration file.

    The file is read from disk on first access to the `config` property
    (or an explicit call to `load`) and cached for subsequent access.
    """

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

    def load(self) -> None:
        """Read the YAML config file from disk and cache its content.

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

    @property
    def config(self) -> dict[str, Any]:
        """Return the config content, loading it from disk if needed."""
        if self._config is None:
            self.load()

        return self._config
