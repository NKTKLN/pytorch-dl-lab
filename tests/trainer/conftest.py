"""Fixtures every trainer test uses."""

import pytest
from loguru import logger


@pytest.fixture(autouse=True)
def _quiet_logs() -> None:
    """Keep loguru's debug output out of the test report."""
    logger.remove()
