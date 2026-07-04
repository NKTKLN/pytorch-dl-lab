"""Training entry point for the linear regression example."""

from dl_roadmap.utils import LoadConfig, LoggerConfig, setup_logger


def main() -> None:
    """Load config, configure logging, and run training."""
    config_loader = LoadConfig("configs/01_linear_regression.yaml")
    config = config_loader.config

    setup_logger(LoggerConfig(**config.get("logging", {})))


if __name__ == "__main__":
    main()
