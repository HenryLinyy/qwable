"""CLI entry point for Qwable Gateway."""

import uvicorn
from qwable.config import FusionConfig


def main():
    config = FusionConfig()
    uvicorn.run(
        "qwable.server:app",
        host=config.qwable_host,
        port=config.qwable_port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
