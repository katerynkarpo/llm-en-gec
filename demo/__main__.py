"""Command-line launcher for the FastAPI demo."""

from __future__ import annotations

import argparse
import os

import uvicorn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the interactive English GEC demo.")
    parser.add_argument("--config", default="configs/demo.yaml", help="Demo YAML config.")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host.")
    parser.add_argument("--port", default=8080, type=int, help="Bind port.")
    parser.add_argument("--reload", action="store_true", help="Reload after code changes.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ["GEC_DEMO_CONFIG"] = args.config
    uvicorn.run(
        "demo.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
