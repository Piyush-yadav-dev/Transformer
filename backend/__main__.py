"""Launch the local shell with python -m transformer.web."""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Transformer local application interface (Phase 15)."
    )
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    parser.add_argument(
        "--development-origin", choices=("http://127.0.0.1:5173", "http://localhost:5173")
    )
    args = parser.parse_args()
    if not 1024 <= args.port <= 65535:
        parser.error("--port must be between 1024 and 65535.")
    try:
        import uvicorn

        from transformer.web.application import create_app
    except ImportError:
        parser.error(
            "Install the optional web dependencies: uv sync --locked --extra web --no-editable"
        )
    uvicorn.run(
        create_app(
            project_root=args.project_root,
            device=args.device,
            development_origin=args.development_origin,
        ),
        host="127.0.0.1",
        port=args.port,
        workers=1,
        access_log=False,
    )


if __name__ == "__main__":
    main()
