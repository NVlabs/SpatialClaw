"""Entry point: python -m spatial_agent.gpu_dashboard --port 8502 --db <path>"""

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Spatial Agent GPU Dashboard")
    parser.add_argument("--port", type=int, default=8502)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument(
        "--db",
        type=str,
        default=None,
        help="SQLite DB path (default: <project_root>/spatial_agent/logs/gpu_dashboard.db)",
    )
    parser.add_argument(
        "--project-root",
        type=str,
        default=None,
        help="Project root (default: inferred from this file's location)",
    )
    args = parser.parse_args()

    project_root = Path(args.project_root) if args.project_root else \
        Path(__file__).resolve().parent.parent.parent
    db_path = args.db or str(project_root / "spatial_agent" / "logs" / "gpu_dashboard.db")

    import uvicorn
    from .app import create_app

    app = create_app(db_path=db_path, project_root=project_root)
    print(f"GPU dashboard serving at http://{args.host}:{args.port}")
    print(f"Reading from: {db_path}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
