"""Entry point: python -m spatial_agent.visualization_server [--port PORT] [--work-dir PATH]"""

import argparse
import os


def main():
    parser = argparse.ArgumentParser(description="Spatial Agent Results Visualization Server")
    parser.add_argument("--port", type=int, default=8501, help="Server port (default: 8501)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host (default: 0.0.0.0)")
    parser.add_argument(
        "--work-dir",
        type=str,
        default="spatial_agent/work_dir",
        help="Path to work_dir (default: spatial_agent/work_dir)",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="data",
        help="Path to benchmark data root (default: data)",
    )
    args = parser.parse_args()

    work_dir = os.path.abspath(args.work_dir)
    if not os.path.isdir(work_dir):
        print(f"Error: work_dir not found: {work_dir}")
        print("Make sure you run from the project root, or pass --work-dir explicitly.")
        return

    data_root = os.path.abspath(args.data_root)

    # Import here to avoid slow startup if args are wrong
    import uvicorn
    from .app import create_app

    app = create_app(work_dir, data_root=data_root)
    print(f"Scanning {work_dir} ...")
    print(f"Starting server at http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
