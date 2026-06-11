"""Entry point: python -m spatial_agent.launch_managers.gpu_server_manager"""

from spatial_agent.launch_managers.gpu_server_manager.cli import GPUServerManagerCLI


def main():
    cli = GPUServerManagerCLI()
    cli.run()


if __name__ == "__main__":
    main()
