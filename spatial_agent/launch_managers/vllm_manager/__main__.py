"""Entry point: python -m spatial_agent.launch_managers.vllm_manager"""

from spatial_agent.launch_managers.vllm_manager.cli import VLLMManagerCLI


def main():
    cli = VLLMManagerCLI()
    cli.run()


if __name__ == "__main__":
    main()
