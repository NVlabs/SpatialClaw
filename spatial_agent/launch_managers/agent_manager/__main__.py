"""Entry point for: python -m spatial_agent.launch_managers.agent_manager"""

from spatial_agent.launch_managers.agent_manager.cli import AgentManagerCLI


def main():
    cli = AgentManagerCLI()
    cli.run()


if __name__ == "__main__":
    main()
