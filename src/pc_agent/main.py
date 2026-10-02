"""Entry point for the `pc-agent` command."""

import sys

from pc_agent.cli.interface import main as cli_main


def main() -> None:
    sys.exit(cli_main())


if __name__ == "__main__":
    main()
