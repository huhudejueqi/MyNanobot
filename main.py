"""Simple entry point: python main.py"""
import sys
from pathlib import Path

# Make sure the project root is on sys.path so local imports work
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from nanobot.config.loader import load_config, resolve_config_env_vars
from nanobot.cli.commands import _run_gateway


def main():
    config = resolve_config_env_vars(load_config())
    _run_gateway(config)


if __name__ == "__main__":
    main()
