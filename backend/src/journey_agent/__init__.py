"""Nubazo Journey Agent — a local, page-scoped customer assistant."""

__version__ = "0.1.0"


def main() -> None:
    """Console entry point: `uv run journey-agent`."""
    from journey_agent.main import run

    run()
