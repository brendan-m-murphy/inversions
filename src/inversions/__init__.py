# Package entrypoint for the `inversions` console script.
# The pyproject entrypoint `inversions = "inversions:main"` will call this.
from . import cli

__all__ = ["cli"]


def main(argv=None):
    """Entry point used by the console script `inversions`."""
    return cli.main(argv)
