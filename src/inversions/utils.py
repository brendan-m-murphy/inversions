"""Utility functions.

Some of these functions replace ipython magic.
"""

from pathlib import Path


def ls(path: str | Path) -> list[str]:
    """Recreate ipython !ls magic.

    Returns list of file names.
    """
    path = Path(path)
    return [p.name for p in path.iterdir()]


def glob_ls(path: str | Path, glob: str) -> list[str]:
    """Recreate ipython !ls magic with glob.

    Returns list of paths (not just file names!).
    """
    path = Path(path)
    return list(map(str, path.glob(glob)))
