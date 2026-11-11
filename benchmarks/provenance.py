"""Where and from what code a shard or an estimator was produced.

With one machine the answer was always the same. With two, a paired comparison whose arms
ran on different nodes is only a comparison if both nodes ran the same code, so every
generated shard and every trained estimator records the node and the commit.
"""

import socket
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, capture_output=True, text=True, check=True, timeout=30
    ).stdout.strip()


def stamp() -> dict:
    """``{"host", "commit"}``; the commit carries ``+dirty`` if tracked files are modified.

    ``commit`` is ``"unknown"`` outside a git checkout rather than an error: a run should
    not die at its last step for want of a label.
    """
    try:
        commit = _git("rev-parse", "--short=12", "HEAD")
        if _git("status", "--porcelain", "--untracked-files=no"):
            commit += "+dirty"
    except (OSError, subprocess.SubprocessError):
        commit = "unknown"
    return {"host": socket.gethostname(), "commit": commit}
