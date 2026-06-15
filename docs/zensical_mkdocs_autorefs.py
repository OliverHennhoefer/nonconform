"""Bridge Zensical's mkdocstrings shim with current mkdocs-autorefs.

Zensical 0.0.45 registers its compatibility processor as ``autorefs``, while
mkdocstrings 0.30 and mkdocs-autorefs 1.4 expect ``mkdocs-autorefs``.
"""

from __future__ import annotations

from typing import Any

from markdown.extensions import Extension
from mkdocs_autorefs import AutorefsExtension


def makeExtension(**kwargs: Any) -> Extension:  # noqa: N802
    """Create the mkdocs-autorefs Markdown extension."""
    return AutorefsExtension(**kwargs)
