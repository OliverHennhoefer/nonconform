"""Execute the documented online-FDR example against installed distributions."""

import argparse
from importlib.metadata import version
from pathlib import Path

import numpy as np


def main() -> None:
    """Check the installed API using the actual documentation example."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("document", type=Path)
    args = parser.parse_args()

    installed_version = version("online-fdr")

    document = args.document.read_text(encoding="utf-8")
    _, heading, section = document.partition("### Online FDR\n")
    if not heading:
        raise RuntimeError("Online FDR documentation section is missing")
    section = section.split("\n## ", 1)[0].split("\n### ", 1)[0]
    _, fence, example = section.partition("```python\n")
    if not fence or "```" not in example:
        raise RuntimeError("Online FDR Python example is missing")
    code = example.split("```", 1)[0]
    namespace: dict[str, object] = {"__name__": "__main__"}
    exec(compile(code, str(args.document), "exec"), namespace)
    np.testing.assert_array_equal(np.flatnonzero(namespace["rejections"]), [30, 70])
    print(f"online-fdr {installed_version}: documented example passed")


if __name__ == "__main__":
    main()
