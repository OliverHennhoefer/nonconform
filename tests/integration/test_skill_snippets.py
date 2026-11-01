"""Check the agent skill in ``skills/nonconform``: metadata, links and examples."""

from __future__ import annotations

import logging
import re
from pathlib import Path

import pytest

SKILL_DIR = Path(__file__).parents[2] / "skills" / "nonconform"
SKILL_FILE = SKILL_DIR / "SKILL.md"
DOC_FILES = [SKILL_FILE, *sorted((SKILL_DIR / "references").glob("*.md"))]
CODE_BLOCK = re.compile(r"```python\n(.*?)```", re.DOTALL)


def _frontmatter(text: str) -> dict[str, str]:
    match = re.match(r"---\n(.*?)\n---\n", text, re.DOTALL)
    assert match, "SKILL.md must start with a YAML frontmatter block"
    fields = {}
    for line in match.group(1).splitlines():
        key, _, value = line.partition(":")
        fields[key.strip()] = value.strip()
    return fields


def _snippets() -> list:
    params = []
    for path in DOC_FILES:
        blocks = CODE_BLOCK.findall(path.read_text(encoding="utf-8"))
        name = path.relative_to(SKILL_DIR).as_posix()
        params.extend(
            pytest.param(code, id=f"{name}-{index}")
            for index, code in enumerate(blocks, start=1)
        )
    return params


def test_skill_frontmatter_follows_agent_skills_format():
    fields = _frontmatter(SKILL_FILE.read_text(encoding="utf-8"))
    assert fields["name"] == SKILL_DIR.name
    assert re.fullmatch(r"[a-z0-9]+(-[a-z0-9]+)*", fields["name"])
    assert 0 < len(fields["description"]) <= 1024
    assert "<" not in fields["description"]
    assert ">" not in fields["description"]


def test_skill_reference_links_resolve():
    text = SKILL_FILE.read_text(encoding="utf-8")
    referenced = set(re.findall(r"references/[\w.-]+\.md", text))
    assert referenced, "SKILL.md should point to its reference files"
    for relative in referenced:
        assert (SKILL_DIR / relative).is_file(), relative


@pytest.mark.parametrize("code", _snippets())
def test_skill_snippet_runs(code):
    if "pyod" in code:
        pytest.importorskip("pyod")
    logger = logging.getLogger("nonconform")
    level, handlers = logger.level, list(logger.handlers)
    try:
        # A "<...>" filename tells coverage the code has no source file to report.
        exec(compile(code, "<skill snippet>", "exec"), {"__name__": "__main__"})
    finally:
        logger.setLevel(level)
        logger.handlers[:] = handlers
