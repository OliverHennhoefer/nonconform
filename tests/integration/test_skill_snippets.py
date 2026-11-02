"""Check the agent skill plugin: Claude Code and Codex packaging, links, examples."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parents[2]
PLUGIN_DIR = REPO_ROOT / "plugins" / "nonconform"
SKILL_DIR = PLUGIN_DIR / "skills" / "nonconform"
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


def _load_json(relative: str) -> dict:
    return json.loads((REPO_ROOT / relative).read_text(encoding="utf-8"))


def test_claude_code_marketplace_points_to_plugin():
    marketplace = _load_json(".claude-plugin/marketplace.json")
    (entry,) = marketplace["plugins"]
    assert entry["name"] == "nonconform"
    assert (REPO_ROOT / entry["source"]).resolve() == PLUGIN_DIR.resolve()


def test_codex_marketplace_points_to_plugin():
    marketplace = _load_json(".agents/plugins/marketplace.json")
    (entry,) = marketplace["plugins"]
    assert entry["name"] == "nonconform"
    assert entry["source"]["source"] == "local"
    assert (REPO_ROOT / entry["source"]["path"]).resolve() == PLUGIN_DIR.resolve()
    assert {"installation", "authentication"} <= entry["policy"].keys()
    assert entry["category"]


def test_plugin_manifest_matches_skill():
    manifest = _load_json("plugins/nonconform/plugin.json")
    assert manifest["name"] == "nonconform"
    assert re.fullmatch(r"\d+\.\d+\.\d+", manifest["version"])
    skills = sorted(p.parent.name for p in (PLUGIN_DIR / "skills").glob("*/SKILL.md"))
    assert skills == ["nonconform"]


def test_codex_skill_metadata():
    yaml = pytest.importorskip("yaml")
    metadata = yaml.safe_load(
        (SKILL_DIR / "agents" / "openai.yaml").read_text(encoding="utf-8")
    )
    assert metadata["interface"]["display_name"]
    assert metadata["interface"]["short_description"]
    assert isinstance(metadata["policy"]["allow_implicit_invocation"], bool)


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
