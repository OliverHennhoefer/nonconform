"""MkDocs hook to guard against Pygments filename=None crashes.

Some extension stacks may pass ``filename=None`` into ``HtmlFormatter``.
Recent Pygments versions call ``html.escape`` on that value and raise
``AttributeError`` because ``None`` has no ``replace`` method.
"""

from __future__ import annotations

from pygments.formatters.html import HtmlFormatter

_ORIGINAL_HTML_FORMATTER_INIT = HtmlFormatter.__init__


def _patched_html_formatter_init(self, **options):
    if options.get("filename") is None:
        options["filename"] = ""
    return _ORIGINAL_HTML_FORMATTER_INIT(self, **options)


def on_config(config):
    """Patch HtmlFormatter before any page rendering begins."""
    if HtmlFormatter.__init__ is not _patched_html_formatter_init:
        HtmlFormatter.__init__ = _patched_html_formatter_init
    return config
