"""Render the Markdown report as a standalone, downloadable HTML document."""

from __future__ import annotations

from datetime import datetime


def _markdown_to_html(md: str) -> str:
    """Convert Markdown to HTML.

    Prefers the ``markdown`` package if installed; otherwise falls back to a
    minimal converter that handles the headings/bullets/bold we emit.
    """
    try:
        import markdown  # type: ignore

        return markdown.markdown(md, extensions=["extra", "sane_lists"])
    except Exception:
        return _minimal_markdown(md)


def _minimal_markdown(md: str) -> str:
    import html
    import re

    lines = md.splitlines()
    out = []
    in_list = False
    for raw in lines:
        line = raw.rstrip()
        if not line:
            if in_list:
                out.append("</ul>")
                in_list = False
            continue
        # Headings
        m = re.match(r"^(#{1,6})\s+(.*)$", line)
        if m:
            if in_list:
                out.append("</ul>")
                in_list = False
            level = len(m.group(1))
            out.append(f"<h{level}>{html.escape(m.group(2))}</h{level}>")
            continue
        if line.strip() == "---":
            out.append("<hr/>")
            continue
        # Bullets
        if line.lstrip().startswith(("- ", "* ")):
            if not in_list:
                out.append("<ul>")
                in_list = True
            text = line.lstrip()[2:]
            out.append(f"<li>{_inline(text)}</li>")
            continue
        if in_list:
            out.append("</ul>")
            in_list = False
        out.append(f"<p>{_inline(line)}</p>")
    if in_list:
        out.append("</ul>")
    return "\n".join(out)


def _inline(text: str) -> str:
    import re

    # Apply markdown formatting on raw text first, then escape.
    # **bold** must be matched before *italic* to avoid `**bold**` being parsed as `<i>*bold*</i>`.
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"<i>\1</i>", text)
    text = re.sub(r"`(.+?)`", r"<code>\1</code>", text)
    # Now escape HTML entities, protecting the tags we just created.
    text = text.replace("&", "&amp;")
    text = text.replace("<b>", "<strong>").replace("</b>", "</strong>")
    text = text.replace("<i>", "<em>").replace("</i>", "</em>")
    text = text.replace("<code>", "<code>").replace("</code>", "</code>")
    # Escape remaining < and >
    text = text.replace("<", "&lt;").replace(">", "&gt;")
    # Restore our tags
    text = text.replace("&lt;strong&gt;", "<strong>").replace(
        "&lt;/strong&gt;", "</strong>"
    )
    text = text.replace("&lt;em&gt;", "<em>").replace("&lt;/em&gt;", "</em>")
    text = text.replace("&lt;code&gt;", "<code>").replace("&lt;/code&gt;", "</code>")
    return text


_HEAD = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Insight Autopilot Report</title>
<style>
  body { font-family: -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
         max-width: 820px; margin: 40px auto; padding: 0 20px; color: #1f2933; line-height: 1.6;
         background: #fff; }
  h1 { border-bottom: 3px solid #2563eb; padding-bottom: 8px; }
  h2 { color: #2563eb; margin-top: 32px; }
  code { background: #f1f5f9; padding: 2px 5px; border-radius: 4px; }
  hr { border: none; border-top: 1px solid #e2e8f0; margin: 32px 0; }
  .meta { color: #64748b; font-size: 0.9em; }
  ul { padding-left: 22px; }
</style>
</head>
<body>
<h1>Insight Autopilot — Analysis Report</h1>
<p class="meta">Generated __TIMESTAMP____QUESTION__</p>
__BODY__
</body>
</html>
"""


def render_html(markdown_report: str, question: str = "", timestamp: str = "") -> str:
    """Return a full standalone HTML document for the given Markdown report."""
    body = _markdown_to_html(markdown_report)
    q = f" &middot; Question: &ldquo;{question}&rdquo;" if question else ""
    ts = timestamp or datetime.now().strftime("%Y-%m-%d %H:%M")
    return (
        _HEAD.replace("__TIMESTAMP__", ts)
        .replace("__QUESTION__", q)
        .replace("__BODY__", body)
    )
