"""Render the Kaggle write-up body to HTML for pasting into Kaggle's rich-text Content editor.

Kaggle's write-up Content box is a WYSIWYG editor: pasting raw markdown lands as literal text.
Pasting *rendered HTML* from a browser preserves headings, bold, tables, links and images.

    python scripts/make_writeup_html.py
    -> research/WRITEUP_KAGGLE_FORM.html

Then: open that file in Chrome, Ctrl+A, Ctrl+C, click into the Kaggle Content box, Ctrl+V.
"""
import re
from pathlib import Path

import markdown

BASE = Path(__file__).resolve().parent.parent
SRC = BASE / "research" / "WRITEUP_KAGGLE_FORM.md"
DST = BASE / "research" / "WRITEUP_KAGGLE_FORM.html"

body = markdown.markdown(
    SRC.read_text(encoding="utf-8"),
    extensions=["tables", "fenced_code", "sane_lists", "attr_list"],
)


def autolink(html: str) -> str:
    """Wrap bare http(s) URLs in anchors, but never inside a tag, an existing link, or code."""
    parts = re.split(r"(<[^>]+>)", html)
    out, skip = [], 0
    for part in parts:
        if part.startswith("<"):
            low = part.lower()
            if low.startswith(("<a", "<code", "<pre")):
                skip += 1
            elif low.startswith(("</a", "</code", "</pre")):
                skip = max(0, skip - 1)
            out.append(part)
        elif skip:
            out.append(part)
        else:
            out.append(re.sub(r"(https?://[^\s<>()\[\]]+)",
                              lambda m: '<a href="%s">%s</a>' % (m.group(1), m.group(1)),
                              part))
    return "".join(out)


body = autolink(body)

# Inline styles only. Kaggle strips most CSS on paste, but these make the page readable in the
# browser beforehand and survive as inline attributes where the editor keeps them.
STYLE = """
<style>
  body { max-width: 860px; margin: 32px auto; padding: 0 20px;
         font-family: -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
         font-size: 16px; line-height: 1.65; color: #16211c; }
  h1 { font-size: 28px; margin: 34px 0 10px; line-height: 1.25; }
  h2 { font-size: 22px; margin: 30px 0 10px; line-height: 1.3; }
  h3 { font-size: 18px; margin: 24px 0 8px; }
  p  { margin: 12px 0; }
  img { max-width: 100%; height: auto; display: block; margin: 18px 0; }
  table { border-collapse: collapse; margin: 16px 0; width: 100%; }
  th, td { border: 1px solid #d7ddd9; padding: 7px 11px; text-align: left; font-size: 15px; }
  th { background: #f2f5f3; }
  code { background: #f2f5f3; padding: 1px 5px; border-radius: 3px;
         font-family: "SFMono-Regular", Consolas, monospace; font-size: 90%; }
  pre { background: #f2f5f3; padding: 12px 14px; border-radius: 5px; overflow-x: auto; }
  pre code { background: none; padding: 0; }
  hr { border: none; border-top: 1px solid #d7ddd9; margin: 28px 0; }
  ul { margin: 12px 0; padding-left: 22px; }
  li { margin: 7px 0; }
  blockquote { border-left: 3px solid #c2733a; margin: 16px 0; padding: 2px 16px; color: #4a554f; }
</style>
"""

DST.write_text(
    "<!doctype html><html><head><meta charset='utf-8'>"
    "<title>Kaggle write-up body</title>" + STYLE + "</head><body>\n" + body + "\n</body></html>",
    encoding="utf-8",
)

print("wrote", DST)
print("  h2 %d | tables %d | images %d | links %d | bold %d"
      % (body.count("<h2"), body.count("<table"), body.count("<img"),
         body.count("<a href="), body.count("<strong")))
print("\nNext: open it in Chrome, Ctrl+A, Ctrl+C, then paste into Kaggle's Content box.")
