"""Plain text from an arXiv (LaTeXML) HTML page: headings, paragraphs, list items, table cells, figure captions;
math is written as its LaTeX alttext. Standard library only, so it can run with `python3 -I` on untrusted downloads.

usage: python3 -I scripts/common/html2text.py page.html > page.txt
"""

import sys
from html.parser import HTMLParser

BLOCK = {"p", "div", "section", "h1", "h2", "h3", "h4", "h5", "h6", "li", "tr", "figcaption", "caption", "br", "table",
         "blockquote", "dt", "dd"}
SKIP = {"script", "style", "nav", "header", "footer", "button", "svg"}


class Extract(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.out, self.skip, self.math = [], 0, 0

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag in SKIP:
            self.skip += 1
            return
        if tag == "math":
            self.math += 1
            if not self.skip and self.math == 1:
                self.out.append(" $" + (a.get("alttext") or "").strip() + "$ ")
            return
        if tag in BLOCK:
            self.out.append("\n")
            if tag.startswith("h") and len(tag) == 2:
                self.out.append("#" * int(tag[1]) + " ")
        if tag in ("td", "th"):
            self.out.append(" | ")

    def handle_endtag(self, tag):
        if tag in SKIP:
            self.skip = max(0, self.skip - 1)
        elif tag == "math":
            self.math = max(0, self.math - 1)
        elif tag in BLOCK:
            self.out.append("\n")

    def handle_data(self, data):
        if self.skip or self.math:
            return
        self.out.append(data)


def main():
    p = Extract()
    p.feed(open(sys.argv[1], encoding="utf-8", errors="replace").read())
    text = "".join(p.out)
    lines = [" ".join(l.split()) for l in text.splitlines()]
    out, blank = [], 0
    for l in lines:
        if not l:
            blank += 1
            if blank > 1:
                continue
        else:
            blank = 0
        out.append(l)
    sys.stdout.write("\n".join(out).strip() + "\n")


if __name__ == "__main__":
    main()
