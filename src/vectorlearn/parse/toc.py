"""Reading an EPUB's own table of contents.

Many real-world EPUBs — anything converted from PDF by Calibre, which is a
large share of what people actually own — carry no semantic markup in the
body at all: no <h1>, no <pre>, no <figure>, every block a <p> with the same
class. The structure has not been lost, though. It is in the navigation
document, as labelled entries with anchors pointing into the body.

So the TOC is not a nicety here, it is the primary structural source. It
gives real heading text, real nesting depth, and an exact position — which
is strictly better than any heuristic guess at what looks like a heading.

Both formats are read: EPUB 2's NCX and EPUB 3's nav document.
"""

from __future__ import annotations

import posixpath
import re
import zipfile
from dataclasses import dataclass
from html.parser import HTMLParser
from xml.etree import ElementTree as ET

NCX_NS = {"n": "http://www.daisy.org/z3986/2005/ncx/"}
XHTML_NS = "{http://www.w3.org/1999/xhtml}"


@dataclass(frozen=True)
class TocEntry:
    depth: int          # 1 = chapter, deeper = subsection
    label: str
    doc: str            # normalised path of the content document
    anchor: str | None  # element id within it, if the link carries one


def _clean(text: str | None) -> str:
    """Undo the double-escaping Calibre leaves in nav labels."""
    if not text:
        return ""
    for _ in range(2):
        text = re.sub(r"&#x([0-9A-Fa-f]+);", lambda m: chr(int(m.group(1), 16)), text)
        text = re.sub(r"&#(\d+);", lambda m: chr(int(m.group(1))), text)
        text = text.replace("&amp;", "&")
    return re.sub(r"\s+", " ", text.replace("\xa0", " ")).strip()


def _split_href(href: str, base: str) -> tuple[str, str | None]:
    path, _, anchor = href.partition("#")
    return posixpath.normpath(posixpath.join(base, path)), (anchor or None)


def _from_ncx(raw: bytes, base: str) -> list[TocEntry]:
    root = ET.fromstring(raw)
    nav_map = root.find("n:navMap", NCX_NS)
    if nav_map is None:
        return []

    out: list[TocEntry] = []

    def walk(node, depth: int) -> None:
        for point in node.findall("n:navPoint", NCX_NS):
            label = point.find("n:navLabel/n:text", NCX_NS)
            content = point.find("n:content", NCX_NS)
            if content is not None and content.get("src"):
                doc, anchor = _split_href(content.get("src"), base)  # type: ignore[arg-type]
                text = _clean(label.text if label is not None else None)
                if text:
                    out.append(TocEntry(depth, text, doc, anchor))
            walk(point, depth + 1)

    walk(nav_map, 1)
    return out


class _NavParser(HTMLParser):
    """EPUB 3 nav documents are XHTML: nested <ol>/<li>/<a>."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.entries: list[tuple[int, str, str]] = []
        self._depth = 0
        self._href: str | None = None
        self._buf: list[str] = []
        self._in_toc = False

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if a.get("epub:type") == "toc" or a.get("role") == "doc-toc":
            self._in_toc = True
        if tag == "ol":
            self._depth += 1
        elif tag == "a" and a.get("href"):
            self._href, self._buf = a["href"], []

    def handle_endtag(self, tag):
        if tag == "ol":
            self._depth = max(0, self._depth - 1)
        elif tag == "a" and self._href:
            label = re.sub(r"\s+", " ", "".join(self._buf)).strip()
            if label:
                self.entries.append((max(1, self._depth), label, self._href))
            self._href, self._buf = None, []

    def handle_data(self, data):
        if self._href is not None:
            self._buf.append(data)


def _from_nav(raw: bytes, base: str) -> list[TocEntry]:
    parser = _NavParser()
    parser.feed(raw.decode("utf-8", errors="replace"))
    parser.close()
    out = []
    for depth, label, href in parser.entries:
        doc, anchor = _split_href(href, base)
        out.append(TocEntry(depth, _clean(label), doc, anchor))
    return out


def read_toc(zf: zipfile.ZipFile, opf_path: str) -> list[TocEntry]:
    """Return the book's navigation entries, in reading order."""
    base = posixpath.dirname(opf_path)
    try:
        root = ET.fromstring(zf.read(opf_path))
    except (KeyError, ET.ParseError):
        return []

    ns = {"opf": "http://www.idpf.org/2007/opf"}
    items = root.findall(".//opf:manifest/opf:item", ns)

    # EPUB 3 nav first: it carries nesting more reliably than a converted NCX.
    for item in items:
        if "nav" in (item.get("properties") or "").split():
            try:
                path, _ = _split_href(item.get("href", ""), base)
                if entries := _from_nav(zf.read(path), posixpath.dirname(path)):
                    return entries
            except (KeyError, ValueError):
                pass

    for item in items:
        if item.get("media-type") == "application/x-dtbncx+xml" or \
                (item.get("href") or "").endswith(".ncx"):
            try:
                path, _ = _split_href(item.get("href", ""), base)
                return _from_ncx(zf.read(path), posixpath.dirname(path))
            except (KeyError, ET.ParseError, ValueError):
                pass

    return []


def anchors_by_doc(toc: list[TocEntry]) -> dict[str, dict[str, TocEntry]]:
    """Index TOC entries by document and anchor id, for O(1) lookup."""
    out: dict[str, dict[str, TocEntry]] = {}
    for e in toc:
        if e.anchor:
            out.setdefault(e.doc, {})[e.anchor] = e
    return out
