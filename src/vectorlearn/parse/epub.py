"""Pass A: EPUB -> spans. Deterministic, no model involved.

EPUB is preferred over PDF on purpose: it is structured HTML, so code
listings, figures and headings survive intact. PDF is a layout format that
has forgotten it ever had structure, and recovering that structure is a
separate project.

Implemented against the standard library only - a book parser that cannot
be run because a wheel failed to build is worse than no book parser.
"""

from __future__ import annotations

import posixpath
import re
import zipfile
from html.parser import HTMLParser
from pathlib import Path
from xml.etree import ElementTree as ET

from ..ir import Span, SourceDoc, content_hash, slug

# Tags that end the current span and begin a new one.
BLOCK_TAGS = {
    "p": "prose", "div": "prose", "blockquote": "prose",
    "h1": "heading", "h2": "heading", "h3": "heading",
    "h4": "heading", "h5": "heading", "h6": "heading",
    "pre": "code", "table": "table", "li": "list",
    "figure": "figure", "figcaption": "figure",
}
SKIP_TAGS = {"script", "style", "head"}

# "7.2 Partitioning", "Chapter 7.", "§2.1", "7.2.3  Rotations"
SECTION_HEAD = re.compile(r"^\s*(?:§\s*)?(?:Chapter\s+)?(\d+(?:\.\d+)*)[.\s)]+\S")
EXERCISE_HEAD = re.compile(
    r"\b(exercise|problem set|problems|review question|self[- ]test|drill)s?\b", re.I
)


class _BlockExtractor(HTMLParser):
    """Flattens XHTML into an ordered list of (kind, text) blocks."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.blocks: list[tuple[str, str, int]] = []  # kind, text, heading_level
        self._buf: list[str] = []
        self._kind = "prose"
        self._level = 0
        self._depth = 0  # nesting depth of the block that set self._kind
        self._skip = 0
        self._img_pending = False

    # -- buffer management --------------------------------------------------
    def _flush(self) -> None:
        text = re.sub(r"[ \t\r\f\v]+", " ", "".join(self._buf)).strip()
        text = re.sub(r"\n{3,}", "\n\n", text)
        if text:
            self.blocks.append((self._kind, text, self._level))
        self._buf = []
        self._kind = "prose"
        self._level = 0

    # -- HTMLParser hooks ---------------------------------------------------
    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in SKIP_TAGS:
            self._skip += 1
            return
        if self._skip:
            return

        if tag == "img":
            alt = dict(attrs).get("alt") or ""
            self._img_pending = True
            if alt:
                self._buf.append(f"[figure: {alt}]")
            return

        kind = BLOCK_TAGS.get(tag)
        if kind is None:
            return

        # A nested block inside an open one (e.g. <code> in <pre>) does not
        # restart the span; only a sibling-level block does.
        if self._buf and self._depth == 0:
            self._flush()
        if self._depth == 0:
            self._kind = kind
            self._level = int(tag[1]) if kind == "heading" else 0
            self._depth = 1
        elif kind == "code":
            self._kind = "code"

    def handle_endtag(self, tag: str) -> None:
        if tag in SKIP_TAGS:
            self._skip = max(0, self._skip - 1)
            return
        if self._skip:
            return
        if tag in BLOCK_TAGS and self._depth:
            self._depth = 0
            self._flush()

    def handle_data(self, data: str) -> None:
        if not self._skip:
            self._buf.append(data)

    def close(self) -> None:  # type: ignore[override]
        super().close()
        self._flush()


def _opf_path(zf: zipfile.ZipFile) -> str:
    root = ET.fromstring(zf.read("META-INF/container.xml"))
    ns = {"c": "urn:oasis:names:tc:opendocument:xmlns:container"}
    rootfile = root.find(".//c:rootfile", ns)
    if rootfile is None or not rootfile.get("full-path"):
        raise ValueError("EPUB container.xml declares no rootfile")
    return rootfile.get("full-path")  # type: ignore[return-value]


def _spine_documents(zf: zipfile.ZipFile) -> tuple[str, list[str]]:
    """Return (book title, content document paths in reading order)."""
    opf = _opf_path(zf)
    base = posixpath.dirname(opf)
    root = ET.fromstring(zf.read(opf))
    ns = {"opf": "http://www.idpf.org/2007/opf", "dc": "http://purl.org/dc/elements/1.1/"}

    title_el = root.find(".//dc:title", ns)
    title = (title_el.text or "").strip() if title_el is not None else "Untitled"

    manifest = {
        item.get("id"): item.get("href")
        for item in root.findall(".//opf:manifest/opf:item", ns)
    }
    docs: list[str] = []
    for ref in root.findall(".//opf:spine/opf:itemref", ns):
        href = manifest.get(ref.get("idref"))
        if href and href.split("#")[0].endswith((".xhtml", ".html", ".htm")):
            docs.append(posixpath.normpath(posixpath.join(base, href)))
    return title, docs


def _classify(kind: str, text: str, heading_path: list[str]) -> str:
    """Refine a block kind using its surrounding headings."""
    if kind == "heading":
        return "heading"
    if any(EXERCISE_HEAD.search(h) for h in heading_path[-2:]):
        return "exercise"
    return kind


def parse_epub(path: str | Path) -> SourceDoc:
    """Parse an EPUB into an ordered, addressable list of spans."""
    path = Path(path)
    raw = path.read_bytes()

    with zipfile.ZipFile(path) as zf:
        title, docs = _spine_documents(zf)
        spans: list[Span] = []
        ordinal = 0
        heading_stack: list[tuple[int, str]] = []
        section = None

        for doc_path in docs:
            try:
                html = zf.read(doc_path).decode("utf-8", errors="replace")
            except KeyError:
                continue

            parser = _BlockExtractor()
            parser.feed(html)
            parser.close()
            doc_id = slug(posixpath.basename(doc_path).rsplit(".", 1)[0])

            for kind, text, level in parser.blocks:
                if kind == "heading":
                    heading_stack = [(l, h) for l, h in heading_stack if l < level]
                    heading_stack.append((level, text))
                    m = SECTION_HEAD.match(text)
                    if m:
                        section = m.group(1)

                heading_path = [h for _, h in heading_stack]
                spans.append(
                    Span(
                        span_id=f"{doc_id}:{ordinal:04d}",
                        doc_id=doc_id,
                        ordinal=ordinal,
                        kind=_classify(kind, text, heading_path),  # type: ignore[arg-type]
                        text=text,
                        section=section,
                        heading_path=heading_path,
                    )
                )
                ordinal += 1

    return SourceDoc(
        book_id=slug(title),
        title=title,
        source_hash=content_hash(raw),
        spans=spans,
    )
