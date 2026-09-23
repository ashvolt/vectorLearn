"""Pass A: EPUB -> spans. Deterministic, no model involved.

EPUB is preferred over PDF because it *can* carry structure: code listings,
figures and headings as real elements. Many EPUBs in the wild do not, though
— anything Calibre converted from a PDF is a flat run of <p> elements with
the paragraphs shattered at the PDF's line breaks. Since that is a large
share of what people actually own, handling it is not an edge case.

Two sources of structure, in priority order:

  1. **The book's own table of contents** (`toc.py`). Even a flat EPUB keeps
     its NCX or nav document, with labelled entries anchored into the body.
     That gives exact heading text and real nesting depth — strictly better
     than guessing which paragraph looks like a heading.
  2. **The body markup**, where it exists: <h1>-<h6>, <pre>, <figure>.

When the body turns out to be flat, `recover.py` rejoins the fragments and
reclassifies code and figures before any span is emitted.

Implemented against the standard library only — a book parser that cannot be
run because a wheel failed to build is worse than no book parser.
"""

from __future__ import annotations

import posixpath
import re
import zipfile
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from xml.etree import ElementTree as ET

from ..ir import Span, SourceDoc, content_hash, slug
from . import recover
from .toc import TocEntry, anchors_by_doc, read_toc

BLOCK_TAGS = {
    "p": "prose", "div": "prose", "blockquote": "prose",
    "h1": "heading", "h2": "heading", "h3": "heading",
    "h4": "heading", "h5": "heading", "h6": "heading",
    "pre": "code", "table": "table", "li": "list",
    "figure": "figure", "figcaption": "figure",
}
SKIP_TAGS = {"script", "style", "head"}

SECTION_HEAD = re.compile(r"^\s*(?:§\s*)?(?:Chapter\s+)?(\d+(?:\.\d+)*)[.\s)]+\S")
EXERCISE_HEAD = re.compile(
    r"\b(exercise|problem set|problems|review question|self[- ]test|drill|"
    r"questions)s?\b", re.I
)


@dataclass
class Block:
    kind: str
    text: str
    level: int = 0
    anchors: list[str] = field(default_factory=list)
    has_img: bool = False
    alt: str | None = None
    synthetic: bool = False  # inserted from the TOC rather than found in markup


class _BlockExtractor(HTMLParser):
    """Flattens XHTML into ordered blocks, keeping anchors and images."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.blocks: list[Block] = []
        self._buf: list[str] = []
        self._kind = "prose"
        self._level = 0
        self._depth = 0
        self._skip = 0
        self._anchors: list[str] = []
        self._pending_anchors: list[str] = []
        self._has_img = False
        self._alt: str | None = None

    def _flush(self) -> None:
        text = recover.normalise("".join(self._buf))
        if text or self._has_img:
            self.blocks.append(Block(
                kind=self._kind, text=text, level=self._level,
                anchors=self._anchors, has_img=self._has_img, alt=self._alt,
            ))
        else:
            # Carry an empty block's anchors forward to the next real one.
            self._pending_anchors.extend(self._anchors)
        self._buf = []
        self._kind = "prose"
        self._level = 0
        self._anchors = list(self._pending_anchors)
        self._pending_anchors = []
        self._has_img = False
        self._alt = None

    def handle_starttag(self, tag, attrs):
        if tag in SKIP_TAGS:
            self._skip += 1
            return
        if self._skip:
            return

        a = dict(attrs)
        if a.get("id"):
            self._anchors.append(a["id"])

        if tag == "img":
            self._has_img = True
            self._alt = recover.useful_alt(a.get("alt"))
            if self._alt:
                self._buf.append(f"[figure: {self._alt}]")
            return

        kind = BLOCK_TAGS.get(tag)
        if kind is None:
            return
        if (self._buf or self._has_img) and self._depth == 0:
            self._flush()
        if self._depth == 0:
            self._kind = kind
            self._level = int(tag[1]) if kind == "heading" else 0
            self._depth = 1
        elif kind == "code":
            self._kind = "code"

    def handle_endtag(self, tag):
        if tag in SKIP_TAGS:
            self._skip = max(0, self._skip - 1)
            return
        if self._skip:
            return
        if tag in BLOCK_TAGS and self._depth:
            self._depth = 0
            self._flush()

    def handle_data(self, data):
        if not self._skip:
            self._buf.append(data)

    def close(self):  # type: ignore[override]
        super().close()
        self._flush()


# --- structure assembly -----------------------------------------------------

def _insert_toc_headings(blocks: list[Block], anchors: dict[str, TocEntry]) -> list[Block]:
    """Put a real heading where the TOC says a section begins."""
    if not anchors:
        return blocks
    out: list[Block] = []
    seen: set[str] = set()
    for block in blocks:
        for anchor in block.anchors:
            entry = anchors.get(anchor)
            if entry and anchor not in seen:
                seen.add(anchor)
                out.append(Block(
                    kind="heading", text=entry.label,
                    level=min(6, entry.depth), synthetic=True,
                ))
        out.append(block)
    return out


def _drop_heading_echo(blocks: list[Block]) -> list[Block]:
    """Remove the body copy of a heading we already inserted from the TOC.

    Converted books usually keep the heading as an ordinary paragraph, so
    without this every section title appears twice — once as structure and
    once as a stray one-line span that teaches nothing.

    Runs *after* rejoining, not before: a title long enough to wrap arrives
    as two fragments ("Packt is searching for authors like" / "you") and only
    becomes recognisable once they have been put back together.
    """
    titles = {
        b.text.casefold() for b in blocks if b.kind == "heading" and b.synthetic
    }
    if not titles:
        return blocks
    # Exact equality only. A standalone paragraph identical to a section title
    # is the title; a paragraph that merely starts with it is ordinary prose.
    return [
        b for b in blocks
        if not (b.kind == "prose" and b.text.casefold() in titles)
    ]


def _recover_flat(blocks: list[Block]) -> list[Block]:
    """Reclassify and rejoin a document that lost its markup."""
    width = recover.line_width(
        [len(b.text) for b in blocks if b.kind == "prose"]
    )
    staged: list[Block] = []
    for b in blocks:
        if b.kind == "heading":
            staged.append(b)
        elif b.has_img and not b.text:
            staged.append(Block(kind="figure", text=f"[figure: {b.alt or 'unlabelled'}]",
                                anchors=b.anchors, has_img=True, alt=b.alt))
        elif b.kind == "prose" and recover.looks_like_code(b.text):
            staged.append(Block(kind="code", text=b.text, anchors=b.anchors))
        else:
            staged.append(b)

    merged: list[Block] = []
    for b in staged:
        prev = merged[-1] if merged else None
        if (
            prev is not None
            and prev.kind == b.kind
            and b.kind in ("prose", "code")
            and recover.rejoin(prev.text, b.text, width)
        ):
            joiner = "" if prev.text.endswith("-") else (
                "\n" if b.kind == "code" else " ")
            body = prev.text[:-1] if prev.text.endswith("-") else prev.text
            merged[-1] = Block(kind=prev.kind, text=f"{body}{joiner}{b.text}",
                               anchors=prev.anchors + b.anchors)
        else:
            merged.append(b)
    return _drop_heading_echo(merged)


def _opf_path(zf: zipfile.ZipFile) -> str:
    root = ET.fromstring(zf.read("META-INF/container.xml"))
    ns = {"c": "urn:oasis:names:tc:opendocument:xmlns:container"}
    rootfile = root.find(".//c:rootfile", ns)
    if rootfile is None or not rootfile.get("full-path"):
        raise ValueError("EPUB container.xml declares no rootfile")
    return rootfile.get("full-path")  # type: ignore[return-value]


def _spine_documents(zf: zipfile.ZipFile, opf: str) -> tuple[str, list[str]]:
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


def _classify(kind: str, heading_path: list[str]) -> str:
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
        opf = _opf_path(zf)
        title, docs = _spine_documents(zf, opf)
        toc = read_toc(zf, opf)
        by_doc = anchors_by_doc(toc)

        spans: list[Span] = []
        ordinal = 0
        heading_stack: list[tuple[int, str]] = []
        section: str | None = None
        chapter: str | None = None
        recovered_docs = 0

        for doc_path in docs:
            try:
                html = zf.read(doc_path).decode("utf-8", errors="replace")
            except KeyError:
                continue

            parser = _BlockExtractor()
            parser.feed(html)
            parser.close()
            blocks = parser.blocks

            flat = recover.looks_flat(
                [b.kind for b in blocks], [len(b.text.split()) for b in blocks]
            )
            blocks = _insert_toc_headings(blocks, by_doc.get(doc_path, {}))
            if flat:
                blocks = _recover_flat(blocks)
                recovered_docs += 1

            doc_id = slug(posixpath.basename(doc_path).rsplit(".", 1)[0])

            for block in blocks:
                if block.kind == "heading":
                    level = block.level or 1
                    heading_stack = [(l, h) for l, h in heading_stack if l < level]
                    heading_stack.append((level, block.text))
                    if level == 1:
                        chapter = block.text
                        # Section numbering restarts with the chapter; carrying
                        # it across would attribute one chapter's numbers to the
                        # next, which matters in back matter that lists them.
                        section = None
                    if m := SECTION_HEAD.match(block.text):
                        section = m.group(1)

                heading_path = [h for _, h in heading_stack]
                spans.append(Span(
                    span_id=f"{doc_id}:{ordinal:04d}",
                    doc_id=doc_id,
                    ordinal=ordinal,
                    kind=_classify(block.kind, heading_path),  # type: ignore[arg-type]
                    text=block.text,
                    section=section,
                    chapter=chapter,
                    heading_path=heading_path,
                ))
                ordinal += 1

    return SourceDoc(
        book_id=slug(title),
        title=title,
        source_hash=content_hash(raw),
        spans=spans,
        recovered_docs=recovered_docs,
        toc_entries=len(toc),
    )
