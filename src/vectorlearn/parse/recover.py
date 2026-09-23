"""Structure recovery for EPUBs that have none.

A PDF-converted EPUB arrives as a flat run of <p> elements: no headings, no
code blocks, no figures, and — worst of all — paragraphs shattered at the
PDF's *line* boundaries, so a single sentence becomes four blocks. Roughly
ten words per block is the signature.

Left alone, that input produces spans too small to teach from, a graph with
no chapters, and coverage numbers that mean nothing. None of the downstream
passes can compensate, so it has to be fixed here.

Three recoveries, in order of how much they matter:

  1. **Rejoin fragments.** A block that does not end in terminal punctuation,
     followed by one that starts mid-sentence, is one paragraph.
  2. **Classify code and output.** Listings survive conversion as prose, but
     their character composition is nothing like prose.
  3. **Mark figures.** An empty block holding only an <img> is a figure, and
     the alt text a converter invents ("Image 106") is not worth keeping.

Headings are not guessed at — they come from the book's own table of
contents, which is exact. See `toc.py`.
"""

from __future__ import annotations

import re

# --- fragment rejoining -----------------------------------------------------

# A paragraph that genuinely ended. Quotes and brackets may trail the stop.
_SENTENCE_END = re.compile(r'[.!?:;][\'"”’)\]]*\s*$')
# A block that cannot be the start of a new paragraph.
_CONTINUES = re.compile(r'^[a-z(\[]|^[0-9]+[),.]?\s+[a-z]')
# Bullets and enumerations start something new even when lowercase follows.
_LIST_START = re.compile(r'^\s*([-•*‣◦·]|\(?[a-z0-9]{1,3}[).])\s+')

MAX_JOINED_CHARS = 4000

# A justified PDF line runs to the full measure; the last line of a paragraph
# is whatever is left over. So a block at (near) full width is mid-paragraph
# and a visibly short one ends it — which works even when the line happens to
# end on a full stop, where punctuation tells us nothing.
FULL_LINE_RATIO = 0.86
LINE_WIDTH_PERCENTILE = 0.90


def line_width(lengths: list[int]) -> int:
    """The document's typical full line, in characters.

    Taken high in the distribution rather than at the median: most blocks are
    full lines, and the ones that are not are exactly what we want to treat
    as different.
    """
    usable = sorted(l for l in lengths if l > 0)
    if len(usable) < 20:
        return 0  # too little evidence; fall back to punctuation alone
    return usable[int(len(usable) * LINE_WIDTH_PERCENTILE) - 1]


def _ends_open(text: str) -> bool:
    return not _SENTENCE_END.search(text)


def is_full_line(text: str, width: int) -> bool:
    return bool(width) and len(text) >= width * FULL_LINE_RATIO


def rejoin(text_a: str, text_b: str, width: int = 0) -> bool:
    """Should these two consecutive prose blocks be one paragraph?"""
    if not text_a or not text_b:
        return False
    if len(text_a) > MAX_JOINED_CHARS:
        return False
    if _LIST_START.match(text_b):
        return False
    # A hyphen at a line break is a split word, not a dash.
    if text_a.endswith("-") and text_b[:1].islower():
        return True
    # A full-width line was broken by the typesetter, not by the author.
    if is_full_line(text_a, width):
        return True
    return _ends_open(text_a) and bool(_CONTINUES.match(text_b))


# --- code detection ---------------------------------------------------------

_CODE_LEAD = re.compile(
    r"^\s*(>>>|\$|#\s|//|import\s|from\s+\w+\s+import|def\s|class\s|return\s|"
    r"print\(|if\s+.*:|for\s+\w+\s+in\s|while\s+.*:|try:|except|with\s+.*:|"
    r"@\w+|\}|\]|\)|</?\w+>)"
)
_ASSIGNMENT = re.compile(r"^\s*[\w.\[\]]+\s*(=|\+=|-=|\*=)\s*\S")
_NUMERIC_ROW = re.compile(r"^\s*[\[\(]?\s*[-+]?\d[\d\s.,eE+\-]*[\]\)]?\s*$")
_PROSE_WORD = re.compile(r"\b(the|and|that|this|which|with|from|for|are|was)\b", re.I)


def symbol_density(text: str) -> float:
    if not text:
        return 0.0
    symbols = sum(1 for c in text if not c.isalnum() and not c.isspace())
    return symbols / len(text)


def looks_like_code(text: str) -> bool:
    """Identify a listing or program output that lost its <pre> wrapper."""
    stripped = text.strip()
    if not stripped or len(stripped) > 2000:
        return False

    if _NUMERIC_ROW.match(stripped):
        return True
    if _CODE_LEAD.match(stripped) or _ASSIGNMENT.match(stripped):
        return True

    # Dense in symbols and thin on connective words: a listing, not a sentence.
    if symbol_density(stripped) > 0.18 and not _PROSE_WORD.search(stripped):
        return True

    # Rows of bracketed numbers, the usual shape of printed array output.
    if stripped.count("[") >= 2 and sum(c.isdigit() for c in stripped) > len(stripped) * 0.3:
        return True

    return False


# --- figures ----------------------------------------------------------------

# Converters emit placeholder alt text that carries no information.
_JUNK_ALT = re.compile(r"^\s*(image|figure|img|picture|photo)[\s_-]*\d*\s*$", re.I)


def useful_alt(alt: str | None) -> str | None:
    if not alt or _JUNK_ALT.match(alt):
        return None
    return alt.strip() or None


# --- whitespace -------------------------------------------------------------

def normalise(text: str) -> str:
    """Collapse the artefacts of justified PDF text.

    Non-breaking spaces are the important one: converters use them for
    inter-word padding in justified lines, and they survive entity decoding
    as \\xa0, which ordinary whitespace collapsing misses.
    """
    text = text.replace("\xa0", " ").replace(" ", " ").replace(" ", " ")
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


# --- flatness ---------------------------------------------------------------

FLAT_WORDS_PER_BLOCK = 25

# Below this, a document is a title page or a colophon, not a chapter. Both
# signals below are statistical, and neither means anything on five blocks —
# guessing there produces false positives on perfectly good markup.
MIN_BLOCKS_FOR_EVIDENCE = 40


def looks_flat(kinds: list[str], word_counts: list[int]) -> bool:
    """Decide whether a document lost its structure in conversion.

    Two independent signals, either of which is decisive: almost no heading
    markup, or blocks so short they must be lines rather than paragraphs.
    """
    if len(kinds) < MIN_BLOCKS_FOR_EVIDENCE:
        return False
    headings = kinds.count("heading")
    structural = kinds.count("code") + kinds.count("figure")
    words = sum(word_counts)
    prose_blocks = max(1, len(kinds) - headings)

    no_structure = headings <= max(2, len(kinds) // 200) and structural == 0
    fragmented = words / prose_blocks < FLAT_WORDS_PER_BLOCK
    return no_structure or (fragmented and headings * 20 < len(kinds))
