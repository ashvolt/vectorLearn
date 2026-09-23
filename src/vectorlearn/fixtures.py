"""A synthetic textbook chapter, used by the tests and by `vectorlearn qualify`.

It ships inside the package rather than the test tree because model
qualification has to run on a machine that has no real book yet — benchmarking
which local model can drive the pipeline is the first thing a new contributor
does, and it should need nothing but Ollama.

The content is deliberately shaped to expose model failure modes: numbered
sections, explicit author cross-references, a code listing with an HTML-escaped
operator, an exercise block, and a figure with alt text only.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

CONTAINER = """<?xml version="1.0"?>
<container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
  <rootfiles><rootfile full-path="OEBPS/content.opf"
    media-type="application/oebps-package+xml"/></rootfiles>
</container>"""

OPF = """<?xml version="1.0"?>
<package xmlns="http://www.idpf.org/2007/opf" version="3.0" unique-identifier="id">
  <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
    <dc:title>Fundamentals of Data Structures</dc:title>
    <dc:identifier id="id">urn:test:fods</dc:identifier>
  </metadata>
  <manifest>
    <item id="front" href="front.xhtml" media-type="application/xhtml+xml"/>
    <item id="c2" href="ch02.xhtml" media-type="application/xhtml+xml"/>
    <item id="c7" href="ch07.xhtml" media-type="application/xhtml+xml"/>
  </manifest>
  <spine>
    <itemref idref="front"/><itemref idref="c2"/><itemref idref="c7"/>
  </spine>
</package>"""

FRONT = """<html xmlns="http://www.w3.org/1999/xhtml"><body>
<h1>Fundamentals of Data Structures</h1>
<p>Copyright 2019. All rights reserved. No part of this book may be
reproduced without permission of the publisher.</p>
<h2>Preface</h2>
<p>This book grew out of a course the author taught for eleven years.</p>
</body></html>"""

CH2 = """<html xmlns="http://www.w3.org/1999/xhtml"><body>
<h1>2. Arrays and Indexing</h1>
<h2>2.1 Contiguous Storage</h2>
<p>An array stores its elements in a contiguous block of memory. Because
every element occupies the same number of bytes, the address of element
<em>i</em> can be computed directly as base + i * width. This is why array
indexing is a constant-time operation.</p>
<pre><code>def get(arr, i):
    return arr[i]        # O(1): one address computation</code></pre>
<h2>2.2 Swapping</h2>
<p>Many algorithms rearrange an array in place by exchanging pairs of
elements. A swap requires one temporary slot.</p>
<pre><code>def swap(a, i, j):
    a[i], a[j] = a[j], a[i]</code></pre>
<h2>Exercises</h2>
<p>2.1 Show that indexing remains constant time when the element width
is not a power of two.</p>
</body></html>"""

CH7 = """<html xmlns="http://www.w3.org/1999/xhtml"><body>
<h1>7. Quicksort</h1>
<h2>7.1 The Divide Step</h2>
<p>Quicksort sorts an array by partitioning it around a pivot element and
then sorting each side independently. Recall from Section 2.1 that indexing
is constant time; the partition routine relies on this to scan the array in
a single pass.</p>
<p>The partitioning procedure described in this section uses the swap
operation we introduced in Section 2.2.</p>
<h2>7.2 Lomuto Partition</h2>
<p>The Lomuto scheme maintains the invariant that every element to the left
of index i is less than or equal to the pivot. It performs a single left to
right scan.</p>
<pre><code>def partition(a, lo, hi):
    pivot = a[hi]
    i = lo
    for j in range(lo, hi):
        if a[j] &lt;= pivot:
            swap(a, i, j)
            i += 1
    swap(a, i, hi)
    return i</code></pre>
<figure><img src="fig7-1.png" alt="Lomuto partition scanning an array of eight elements"/>
<figcaption>Figure 7.1 The scan maintains the invariant at every step.</figcaption></figure>
<p>A common mistake is to place the pivot at the low end without adjusting
the loop bounds, which causes the recursion to fail to make progress on an
already sorted array.</p>
<h2>7.3 Average Case Analysis</h2>
<p>On random input the partition splits the array evenly on average, giving
a recurrence of T(n) = 2T(n/2) + O(n) and therefore O(n log n) expected
time. As we will see in Chapter 9, the same recurrence governs mergesort.</p>
<h2>Exercises</h2>
<p>7.2 Using the partition routine of Section 7.2, write a quicksort that
sorts the smaller side recursively and loops on the larger side.</p>
</body></html>"""


def build_sample_epub(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("mimetype", "application/epub+zip")
        z.writestr("META-INF/container.xml", CONTAINER)
        z.writestr("OEBPS/content.opf", OPF)
        z.writestr("OEBPS/front.xhtml", FRONT)
        z.writestr("OEBPS/ch02.xhtml", CH2)
        z.writestr("OEBPS/ch07.xhtml", CH7)
    return path


# Terms a model can only produce from its own priors — none of them appear
# anywhere in the text above. If a generated lesson mentions one, the model has
# drifted off the source and is teaching its own training data under the
# author's name. Deterministic, offline, and free: no judge required.
DRIFT_TERMS: tuple[str, ...] = (
    "hoare",            # the other partition scheme; the fixture teaches Lomuto only
    "median-of-three",
    "median of three",
    "introsort",
    "timsort",
    "heapsort",
    "tail call",
    "numpy",
    "std::sort",
    "dual-pivot",
    "insertion sort",   # the classic "small subarray" optimisation, never mentioned
)


def find_drift(text: str) -> list[str]:
    """Return drift terms present in generated text."""
    low = text.lower()
    return sorted({t for t in DRIFT_TERMS if t in low})


# --------------------------------------------------------------------------
# A PDF-converted book: the shape Calibre produces, which is a large share of
# what people actually own. No headings, no <pre>, no <figure> — every block a
# <p> of the same class, paragraphs shattered at the PDF's line breaks, and
# the structure surviving only in the NCX.
# --------------------------------------------------------------------------

FLAT_CONTAINER = CONTAINER

FLAT_OPF = """<?xml version="1.0"?>
<package xmlns="http://www.idpf.org/2007/opf" version="2.0" unique-identifier="id">
  <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
    <dc:title>Converted Data Structures</dc:title>
    <dc:identifier id="id">urn:test:flat</dc:identifier>
  </metadata>
  <manifest>
    <item id="ncx" href="toc.ncx" media-type="application/x-dtbncx+xml"/>
    <item id="body" href="index_split_000.html" media-type="application/xhtml+xml"/>
  </manifest>
  <spine toc="ncx"><itemref idref="body"/></spine>
</package>"""

FLAT_NCX = """<?xml version="1.0"?>
<ncx xmlns="http://www.daisy.org/z3986/2005/ncx/" version="2005-1">
 <navMap>
  <navPoint id="n1"><navLabel><text>Sorting</text></navLabel>
    <content src="index_split_000.html#p10"/>
    <navPoint id="n2"><navLabel><text>Quicksort partitioning</text></navLabel>
      <content src="index_split_000.html#p20"/></navPoint>
    <navPoint id="n3">
      <navLabel><text>A heading long enough that the typesetter wrapped it</text></navLabel>
      <content src="index_split_000.html#p30"/></navPoint>
  </navPoint>
 </navMap>
</ncx>"""

# Lines run to a fixed measure, as justified PDF text does; a visibly short
# line is the end of a paragraph. Long enough to clear the evidence floor,
# because flatness is a statistical judgement and five blocks decide nothing.
_FILLER = [
    ("Comparison sorts cannot do better than n log n in the worst case, and the",
     "proof of that bound is worth following once even if the result is already",
     "familiar to you."),
    ("An in-place algorithm rearranges the input array itself rather than building",
     "a second one, which matters when the array is large enough that a copy would",
     "not fit in memory."),
    ("Stability means that two elements comparing equal keep their original relative",
     "order, a property that matters whenever records are sorted on one field after",
     "having been sorted on another."),
    ("The recursion depth of quicksort is bounded by the number of times the larger",
     "side can be halved, so sorting the smaller side first keeps the stack shallow",
     "even on adversarial input."),
    ("Choosing the last element as the pivot is simple to implement and adequate on",
     "random data, though it degrades badly on input that is already in order, as",
     "the exercises explore."),
    ("Merge sort trades memory for a guarantee: it needs room for a second array,",
     "and in exchange its worst case is the same as its average case, which matters",
     "when latency has to be predictable."),
    ("A sorting network fixes the sequence of comparisons in advance, independently",
     "of the data, which makes it a poor general-purpose choice but an excellent one",
     "for hardware."),
    ("Counting sort abandons comparisons entirely and runs in linear time, at the",
     "cost of requiring keys drawn from a small known range, which is a much stronger",
     "precondition than it first appears."),
    ("The practical advice at the end of this chapter is to use the sort your standard",
     "library provides, and to understand these algorithms so that you can tell when",
     "that advice stops applying."),
]


def _filler_blocks() -> str:
    out = []
    for para in _FILLER:
        out += [f'<p class="calibre1">{line}</p>' for line in para]
    return "\n".join(out)


FLAT_BODY = """<html xmlns="http://www.w3.org/1999/xhtml"><body>
<p class="calibre1"><a id="p10"></a>Sorting</p>
<p class="calibre1">Sorting an array means rearranging its elements so that they appear in a known</p>
<p class="calibre1">order. Almost every algorithm in this chapter depends on the ability to compare</p>
<p class="calibre1">two elements and to exchange them.</p>
<p class="calibre1"><a id="p20"></a>Quicksort partitioning</p>
<p class="calibre1">The Lomuto scheme maintains the invariant that every element to the left of the</p>
<p class="calibre1">index i is less than or equal to the pivot. Recall from Section 2.1 that indexing</p>
<p class="calibre1">is a constant-time operation.</p>
<p class="calibre1">def partition(a, lo, hi):</p>
<p class="calibre1">    pivot = a[hi]</p>
<p class="calibre1">    i = lo</p>
<p class="calibre1">    return i</p>
<p class="calibre1"><img src="fig1.png" alt="Image 42" class="calibre2"/></p>
<p class="calibre1"><a id="p30"></a>A heading long enough that the typesetter wrapped</p>
<p class="calibre1">it</p>
<p class="calibre1">Some following prose that belongs to the wrapped heading's section and runs on</p>
<p class="calibre1">for a second line before stopping.</p>
__FILLER__
</body></html>"""


def build_flat_epub(path: Path) -> Path:
    """A PDF-converted EPUB, for exercising structure recovery."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("mimetype", "application/epub+zip")
        z.writestr("META-INF/container.xml", FLAT_CONTAINER)
        z.writestr("OEBPS/content.opf", FLAT_OPF)
        z.writestr("OEBPS/toc.ncx", FLAT_NCX)
        z.writestr("OEBPS/index_split_000.html",
                   FLAT_BODY.replace("__FILLER__", _filler_blocks()))
    return path
