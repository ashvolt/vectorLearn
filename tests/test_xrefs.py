from vectorlearn.parse import mine_xrefs, parse_epub
from vectorlearn.parse.xrefs import xref_stats

from vectorlearn.fixtures import build_sample_epub as build


def _xrefs(tmp_path):
    return mine_xrefs(parse_epub(build(tmp_path / "b.epub")).spans)


def test_backward_references_are_found(tmp_path):
    xrefs = _xrefs(tmp_path)
    back = {(x.from_section, x.target) for x in xrefs if x.backward}
    # "Recall from Section 2.1" and "the swap operation ... in Section 2.2"
    assert ("7.1", "2.1") in back
    assert ("7.1", "2.2") in back


def test_forward_reference_is_not_a_prerequisite(tmp_path):
    xrefs = _xrefs(tmp_path)
    fwd = [x for x in xrefs if x.target == "9"]
    assert fwd, "'as we will see in Chapter 9' should be mined"
    assert all(not x.backward for x in fwd), "forward refs are not prerequisites"


def test_self_reference_is_ignored(tmp_path):
    xrefs = _xrefs(tmp_path)
    assert not [x for x in xrefs if x.from_section == x.target]


def test_stats_shape(tmp_path):
    s = xref_stats(_xrefs(tmp_path))
    assert s["total"] == s["backward"] + s["forward"]
    assert s["explicit_cue"] >= 1
