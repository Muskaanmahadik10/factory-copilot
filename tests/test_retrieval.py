import pytest

from factory_copilot.data.sops import write_sops
from factory_copilot.retrieval import build_index


@pytest.fixture(scope="module")
def index(tmp_path_factory):
    d = tmp_path_factory.mktemp("sops")
    write_sops(d)
    return build_index(d)


@pytest.mark.parametrize(
    "query,expected_doc",
    [
        ("coolant pressure falling and coolant flow low, coolant temperature rising", "SOP-C-201"),
        ("spindle vibration rising with spindle temperature high", "SOP-M-101"),
        ("position error spikes appear and disappear", "SOP-E-301"),
        ("filter differential pressure high, outlet pressure falling", "SOP-C-202"),
        ("position error increasing slowly over many cycles", "SOP-E-302"),
        ("who approves a work order", {"SOP-G-001", "SOP-G-005"}),  # both documents answer this; either is acceptable on top
        ("SOP-G-004 remaining useful life threshold", "SOP-G-004"),
    ],
)
def test_top_hit_is_right_document(index, query, expected_doc):
    hits = index.search(query, k=3)
    assert hits, query
    expected = expected_doc if isinstance(expected_doc, set) else {expected_doc}
    assert hits[0].chunk.doc_id in expected, [h.chunk.citation for h in hits]
    if isinstance(expected_doc, set):  # the approval rule itself must still be retrieved in the top 3
        assert any(h.chunk.doc_id == "SOP-G-001" for h in hits)


def test_citations_are_section_level(index):
    hit = index.search("bearing replacement procedure", k=1)[0]
    assert hit.chunk.citation.startswith("SOP-M-101 §")
    assert "Replacement" in hit.chunk.section or "Immediate" in hit.chunk.section


def test_diversity_limits_chunks_per_doc(index):
    hits = index.search("coolant", k=6)
    from collections import Counter
    assert max(Counter(h.chunk.doc_id for h in hits).values()) <= 2
