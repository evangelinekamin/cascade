"""Tests for the rule-based query decomposer (services.query_decomposer).

Focus: the decomposer must preserve the attributes a stylist's search depends
on -- colour above all, then silhouette/cut -- instead of collapsing every
query down to a bare garment anchor (the bug that made "black pleated midi
skirt" search for "midi skirt").
"""

from services.query_decomposer import decompose_query, decompose_queries


def test_primary_keeps_full_descriptor_stack():
    # The whole point: colour AND every recognised descriptor survive into the
    # primary term -- not just colour + one modifier.
    terms = decompose_query("women's black pleated midi skirt")
    assert terms[0] == "black pleated midi skirt"   # colour + pleated + midi all kept
    assert "midi skirt" in terms                    # shorter recall variants still there


def test_pattern_words_survive():
    # patterns (floral, plaid...) are recognised descriptors too
    assert decompose_query("flowy black floral midi dress")[0] == "black floral midi dress"
    assert decompose_query("red floral satin midi dress")[0] == "red floral satin midi dress"


def test_silhouette_words_survive():
    assert decompose_query("olive cargo parachute pants")[0] == "olive cargo parachute pants"
    assert decompose_query("blue baggy bootcut jeans")[0] == "blue baggy bootcut jeans"
    assert decompose_query("white off shoulder bodycon dress")[0] == "white off-shoulder bodycon dress"


def test_short_query_passes_through_intact():
    assert decompose_query("black midi skirt") == ["black midi skirt"]
    assert decompose_query("red slip dress") == ["red slip dress"]


def test_no_color_keeps_prior_behaviour():
    terms = decompose_query("high waisted wide leg jeans")
    assert "high-waisted jeans" in terms
    assert all("black" not in t for t in terms)


def test_filler_stripped_to_anchor():
    assert decompose_query("cute going out top") == ["top"]


def test_decompose_queries_dedups_across_inputs():
    assert decompose_queries(["black midi dress", "black midi dress"]) == ["black midi dress"]
