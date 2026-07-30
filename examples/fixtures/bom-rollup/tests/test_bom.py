from src.bom import rollup_bom


def test_rollup_groups_by_part_and_sorts_references():
    components = [
        {"reference": "R10", "value": "10k", "footprint": "0402", "lcsc": "C25744"},
        {"reference": "R2", "value": "10k", "footprint": "0402", "lcsc": "C25744"},
        {"reference": "C1", "value": "1uF", "footprint": "0603", "lcsc": "C15849"},
    ]
    assert rollup_bom(components) == [
        {
            "lcsc": "C15849",
            "value": "1uF",
            "footprint": "0603",
            "quantity": 1,
            "references": ["C1"],
        },
        {
            "lcsc": "C25744",
            "value": "10k",
            "footprint": "0402",
            "quantity": 2,
            "references": ["R2", "R10"],
        },
    ]


def test_rollup_excludes_dnp_and_requires_manufacturer_part_number():
    components = [
        {"reference": "R1", "value": "DNP", "footprint": "0402", "lcsc": "C1"},
        {"reference": "R2", "value": "1k", "footprint": "0402", "lcsc": ""},
        {"reference": "R3", "value": "1k", "footprint": "0402", "lcsc": "C9"},
    ]
    assert rollup_bom(components) == [
        {
            "lcsc": "C9",
            "value": "1k",
            "footprint": "0402",
            "quantity": 1,
            "references": ["R3"],
        }
    ]


def test_rollup_does_not_merge_same_lcsc_with_conflicting_metadata():
    components = [
        {"reference": "U1", "value": "MCU-A", "footprint": "QFN", "lcsc": "C42"},
        {"reference": "U2", "value": "MCU-B", "footprint": "TQFP", "lcsc": "C42"},
    ]
    assert len(rollup_bom(components)) == 2
