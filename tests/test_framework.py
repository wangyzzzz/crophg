from crophg.framework import get_section, list_sections


def test_public_section_count() -> None:
    public_specs = list_sections(public_only=True)
    assert len(public_specs) == 4
    assert {spec.code for spec in public_specs} == {"3.3A", "3.3B", "3.4A", "3.4B"}


def test_internal_section_lookup() -> None:
    spec = get_section("3.2C")
    assert spec.public is False
    assert spec.old_result == "old 3.3C"


def test_public_section_count_still_stable_after_internal_analysis_additions() -> None:
    public_specs = list_sections(public_only=True)
    assert len(public_specs) == 4
