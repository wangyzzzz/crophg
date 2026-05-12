from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SectionSpec:
    code: str
    paper_section: str
    title: str
    old_result: str
    public: bool
    description: str


SECTION_SPECS: tuple[SectionSpec, ...] = (
    SectionSpec("3.1A", "3.1A", "H-only deployment loss", "old 3.1", False, "H-only 在四类部署场景下的失稳。"),
    SectionSpec("3.1B", "3.1B", "G compensation and G+FULLH complementarity", "old 3.2", False, "G 的稳定基线与 G+FULLH 的条件性互补。"),
    SectionSpec("3.2A", "3.2A", "Effective G+H windows across scenarios", "old 3.3A", False, "四场景下 G+H 的 trait-specific 有效窗口。"),
    SectionSpec("3.2B", "3.2B", "Single-VI change under H-only", "old 3.3B", False, "H-only 条件下单一 VI 的跨场景变化。"),
    SectionSpec("3.2C", "3.2C", "Single-VI incremental value under G background", "old 3.3C", False, "G 背景下单一 VI 的真实增量与重排。"),
    SectionSpec("3.3A", "3.3A", "Final model definition and input representation", "old 3.4A", True, "最终模型定义与输入表示比较。"),
    SectionSpec("3.3B", "3.3B", "Predictive performance and compression gain", "old 3.4B", True, "以 H_FULL 为 baseline，比较 G+FULLH、H_ANCHOR_AUTO、G+H_ANCHOR_AUTO 的性能与压缩收益。"),
    SectionSpec("3.4A", "3.4A", "Growth-prefix performance curve", "old 3.5A", True, "回答 G+FULLH、H_ANCHOR_AUTO、G+H_ANCHOR_AUTO 是否比 H_FULL 更早建立有效预测。"),
    SectionSpec("3.4B", "3.4B", "Evolution of selected anchor×VI units", "old 3.5B", True, "表示层随生长推进的 anchor×VI 演化。"),
)

SECTION_INDEX = {spec.code: spec for spec in SECTION_SPECS}


def list_sections(*, public_only: bool = False) -> list[SectionSpec]:
    specs = list(SECTION_SPECS)
    if public_only:
        specs = [spec for spec in specs if spec.public]
    return specs


def get_section(code: str) -> SectionSpec:
    norm = str(code).strip().upper()
    if norm not in SECTION_INDEX:
        raise KeyError(f"Unknown section: {code}")
    return SECTION_INDEX[norm]
