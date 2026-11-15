"""Docs may only quote headline numbers that exist in reports/final_results.json (scripts/09_results_table.py).

1. Every number on a line that talks about a headline KPI (macro-F1, balanced accuracy, detection, early warning,
   delay, false alarms, leakage gap / random split) must be a rendering of a value in final_results.json.
2. Stale v0.1 claims may appear only in docs/WORK_LOG.md (history).
3. Two words describing the author's role are not used anywhere in the repo.
"""
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "reports" / "final_results.json"
KPI_DOCS = ["README.md", "docs/LAUNCH_KIT.md", "docs/MODEL_CARD.md", "docs/PRESENTATION_OUTLINE.md",
            "marketing/assets/facts_sheet.md"]
KPI_WORDS = re.compile(r"macro[- ]?f1|balanced accuracy|detect|early[- ]warning|delay|false alarm|"
                       r"leakage gap|random (?:split|window)|unseen wells say|×", re.I)
# numbers that are structure, not results (years, versions, dataset facts, URLs/paths are stripped first)
STRUCTURAL = {"0", "1", "2", "3", "4", "5", "10", "100", "1.0", "0.1", "2.0.0", "v1.0", "v0.1", "2019", "2026",
              "181", "106223", "3W", "1,300"}
STALE = [r"65 ?%", r"F1 0\.87", r"(?<![\d.])0\.73(?!\d)", r"(?<![\d.])0\.63(?!\d)", r"3\.4 ?[×x]",
         r"(?<![\d.])0\.975(?!\d)", r"(?<![\d.])0\.291(?!\d)", r"(?<![\d.])97 ?%", r"(?<![\d.])29 ?%", r"91 ?h\b"]
BANNED = ["".join(["O", "J", "T"]), "".join(["tra", "inee"])]


def _numbers_in(obj, out):
    if isinstance(obj, dict):
        for v in obj.values():
            _numbers_in(v, out)
    elif isinstance(obj, list):
        for v in obj:
            _numbers_in(v, out)
    elif isinstance(obj, bool):
        return
    elif isinstance(obj, (int, float)):
        v = float(obj)
        for x in (v, 100 * v, -v, -100 * v):
            for d in (0, 1, 2, 3):
                out.add(f"{x:.{d}f}")
        if v == int(v):
            out.add(f"{int(v):,}")
    elif isinstance(obj, str):
        out.update(re.findall(r"\d+(?:\.\d+)?", obj))


@pytest.fixture(scope="module")
def allowed():
    if not RESULTS.exists():
        pytest.skip("reports/final_results.json missing (run scripts/09_results_table.py)")
    out = set(STRUCTURAL)
    _numbers_in(json.loads(RESULTS.read_text(encoding="utf-8")), out)
    return out


def _kpi_numbers(line: str) -> list[str]:
    line = re.sub(r"https?://\S+|`[^`]*`|\[[^\]]*\]\([^)]*\)|E\d{2}|WELL-\d+|fold \d|v\d\.\d(?:\.\d)?", " ", line)
    return re.findall(r"(?<![\w.])\d[\d,]*(?:\.\d+)?", line)


@pytest.mark.parametrize("doc", KPI_DOCS)
def test_kpi_numbers_come_from_final_results(doc, allowed):
    path = ROOT / doc
    if not path.exists():
        pytest.skip(f"{doc} not written yet")
    bad = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not KPI_WORDS.search(line):
            continue
        for n in _kpi_numbers(line):
            if n not in allowed and n.replace(",", "") not in allowed:
                bad.append(f"{doc}:{i}: '{n}' in: {line.strip()[:110]}")
    assert not bad, "Numbers not found in reports/final_results.json:\n" + "\n".join(bad)


def _text_files():
    skip_dirs = {".git", "data", "models", ".venv", "venv", "__pycache__", ".pytest_cache", "node_modules"}
    for p in ROOT.rglob("*"):
        if p.is_file() and not (set(p.relative_to(ROOT).parts) & skip_dirs) and \
                p.suffix in {".md", ".py", ".yaml", ".yml", ".cff", ".txt", ".toml", ".json", ".ps1", ".sh"}:
            yield p


def test_no_stale_claims_outside_work_log():
    bad = []
    for p in _text_files():
        rel = p.relative_to(ROOT).as_posix()
        if (p.suffix != ".md" and rel != "CITATION.cff") or rel == "docs/WORK_LOG.md" or rel.startswith("reports/"):
            continue
        text = p.read_text(encoding="utf-8", errors="ignore")
        for pat in STALE:
            for m in re.finditer(pat, text):
                bad.append(f"{rel}: stale claim '{m.group(0)}'")
    assert not bad, "\n".join(bad)


def test_banned_words_absent():
    bad = [f"{p.relative_to(ROOT).as_posix()}: '{w}'" for p in _text_files()
           for w in BANNED if re.search(rf"\b{w}\b", p.read_text(encoding="utf-8", errors="ignore"), re.I)]
    assert not bad, "\n".join(bad)
