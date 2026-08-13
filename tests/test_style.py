"""Style gate: every source file in this repo must be strict ASCII.

A personal quality rule enforced by CI, the same way any other convention
(formatting, lint) is enforced: automatically, on every push.
"""
import pathlib

HERE = pathlib.Path(__file__).parent.parent
CHECK = ["train.py", "gate.py", "build_report.py", "README.md",
         "tests/test_data.py", "tests/test_model.py", "tests/test_gate.py",
         "tests/test_style.py",
         ".github/workflows/ci.yml", ".github/workflows/cd.yml"]


def test_all_source_files_are_ascii():
    offenders = {}
    for rel in CHECK:
        p = HERE / rel
        if not p.exists():
            continue
        text = p.read_text(encoding="utf-8")
        bad = sorted({hex(ord(c)) for c in text if ord(c) > 127})
        if bad:
            offenders[rel] = bad
    assert not offenders, f"non-ASCII characters found: {offenders}"
