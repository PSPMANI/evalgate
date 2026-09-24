"""Style gate: every source file in this repo must be strict ASCII.

A personal quality rule enforced by CI, the same way any other convention
(formatting, lint) is enforced: automatically, on every push. Files are discovered by
pattern, so a new module cannot slip past the rule by not being listed.
"""
import pathlib

HERE = pathlib.Path(__file__).parent.parent
PATTERNS = ["*.py", "*.md", "*.toml", "evalgate/*.py", "tests/*.py", ".github/workflows/*.yml"]


def test_all_source_files_are_ascii():
    files = sorted({p for pat in PATTERNS for p in HERE.glob(pat)})
    assert len(files) >= 12, "style gate found suspiciously few files"
    offenders = {}
    for p in files:
        text = p.read_text(encoding="utf-8")
        bad = sorted({hex(ord(c)) for c in text if ord(c) > 127})
        if bad:
            offenders[str(p.relative_to(HERE))] = bad
    assert not offenders, f"non-ASCII characters found: {offenders}"
