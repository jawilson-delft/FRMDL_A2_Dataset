from pathlib import Path

_TARGET = Path(__file__).resolve().parent / "scripts/analysis/audit/audit_whole_domain_structure.py"
exec(compile(_TARGET.read_text(encoding="utf-8"), str(_TARGET), "exec"), globals(), globals())
