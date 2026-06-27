from pathlib import Path

_TARGET = Path(__file__).resolve().parent / "scripts/analysis/diagnostics/diagnose_val_split.py"
exec(compile(_TARGET.read_text(encoding="utf-8"), str(_TARGET), "exec"), globals(), globals())
