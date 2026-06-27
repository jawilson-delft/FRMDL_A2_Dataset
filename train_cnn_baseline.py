from pathlib import Path

_TARGET = Path(__file__).resolve().parent / "scripts/training/train_cnn_baseline.py"
exec(compile(_TARGET.read_text(encoding="utf-8"), str(_TARGET), "exec"), globals(), globals())
