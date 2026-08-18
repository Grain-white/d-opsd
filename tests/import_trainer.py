import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "d-opsd"))
import d_opsd_config  # noqa: E402,F401
import d_opsd_trainer  # noqa: E402,F401

print("trainer import ok")
