import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "d-opsd"))

from data_utils import get_gsm8k_questions  # noqa: E402


dataset = get_gsm8k_questions("train")
assert len(dataset) == 7473
assert dataset[0]["answer"] is not None
print(f"local GSM8K loader ok: {len(dataset)} examples")
