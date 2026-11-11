"""Build every data artefact: python -m factory_copilot.data.build"""

from __future__ import annotations

from factory_copilot.config import DATA_DIR
from factory_copilot.data.generate import write_dataset
from factory_copilot.data.sops import write_sops


def build_all(units: int = 60, seed: int = 7) -> None:
    meta = write_dataset(DATA_DIR / "generated", units=units, seed=seed)
    n = write_sops(DATA_DIR / "sops")
    print(f"sensors: {meta['rows']} rows / {meta['units']} units; maintenance log + {n} SOPs written to {DATA_DIR}")


if __name__ == "__main__":
    build_all()
