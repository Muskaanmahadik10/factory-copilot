import pytest

from factory_copilot.config import ARTIFACTS_DIR, DATA_DIR


@pytest.fixture(scope="session", autouse=True)
def built_data():
    """Tests that need the real data/models build them once if missing (fast: ~20 s)."""
    if not (DATA_DIR / "generated" / "sensors.csv").exists():
        from factory_copilot.data.build import build_all
        build_all()
    if not (ARTIFACTS_DIR / "models.pkl").exists():
        from factory_copilot.analytics.train import main
        main()
