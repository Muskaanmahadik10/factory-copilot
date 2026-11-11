"""Central configuration. Values come from environment variables or a .env file; nothing secret is stored in code."""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
ARTIFACTS_DIR = ROOT / "artifacts"


class Settings(BaseSettings):
    """Runtime settings.

    LLM_PROVIDER: "mock" (deterministic, no key), "openai_compatible" (OpenAI, Groq, Ollama, vLLM...), or "anthropic".
    """

    model_config = SettingsConfigDict(env_file=ROOT / ".env", env_file_encoding="utf-8", extra="ignore")

    llm_provider: str = "mock"
    llm_base_url: str = "https://api.groq.com/openai/v1"
    llm_api_key: str = ""
    cheap_model: str = "openai/gpt-oss-20b"
    strong_model: str = "openai/gpt-oss-120b"
    # USD per 1M tokens (input, output); used for cost logging and the per-incident budget.
    cheap_price_in: float = 0.10
    cheap_price_out: float = 0.50
    strong_price_in: float = 0.15
    strong_price_out: float = 0.75

    max_steps: int = 8
    max_cost_usd: float = 0.05
    request_timeout_s: float = 60.0

    data_dir: Path = DATA_DIR
    artifacts_dir: Path = ARTIFACTS_DIR


settings = Settings()
