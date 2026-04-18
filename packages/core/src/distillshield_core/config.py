from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="DISTILLSHIELD_", env_file=".env", extra="ignore")

    env: str = "development"
    db_url: str = "sqlite:///./data/distillshield.db"
    data_dir: Path = Path("./data")
    model_dir: Path = Path("./data/models")
    experiment_dir: Path = Path("./data/experiments")
    frontend_origin: str = "http://localhost:5173"
    api_port: int = 8000

    trusted_lab_orgs: list[str] = Field(default_factory=lambda: ["trusted-lab", "university-ai-lab"])
    normal_threshold: float = 0.30
    lab_threshold: float = 0.45
    suspicious_threshold: float = 0.70
    block_threshold: float = 0.92


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.model_dir.mkdir(parents=True, exist_ok=True)
    settings.experiment_dir.mkdir(parents=True, exist_ok=True)
    return settings
