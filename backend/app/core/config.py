from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    llm_base_url: str
    llm_api_key: str
    llm_model: str

    model_config = SettingsConfigDict(
        env_file=ROOT_DIR / ".env",        
        extra="ignore",
    )


settings = Settings()