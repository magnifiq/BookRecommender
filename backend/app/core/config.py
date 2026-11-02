from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    llm_base_url: str
    llm_api_key: str
    llm_model: str
    tool_execution_timeout_seconds: float = 10.0
    guardrail_max_user_message_length: int = 4000
    guardrail_max_output_length: int = 12000
    guardrail_max_repeated_characters: int = 100
    open_library_timeout_seconds: float = 4.0
    open_library_min_request_interval_seconds: float = 1.0
    open_library_user_agent: str = "BookRecommender/0.1"
    conversation_database_path: str = str(ROOT_DIR / "data" / "conversations.sqlite3")
    conversation_context_turns: int = 12
    plan_max_steps: int = 12
    plan_max_tool_calls: int = 8
    plan_max_duration_seconds: float = 60.0
    plan_max_step_output_bytes: int = 32000
    plan_tool_retry_limit: int = 1

    model_config = SettingsConfigDict(
        env_file=ROOT_DIR / ".env",        
        extra="ignore",
    )


settings = Settings()
