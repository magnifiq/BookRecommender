import os
from dataclasses import dataclass


@dataclass(frozen=True)
class FrontendConfig:
    backend_api_url: str = "http://127.0.0.1:8000"
    host: str = "127.0.0.1"
    port: int = 7860
    request_timeout_seconds: float = 90.0

    @classmethod
    def from_env(cls) -> "FrontendConfig":
        return cls(
            backend_api_url=os.getenv("BOOKRECOMMENDER_API_URL", "http://127.0.0.1:8000").rstrip("/"),
            host=os.getenv("GRADIO_SERVER_NAME", "127.0.0.1"),
            port=int(os.getenv("GRADIO_SERVER_PORT", "7860")),
            request_timeout_seconds=float(os.getenv("BOOKRECOMMENDER_API_TIMEOUT", "90")),
        )
