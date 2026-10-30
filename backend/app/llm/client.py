from openai import AsyncOpenAI
from app.core.config import settings

llm_client = AsyncOpenAI(
    base_url=settings.llm_base_url,
    api_key=settings.llm_api_key,
    timeout=60.0,
)