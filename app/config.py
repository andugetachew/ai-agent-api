from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    anthropic_api_key: str
    agent_model: str = "claude-sonnet-4-6"
    agent_max_steps: int = 10
    agent_max_seconds: float = 90.0
    database_url: str
    celery_broker_url: str
    serper_api_key: str | None = None
    rate_limit: str = "5/minute"

    # Multiple model provider support -- OpenAI and Ollama are optional,
    # only needed if a run actually requests that provider.
    openai_api_key: str | None = None
    openai_model: str = "gpt-4o"
    ollama_base_url: str = "http://localhost:11434/v1"
    ollama_model: str = "llama3.1"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


settings = Settings()