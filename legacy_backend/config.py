"""
Application configuration.
All settings are loaded from environment variables via Pydantic Settings.
Never hardcode secrets — always use settings.* from this module.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # OpenAI
    openai_api_key: str

    # Neo4j
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str

    # Embedding model — never change mid-project (would require re-embedding all vectors)
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536

    # LLM model — for answer generation
    llm_model: str = "gpt-4o"
    # Fast/cheap model for text2Cypher generation (avoids hitting TPM limits on gpt-4o)
    llm_model_fast: str = "gpt-4o-mini"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


settings = Settings()
