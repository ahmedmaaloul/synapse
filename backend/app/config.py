"""
Project Synapse — Configuration
Loads environment variables via pydantic-settings.
"""

from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings loaded from environment / .env file."""

    # Google AI
    google_api_key: str = ""

    # Neo4j
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "synapse_secret"

    # App
    app_name: str = "Project Synapse"
    debug: bool = True

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


@lru_cache
def get_settings() -> Settings:
    return Settings()
