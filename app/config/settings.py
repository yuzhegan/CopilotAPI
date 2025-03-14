"""Application configuration settings."""

import os
from typing import List, Dict, Any, Optional
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # API settings
    api_title: str = "CopilotAPI"
    api_description: str = "API for GitHub Copilot integration"
    api_version: str = "0.1.0"
    allow_origins_str: str = Field("*", env="ALLOW_ORIGINS")
    
    # GitHub Copilot settings
    github_copilot_token: Optional[str] = Field("", env="GITHUB_COPILOT_TOKEN")
    default_model: str = Field("claude-3.7-sonnet", env="DEFAULT_MODEL")
    fallback_model: Optional[str] = Field(None, env="FALLBACK_MODEL")
    
    # Cached token configuration
    copilot_config: Optional[Dict[str, Any]] = None
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )
    
    @field_validator("allow_origins_str")
    def parse_allow_origins(cls, v: str) -> str:
        return v
    
    @property
    def allow_origins(self) -> List[str]:
        """Parse the allow_origins string into a list."""
        if not self.allow_origins_str or self.allow_origins_str == "*":
            return ["*"]
        return [origin.strip() for origin in self.allow_origins_str.split(",")]

# Create global settings instance
settings = Settings()

def get_settings() -> Settings:
    """Return application settings."""
    return settings
