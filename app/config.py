from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


ROOT_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="SQL_LLM_",
        env_file=".env",
        extra="ignore",
    )

    host: str = "127.0.0.1"
    port: int = 8000
    default_model: str = "ollama"
    scratch_dir: Path = ROOT_DIR / "artifacts" / "scratch"
    own_model_dir: Path = ROOT_DIR / "artifacts" / "models" / "own-sql-tiny"
    own_ddl_model_dir: Path = ROOT_DIR / "artifacts" / "models" / "own-ddl"
    own_ddl_encoder_decoder_dir: Path = ROOT_DIR / "artifacts" / "models" / "own-ddl-encoder-decoder-v1"
    assistant_default_model: str = "slm"
    ollama_url: str = "http://127.0.0.1:11434"
    ollama_model: str = "qwen2.5-coder:0.5b"
    ollama_timeout_seconds: float = 30.0
    enable_execution: bool = False
    database_url: str = ""
    max_new_tokens: int = 192
    max_request_chars: int = 4000
    api_key: str = ""
    allow_destructive: bool = False
    retry_invalid_generation: bool = False
    data_dir: Path = ROOT_DIR / "data"
    session_db_path: Path = ROOT_DIR / "artifacts" / "sessions.sqlite3"


settings = Settings()
