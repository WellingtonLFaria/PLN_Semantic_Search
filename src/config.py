from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    articles_file_path: str
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


settings = Settings()  # type: ignore[call-arg]
