"""Configuration: database.yaml, data.yaml, phase/source."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs"

DataSource = Literal["mock", "mimic"]
MimicSource = Literal["demo", "full", "mimic"]
DataPhase = Literal["P0", "P1", "P2"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    database_url: str = "postgresql+psycopg://icu_dev:icu_dev@localhost:5432/icu_decision"


@lru_cache
def get_settings() -> Settings:
    db = load_yaml("database.yaml")
    url = db.get("database", {}).get("url")
    if url:
        return Settings(database_url=url)
    return Settings()


def load_yaml(name: str) -> dict[str, Any]:
    path = CONFIG_DIR / name
    if not path.exists():
        example = CONFIG_DIR / f"{name}.example"
        path = example if example.exists() else path
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def get_data_source() -> DataSource:
    return load_yaml("data.yaml").get("source", "mock")


def get_data_phase() -> DataPhase:
    return load_yaml("data.yaml").get("phase", "P0")


def get_mimic_source() -> MimicSource:
    layer0 = load_yaml("database.yaml").get("layer0", {})
    return layer0.get("mimic_source", "demo")


def get_layer0_dsn() -> str | None:
    layer0 = load_yaml("database.yaml").get("layer0", {})
    source = get_mimic_source()
    key = {"demo": "demo_dsn", "full": "full_dsn", "mimic": "mimic_dsn"}.get(source, "demo_dsn")
    return layer0.get(key)
