"""Central settings. The ONLY place env vars are read (golden rule: no os.getenv elsewhere).

Paper vs live is decided here and only here:
  - TRIDENT_ENV=paper  -> paper keys, paper endpoint. Default. Safe.
  - TRIDENT_ENV=live   -> requires config/live.yaml armed=true AND live keys present,
                          otherwise raises at startup. Never soft-fallback to paper
                          silently in live mode: fail loudly instead.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "config"
CACHE_DIR = REPO_ROOT / "data" / "cache"
ARTIFACTS_DIR = REPO_ROOT / "artifacts"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=REPO_ROOT / ".env", env_file_encoding="utf-8", extra="ignore"
    )

    trident_env: Literal["paper", "live"] = "paper"

    # Broker — paper
    alpaca_api_key: str = ""
    alpaca_secret_key: str = ""
    alpaca_paper_base_url: str = "https://paper-api.alpaca.markets"
    # Broker — live (empty until docs/10_GO_LIVE.md gates pass)
    alpaca_live_api_key: str = ""
    alpaca_live_secret_key: str = ""

    # LLM
    anthropic_api_key: str = ""
    oracle_model: str = "claude-sonnet-4-6"
    bulk_model: str = "claude-haiku-4-5-20251001"

    # Data providers
    finnhub_api_key: str = ""
    alphavantage_api_key: str = ""
    newsapi_key: str = ""
    reddit_client_id: str = ""
    reddit_client_secret: str = ""
    reddit_user_agent: str = ""
    edgar_user_agent: str = ""
    news_user_agent: str = "trident-research/0.1"
    fmp_api_key: str = ""
    fred_api_key: str = ""
    polygon_api_key: str = ""
    tiingo_api_key: str = ""

    # Storage & alerts
    database_url: str = ""
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""

    # -------------------------------------------------------------- helpers
    @property
    def is_live(self) -> bool:
        return self.trident_env == "live"

    def broker_credentials(self) -> tuple[str, str, bool]:
        """(key, secret, paper_flag) — the single switch point for promotion."""
        if self.is_live:
            live_cfg = load_live_config()
            if not live_cfg.get("armed", False):
                raise RuntimeError(
                    "TRIDENT_ENV=live but config/live.yaml is not armed. "
                    "Walk docs/10_GO_LIVE.md gates first."
                )
            if not (self.alpaca_live_api_key and self.alpaca_live_secret_key):
                raise RuntimeError("Live mode requires ALPACA_LIVE_API_KEY/SECRET in .env.")
            return self.alpaca_live_api_key, self.alpaca_live_secret_key, False
        if not (self.alpaca_api_key and self.alpaca_secret_key):
            raise RuntimeError("Missing ALPACA_API_KEY/ALPACA_SECRET_KEY (paper) in .env.")
        return self.alpaca_api_key, self.alpaca_secret_key, True

    def require(self, *fields: str) -> None:
        """Fail loudly, listing every missing key at once (Phase 0 acceptance)."""
        missing = [f for f in fields if not getattr(self, f)]
        if missing:
            raise RuntimeError(
                "Missing required settings: "
                + ", ".join(m.upper() for m in missing)
                + " — fill them in .env (see .env.example for signup URLs)."
            )


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open() as f:
        return yaml.safe_load(f) or {}


def load_live_config() -> dict:
    return _load_yaml(CONFIG_DIR / "live.yaml")


def load_risk_limits() -> dict:
    return _load_yaml(CONFIG_DIR / "risk_limits.yaml")


def required_risk_limit(key: str, limits: dict | None = None) -> float:
    """A REQUIRED risk parameter — fail LOUD if it's missing rather than fall back to a silent
    in-code default (which is a magic number hiding a config gap). Pass an already-loaded
    `limits` dict when reading several keys so the yaml isn't re-read each call."""
    limits = limits if limits is not None else load_risk_limits()
    if key not in limits:
        raise KeyError(f"required risk limit '{key}' missing from config/risk_limits.yaml")
    return float(limits[key])


def load_universe() -> list[str]:
    data = _load_yaml(CONFIG_DIR / "universe.yaml")
    return [t.upper() for t in data.get("tickers", [])]


def load_sectors() -> dict[str, str]:
    """{TICKER: GICS sector} from universe.yaml — ORACLE's compare_peers groups on this.

    Static config rather than a derived field: EDGAR carries SIC, not GICS, and SIC would
    split these 40 names into groups of one. Tickers absent from the map have no peer group;
    callers must handle that rather than assume full coverage.
    """
    data = _load_yaml(CONFIG_DIR / "universe.yaml")
    return {t.upper(): s for t, s in (data.get("sectors") or {}).items()}


def load_cik_overrides() -> dict[str, str]:
    """{TICKER: zero-padded CIK} pins that override SEC's ticker->CIK registry.

    The registry is authoritative until it isn't; a bad entry resolves to a filing shell with
    no us-gaap facts, which reads as "this company reports nothing" rather than as an error.
    Each pin in universe.yaml carries the date and the evidence for it.
    """
    data = _load_yaml(CONFIG_DIR / "universe.yaml")
    return {t.upper(): f"{int(c):010d}" for t, c in (data.get("cik_overrides") or {}).items()}


def sector_peers(ticker: str) -> list[str]:
    """Other universe tickers in the same sector. Empty when the sector is unknown.

    Peer groups here are small by construction (Energy and Industrials hold 2 names each),
    so callers must report the peer count — a one-peer 'sector median' is not a median.
    """
    sectors = load_sectors()
    own = sectors.get(ticker.upper())
    if own is None:
        return []
    return [t for t, s in sectors.items() if s == own and t != ticker.upper()]


def load_sentinel_sources() -> dict[str, dict]:
    """SENTINEL source registry (doc 04). Returns {source_name: config}. The pipeline
    reads this — no source names/weights are hardcoded; disabled/missing-env sources
    are renormalized away at aggregation time."""
    data = _load_yaml(CONFIG_DIR / "sentinel_sources.yaml")
    return data.get("sources", {})


@lru_cache
def get_settings() -> Settings:
    return Settings()
