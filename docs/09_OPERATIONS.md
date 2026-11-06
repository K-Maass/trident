# Optional local services

The offline demo and CI checks in the [README](../README.md) run without Docker or a broker. Use this setup for local persistence and development against a separate paper account.

## Local Postgres

1. Copy `.env.example` to `.env` and keep it local.
2. Generate a password with `openssl rand -hex 24`. Set `TRIDENT_DB_PASSWORD` to that value and `DATABASE_URL` to `postgresql+asyncpg://trident:<password>@127.0.0.1:5432/trident`, replacing `<password>` with the same value.
3. Run `make up`, then `make db-init` once Postgres is healthy. Both Postgres and Adminer listen on localhost. The Adminer server name is `db` when connecting from inside Compose.

The password initializes a new database volume. For an existing volume, manage the password in Postgres and update the local settings to match.

## Optional provider settings

For uncached SEC queries, set `EDGAR_USER_AGENT` to your application name and contact email. Set `NEWS_USER_AGENT` to customize the news client identification. Configure Alpaca paper credentials only if using the paper broker adapter. ORACLE requires provider credentials for model calls.

## Integration tests

Create a disposable database with a name ending in `_test`, separate from the local application database. Point `TRIDENT_TEST_DATABASE_URL` at it and run `uv run --no-sync pytest -q -m integration -k "not finbert_orders_sentiment"`. The database tests seed and clear tables there.

The separate FinBERT check requires the `sentiment` extra and model weights. Install it with `uv sync --locked --python 3.12 --extra dev --extra oracle --extra sentiment`, then run `uv run --no-sync pytest -q -m integration tests/test_sentiment_score.py`. The model may download on the first run. CI runs the non-integration suite.
