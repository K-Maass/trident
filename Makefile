.PHONY: up down db-init demo test lint

up:
	docker compose up -d --wait

down:
	docker compose down

db-init:
	docker compose exec -T db psql -v ON_ERROR_STOP=1 -U trident -d trident < db/schema.sql

demo:
	uv run --no-sync python scripts/demo.py

test:
	uv run --no-sync pytest -q -m "not integration"

lint:
	uv run --no-sync ruff format --check .
	uv run --no-sync ruff check .
