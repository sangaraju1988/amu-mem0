.PHONY: install test lint format bench

install:
	uv sync

test:
	MEM0_TELEMETRY=False uv run pytest -q

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff check --fix .
	uv run ruff format .

bench:
	MEM0_TELEMETRY=False uv run python bench/run_bench.py
