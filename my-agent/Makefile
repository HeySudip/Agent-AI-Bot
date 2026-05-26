.PHONY: help install install-dev run test cov lint fmt fix clean docker docker-up

PYTHON ?= python3

help:
	@echo "Targets:"
	@echo "  install       install runtime dependencies"
	@echo "  install-dev   install runtime + dev dependencies"
	@echo "  run           run the Telegram bot"
	@echo "  test          run the test suite"
	@echo "  cov           run tests with coverage report"
	@echo "  lint          run ruff + mypy"
	@echo "  fmt           reformat with ruff"
	@echo "  fix           run ruff with --fix"
	@echo "  docker        build the Docker image"
	@echo "  docker-up     start with docker compose"
	@echo "  clean         remove caches and build artifacts"

install:
	$(PYTHON) -m pip install -r requirements.txt

install-dev: install
	$(PYTHON) -m pip install -e ".[dev,observability]"

run:
	$(PYTHON) bot.py

test:
	$(PYTHON) -m pytest tests/

cov:
	$(PYTHON) -m pytest --cov=. --cov-report=term-missing --cov-report=html tests/

lint:
	$(PYTHON) -m ruff check .
	$(PYTHON) -m mypy

fmt:
	$(PYTHON) -m ruff format .

fix:
	$(PYTHON) -m ruff check --fix .

docker:
	docker build -t agent-ai-bot:latest .

docker-up:
	docker compose up --build

clean:
	rm -rf __pycache__ .pytest_cache .ruff_cache .mypy_cache htmlcov .coverage
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type f -name '*.pyc' -delete
