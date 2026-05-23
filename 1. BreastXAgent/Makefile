.PHONY: install install-dev test lint check

install:
	python3 -m pip install -r requirements.txt

install-dev:
	python3 -m pip install -e ".[dev]"

test:
	python3 -m pytest

lint:
	python3 -m ruff check src tests

check:
	python3 -m compileall src tests
	python3 -m pytest

