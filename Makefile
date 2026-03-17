.PHONY: install lint typecheck test cov format

install:
	python3 -m pip install -e .[dev]

lint:
	ruff check .

typecheck:
	mypy src

test:
	pytest

cov:
	pytest --cov=src/label_master --cov-report=term-missing

format:
	ruff format .
