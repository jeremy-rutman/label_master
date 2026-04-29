# label_master

Tools for inferring, validating, converting, and importing bounding-box annotation datasets.

## Local setup

This project requires Python 3.11 or newer.

Check your interpreter version first:

```bash
python3 --version
```

If `python3.11` is available on your system, create and activate a virtual environment with:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
```

If `python3.11` is not installed but `python3` already points to Python 3.11+, use:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
```

## setup.py note

This repository uses [pyproject.toml](./pyproject.toml) as the source of truth for packaging metadata. The root-level `setup.py` is a minimal compatibility shim for older tooling that still expects it.

It is safe to run commands such as:

```bash
python3 setup.py --version
python3 setup.py egg_info
```

For normal installation and development setup, prefer:

```bash
python -m pip install -e '.[dev]'
```

Avoid using `python setup.py install` unless you specifically need legacy behavior.

## Common development commands

After activating the virtual environment, you can run:

```bash
pytest
ruff check .
mypy src
```

Or use the Makefile targets:

```bash
make install
make lint
make typecheck
make test
```
