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

## Running the app

After you activate `.venv` and finish the editable install, this repo can be used in two ways.

### CLI

The package installs the `annobox` command-line tool.

Show the available commands:

```bash
annobox --help
```

Common examples:

```bash
annobox infer --input tests/fixtures/us1/coco_minimal
annobox validate --input tests/fixtures/us1/coco_minimal
annobox convert --input tests/fixtures/us1/coco_minimal --output /tmp/label_master_out --src coco --dst yolo
```

### GUI

The Streamlit GUI entry point is [src/label_master/interfaces/gui/app.py](src/label_master/interfaces/gui/app.py).

Run it with:

```bash
python -m streamlit run src/label_master/interfaces/gui/app.py
```

The GUI defaults to the sample dataset at `tests/fixtures/us1/coco_minimal`, so you can launch it and try the workflow immediately after installation.

The repo also includes [/.streamlit/config.toml](.streamlit/config.toml) to force Streamlit to use polling-based file watching, which avoids Linux `inotify watch limit reached` errors on larger trees.

If you see `ModuleNotFoundError` for `typer` or `streamlit`, the virtual environment is not active yet or the install step did not complete.

### Per-image classification mode

The GUI tab **7. Classification** assigns whole-image class labels with single keystrokes. It works alongside bounding-box labels (the Step 2 preview boxes are drawn over the image) or on a plain folder of images with no bbox labels at all.

Class names and their keys come from a small config file. Put it in the dataset root as `classification.yaml` (also accepted: `classification.yml`, `classification.json`, `classification_config.yaml`), or point the tab at any path with the "Classification config" field. See [docs/classification_example.yaml](docs/classification_example.yaml):

```yaml
classes:
  - key: "1"
    name: helicopter
  - key: "2"
    name: airplane
  - key: "0"
    name: none
multi_label: false                          # optional; true lets keys toggle several classes per image
labels_file: classification_labels.json     # optional; relative to the dataset root
```

`classes: [helicopter, airplane, bird]` also works; keys are then assigned as `1`-`9`, `0`, `a`-`z`. A YOLO style `names:` list or `{id: name}` mapping is accepted too.

In the tab, a key legend shows which keystroke maps to which class and how many images carry each class. Press a class key to label the current image (it is saved immediately and, by default, the view advances to the next image), use Left/Right arrows or the buttons to move, Backspace/Delete to clear a label, and "Next unlabeled" to jump to the next image without a label. Labels are written to the `labels_file` JSON manifest:

```json
{
  "version": 1,
  "classes": ["helicopter", "airplane", "none"],
  "multi_label": false,
  "labels": {"images/frame_0001.jpg": ["helicopter"]}
}
```

Bounding-box label files are never modified by this mode.

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
