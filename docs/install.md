# Installation

## For users

Install the published package, [`astro-pest`](https://pypi.org/project/astro-pest/), from PyPI.

### Using pip

We recommend installing into a virtual environment rather than system-wide:

```bash
python3 -m venv .venv
source .venv/bin/activate

pip install astro-pest
```

### Using uv

If you have [uv](https://docs.astral.sh/uv/) installed, it manages the virtual environment for you:

```bash
uv venv
source .venv/bin/activate

uv pip install astro-pest
```

### Verify the installation

```bash
pest --help
```

This should print the command-line usage for the `pest` pipeline runner. You're now ready to follow the {doc}`usage guide <usage>` to run your first pipeline.

## For developers

Follow this path if you want to contribute to PEST, fix a bug, add a new extractor/transformation/loader, or run the test suite locally.

### Clone the repository

```bash
git clone https://github.com/HITS-AIN/PEST.git
cd PEST
```

### Set up the development environment

uv reads `pyproject.toml` and creates an isolated virtual environment with the exact locked dependency versions from `uv.lock`:

```bash
uv sync --extra dev
```

This installs PEST itself in editable mode plus the `dev` extras (`pytest`, `ruff`, `ipykernel`, ...), so any change you make to the source under `src/pest/` is picked up immediately.

You don't need to manually activate the virtual environment — prefix commands with `uv run` and uv takes care of it. If you prefer an activated shell:

```bash
source .venv/bin/activate
```

### Run the test suite

```bash
uv run --extra dev pytest
```

Run a single test file or test:

```bash
uv run --extra dev pytest tests/test_fits_converter.py
uv run --extra dev pytest tests/test_fits_converter.py::test_some_case
```

### Lint and format

PEST uses [ruff](https://docs.astral.sh/ruff/) for both linting and formatting.

```bash
uv run --extra dev ruff check --no-fix
uv run --extra dev ruff format --check
```

Run these before opening a pull request.

### Run a pipeline locally

Once the environment is set up, the `pest` CLI is available via `uv run`:

```bash
uv run pest pipelines/illustris_skirt.yaml
```

See the {doc}`usage guide <usage>` for details on writing pipeline configuration files.

### Build the documentation

The documentation (this site) is built with [Sphinx](https://www.sphinx-doc.org/) from the `docs/` directory using the `docs` extra:

```bash
uv sync --extra docs
uv run --extra docs sphinx-build -b html docs docs/_build/html
```

Open `docs/_build/html/index.html` in a browser to preview it. For live-reloading while editing:

```bash
uv run --extra docs sphinx-autobuild docs docs/_build/html
```
