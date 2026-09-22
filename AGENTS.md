# Repository Guidelines

## Project Structure & Module Organization

Core package code lives in `mahotas/`. High-level Python modules such as `morph.py`, `texture.py`, and `thresholding.py` sit alongside compiled extension sources like `_morph.cpp` and `_filters.cpp`. Feature-specific code is under `mahotas/features/`, I/O adapters are under `mahotas/io/`, demo scripts and sample data are in `mahotas/demos/`, and the test suite is in `mahotas/tests/`. Documentation sources live in `docs/source/`.

## Build, Test, and Development Commands

Use local editable installs for Python-side work:

```bash
pip install -e .[tests]
```

Build extension modules with the shipped `Makefile`:

```bash
make fast    # normal build into the working tree
make debug   # DEBUG=2 build with extra runtime checks
make clean   # remove build artifacts and compiled .so files
```

Run the test suite with:

```bash
pytest -v
python -c "import mahotas as mh; mh.test()"
make tests
```

Build docs with `make docs` or `cd docs && make html`.

## Coding Style & Naming Conventions

Follow the existing style: 4-space indentation, snake_case for Python modules/functions, and `test_*.py` for tests. Keep Python wrappers and C++ extension code aligned when changing behavior. Prefer small, targeted changes over large refactors. There is no repository-local formatter or linter config, so match surrounding code and keep comments brief and technical.

## Testing Guidelines

Tests use `pytest`. Add regression tests in `mahotas/tests/` next to related functionality, and keep names descriptive, for example `test_thresholding.py` or `test_import()`. When touching compiled code, run the relevant test module first, then the full suite. Debug builds (`make debug`) are useful for catching C++ assertion failures before opening a PR.

## Commit & Pull Request Guidelines

Recent history uses short prefixed subjects such as `BUG`, `DOC`, `TST`, `RLS`, `MIN`, and `ENH`; follow that style or an equally clear imperative subject, for example `BUG Fix uint64 dispatch`. Keep commits focused. Pull requests should describe the behavioral change, note test coverage, and link related issues. Include screenshots only for documentation or demo-output changes.

## Contributor Notes

If you modify internals, read `docs/source/internals.rst` and `docs/source/principles.rst` first. For build debugging, set `DEBUG=1` or `DEBUG=2`; `DEBUG=2` is slower but adds the strongest runtime checks.
