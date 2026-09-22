# Repository Guidelines

## Project Structure & Module Organization

Core package code lives in `mahotas/`. High-level Python modules such as `morph.py`, `texture.py`, and `thresholding.py` sit alongside compiled extension sources like `_morph.cpp` and `_filters.cpp`. Feature-specific code is under `mahotas/features/`, I/O adapters are under `mahotas/io/`, demo scripts and sample data are in `mahotas/demos/`, and the test suite is in `mahotas/tests/`. Documentation sources live in `docs/source/`.

## Architecture

- **Python/C++ split:** each public `module.py` validates and normalizes arguments, then calls into its `_module.cpp` extension. Keep type checks and the `out=` convention on the Python side; use the helpers in `mahotas/internal.py` (`_get_output`, `_verify_is_integer_type`, `_verify_is_floatingpoint_type`, `_make_binary`, ...).
- **C++ pattern:** each extension function is a `py_foo` entry point that parses args via the Python C API, then dispatches on dtype with the `SAFE_SWITCH_ON_*_TYPES_OF(array)` + `HANDLE(type)` macros (`mahotas/numpypp/dispatch.hpp`) into a templated `foo<T>` that does the work. The template usually starts with `gil_release nogil;`. Array wrappers (`numpy::aligned_array<T>`, iterators) are in `mahotas/numpypp/array.hpp`.
- **Shared filter code:** `_filters.cpp`/`_filters.h` provide `filter_iterator` (derived from `scipy.ndimage`) for neighbourhood operations and border modes. They are linked into several extensions (`_convolve`, `_interpolate`, `_labeled`, `_morph`, `features/_texture`).
- **Build system:** meson-python (`meson.build`, `pyproject.toml`). `meson.build` lists **every** installed file explicitly: Python sources (including tests), package data (demo/test images), and extension sources. A new `.py`, data file, or `.cpp` must be added there too, or it will be missing from installs and wheels.
- **Public API:** functions are re-exported from `mahotas/__init__.py`, and `docs/source/api.rst` automodules them. The I/O backend (imread/FreeImage/PIL) is a soft dependency: `import mahotas` must work without one.

## Build, Test, and Development Commands

Use local editable installs for Python-side work:

```bash
pip install -e .[tests]
```

Build extension modules with the shipped `Makefile`. These targets rebuild the editable install with `--no-build-isolation`, so meson, meson-python, ninja, and numpy must be installed in the active environment:

```bash
make fast    # optimized release build (build/fast)
make debug   # release build with assertions (b_ndebug=false) and _GLIBCXX_DEBUG checked iterators (build/debug)
make clean   # remove build dirs and compiled .so files
```

Run the test suite with:

```bash
pytest -v
pytest mahotas/tests/test_morph.py            # single module
pytest mahotas/tests/test_morph.py::test_open  # single test
python -c "import mahotas as mh; mh.test()"
make tests   # make debug, then pytest -v
```

Build docs with `make docs` or `cd docs && make html`.

CI (`.github/workflows/`) tests Python 3.10–3.14 against a matrix of numpy 2.x versions. Wheels are built with cibuildwheel (config in `pyproject.toml`).

## Coding Style & Naming Conventions

Follow the existing style: 4-space indentation, snake_case for Python modules/functions, and `test_*.py` for tests. Keep Python wrappers and C++ extension code aligned when changing behavior. Prefer small, targeted changes over large refactors. There is no repository-local formatter or linter config, so match surrounding code and keep comments brief and technical. Every public function needs a complete numpydoc-style docstring.

## Testing Guidelines

Tests use `pytest`. Add regression tests in `mahotas/tests/` next to related functionality, and keep names descriptive, for example `test_thresholding.py` or `test_import()`. Every fixed bug gets a regression test, and new features need at least a smoke test. User input must never crash the interpreter: bad input should raise a Python exception. When touching compiled code, run the relevant test module first, then the full suite. Debug builds (`make debug`) are useful for catching C++ assertion failures before opening a PR.

## Commit & Pull Request Guidelines

Recent history uses short prefixed subjects such as `BUG`, `DOC`, `TST`, `BLD`, `RLS`, `MIN`, and `ENH`; follow that style or an equally clear imperative subject, for example `BUG Fix uint64 dispatch`. Keep commits focused. Pull requests should describe the behavioral change, note test coverage, and link related issues. Include screenshots only for documentation or demo-output changes.

## Contributor Notes

If you modify internals, read `docs/source/internals.rst` and `docs/source/principles.rst` first. Debug builds are configured through meson options (`-Db_ndebug=false`, `-Dglibcpp_debug=true`); see the `debug` target in the `Makefile`. The old `DEBUG=1`/`DEBUG=2` environment variables are no longer used.
