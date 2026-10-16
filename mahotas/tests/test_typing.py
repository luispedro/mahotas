import ast
import importlib
import inspect
from importlib import resources
from pathlib import Path
from typing import get_overloads, get_type_hints

import pytest


MODULES = [
    'mahotas', 'mahotas._filters', 'mahotas.internal', 'mahotas.bbox',
    'mahotas.center_of_mass', 'mahotas.colors', 'mahotas.convolve',
    'mahotas.demos', 'mahotas.distance', 'mahotas.edge', 'mahotas.euler',
    'mahotas.features.lbp', 'mahotas.features.moments', 'mahotas.features.shape',
    'mahotas.features.surf', 'mahotas.features.tas', 'mahotas.features.texture',
    'mahotas.features.zernike', 'mahotas.features_cli', 'mahotas.histogram',
    'mahotas.interpolate', 'mahotas.io', 'mahotas.labeled', 'mahotas.morph',
    'mahotas.polygon', 'mahotas.resize', 'mahotas.segmentation', 'mahotas.stretch',
    'mahotas.tests', 'mahotas.thin', 'mahotas.thresholding',
]


@pytest.mark.parametrize('module_name', MODULES)
def test_annotations_resolve(module_name: str) -> None:
    module = importlib.import_module(module_name)
    for name, function in inspect.getmembers(module, inspect.isfunction):
        # Python 3.14+ generates __annotate__ for module-level annotations (PEP 649)
        if function.__module__ != module_name or name == '__annotate__':
            continue
        for signature in [function, *get_overloads(function)]:
            hints = get_type_hints(signature)
            assert 'return' in hints
            assert set(inspect.signature(signature).parameters) <= hints.keys()


@pytest.mark.parametrize('module_name', MODULES)
def test_annotations_python311(module_name: str) -> None:
    module = importlib.import_module(module_name)
    tree = ast.parse(Path(module.__file__).read_text(), feature_version=(3, 11))
    for function in ast.walk(tree):
        if not isinstance(function, ast.FunctionDef):
            continue
        assert function.returns is not None, function.name
        arguments = [*function.args.posonlyargs, *function.args.args, *function.args.kwonlyargs]
        arguments.extend(a for a in (function.args.vararg, function.args.kwarg) if a is not None)
        for argument in arguments:
            if argument.arg not in ('self', 'cls'):
                assert argument.annotation is not None, (function.name, argument.arg)


def test_typing_package_data() -> None:
    package = resources.files('mahotas')
    assert package.joinpath('py.typed').is_file()
    assert package.joinpath('io', '__init__.pyi').is_file()


def test_optional_backend_annotations() -> None:
    for module_name in ('mahotas.io.pil', 'mahotas.io.freeimage'):
        try:
            importlib.import_module(module_name)
        except (ImportError, OSError):
            continue
        test_annotations_resolve(module_name)
        test_annotations_python311(module_name)
