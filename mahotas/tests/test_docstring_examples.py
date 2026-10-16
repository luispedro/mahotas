import doctest
import importlib
import pytest

# Modules whose docstrings contain doctest examples (``>>>``)
_modules = [
    'mahotas.bbox',
    'mahotas.distance',
    'mahotas.labeled',
    'mahotas.morph',
    'mahotas.thresholding',
    'mahotas.features.texture',
]

@pytest.mark.parametrize('modname', _modules)
def test_docstring_examples(modname):
    mod = importlib.import_module(modname)
    failures, tried = doctest.testmod(mod, optionflags=doctest.NORMALIZE_WHITESPACE)
    assert tried > 0
    assert failures == 0
