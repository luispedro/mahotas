import numpy as np
from mahotas.io import error_imread, error_imsave
from os import path
import mahotas as mh
import pytest

filename = path.join(
            path.dirname(__file__),
            'data',
            'rgba.png')

def skip_on(etype):
    from functools import wraps
    def skip_on2(test):
        @wraps(test)
        def execute(*args, **kwargs):
            try:
                test(*args, **kwargs)
            except Exception as e:
                if isinstance(e, etype):
                    pytest.skip("Missing dependency")
                raise
        return execute
    return skip_on2


def test_error_imread():
    with pytest.raises(ImportError):
        error_imread(filename)

def test_error_imsave():
    with pytest.raises(ImportError):
        error_imsave('/tmp/test_mahotas.png', np.arange(16, dtype=np.uint8).reshape((4,4)))

@skip_on(IOError)
def test_as_grey():
    filename = path.join(
            path.dirname(__file__),
            '..',
            'demos',
            'data',
            'luispedro.jpg')
    im = mh.imread(filename, as_grey=1)
    assert im.ndim == 2


def test_pil():
    import mahotas as mh
    import numpy as np
    try:
        from mahotas.io import pil
    except ImportError:
        pytest.skip("Missing PIL. Skipping PIL-dependent tests")
    lena = mh.demos.load('lena')
    filename = path.join(
            path.dirname(__file__),
            '..',
            'demos',
            'data',
            'lena.jpg')
    assert np.all( pil.imread(filename) == lena )
    assert pil.imread(filename, as_grey=1).ndim == 2
