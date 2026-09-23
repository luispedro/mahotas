'''
Image input/output

This module provides ``imread`` and ``imsave``, which are also re-exported as
``mahotas.imread`` and ``mahotas.imsave``. The actual work is delegated to one
of the following backends (the first one available is used):

1. the `imread <https://imread.readthedocs.io/>`__ package
2. `Pillow <https://python-pillow.org/>`__ (``mahotas.io.pil``)
3. FreeImage, through ctypes (``mahotas.io.freeimage``; deprecated)

None of these is a hard dependency: ``import mahotas`` works without any of
them. If none is available, ``imread`` and ``imsave`` are replaced by
functions which raise ``ImportError`` with installation instructions.
'''

_error_message = '''
mahotas.%%s depends on one of (in order of preference):

1. imread
2. pillow (PIL)
3. freeimage (deprecated)

None of which could be found!

Everything else will work, though, so this error is only triggered when you
attempt to use these optional functions.

The easiest solution is to install either imread or pillow, e.g.::

    pip install imread

or::

    pip install pillow

Both are also available from conda-forge::

    conda install -c conda-forge imread

Full error was: %s'''
def error_imread(*args, **kwargs):
    '''
    Placeholder for ``imread`` when no I/O backend is available

    Raises
    ------
    ImportError
        Always, with a message explaining how to install a backend
    '''
    raise ImportError(_error_message % 'imread')
def error_imsave(*args, **kwargs):
    '''
    Placeholder for ``imsave`` when no I/O backend is available

    Raises
    ------
    ImportError
        Always, with a message explaining how to install a backend
    '''
    raise ImportError(_error_message % 'imsave')

__all__ = [
    'imread',
    'imsave',
    ]
try:
    try:
        from imread import imread, imsave
    except ImportError: # pragma: no cover
        try:
            from .pil import imread, imsave
        except ImportError:
            from .freeimage import imread, imsave
            from os import environ
            if 'MAHOTAS_NO_FREEIMAGE_DEPRECATION' not in environ:
                import warnings
                warnings.warn('mahotas.freeimage is deprecated. Please install the `imread` package')
                warnings.warn('To suppress this warning, set the environment variable MAHOTAS_NO_FREEIMAGE_DEPRECATION')
# Importing freeimage can throw both ImportError and OSError, so check for both
except (OSError, ImportError): # pragma: no cover
    import sys
    _,e,_ = sys.exc_info()
    _error_message %= e
    imread = error_imread
    imsave = error_imsave

