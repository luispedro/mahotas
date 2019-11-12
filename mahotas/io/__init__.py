
_error_message = '''
mahotas.%%s depends on one of (in order of preference):

1. imread
2. freeimage
3. pillow (PIL)

None of which could be found!

Everything else will work, though, so this error is only triggered when you
attempt to use these optional functions.

To install imread:

On **Ubuntu**, run the following commands::

    sudo apt-get install libpng12-dev libtiff4-dev libwebp-dev python-pip python-dev g++
    sudo pip install imread

On **Mac OS**, if using ``port``, run the following commands::

    sudo port install libpng tiff webp
    sudo pip install imread

On **Windows**, use Christoph Gohlke's packages. See:

http://www.lfd.uci.edu/~gohlke/pythonlibs/#imread




To install FreeImage:

You need to have the freeimage installed for imread/imsave (everything else
will work, though, so this error is only triggered when you attempt to use
these optional functions). Freeimage is not a Python package, but a regular
package.

Under Linux, look for a package called `freeimage` in your distribution (it is
actually called `libfreeimage3` in debian/ubuntu, for example).

Under Windows, consider using the third-party mahotas packages at
http://www.lfd.uci.edu/~gohlke/pythonlibs/ (kindly maintained by Christoph
Gohlke), which already package freeimage.

Full error was: %s'''
def error_imread(*args, **kwargs):
    raise ImportError(_error_message % 'imread')
def error_imsave(*args, **kwargs):
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
# Importing freeimage can throw both ImportError and OSError, so check for both
except (OSError, ImportError): # pragma: no cover
    import sys
    _,e,_ = sys.exc_info()
    _error_message %= e
    imread = error_imread
    imsave = error_imsave

