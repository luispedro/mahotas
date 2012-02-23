_error_message = '''
mahotas.%%s depends on imread or freeimage, neither of which could be found.

Everything else will work, though, so this error is only triggered when you
attempt to use these optional functions.

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
    except:
        from .freeimage import imread, imsave
except OSError, e:
    _error_message %= e
    imread = error_imread
    imsave = error_imsave

