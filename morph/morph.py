'''
PyMorph++ 

This is a companion package to pymorph which includes a C++ implementation
of some of the algorithms for speed.

The license on this package is GPL (as opposed to pymorph which is BSD).
'''
import numpy
import numpy as np
try:
    import _morph
except:
    print '''Import of _morph implementation module failed.
Please check your installation.'''

def _verify_types(A,allowed_types,function):
    if A.dtype not in allowed_types:
        raise RuntimeException('%s: Type %s not allowed for this function.' % (function,A.dtype))

def _verify_is_integer_type(A,function):
    int_types=[
                numpy.bool,
                numpy.uint8,
                numpy.int8,
                numpy.uint16,
                numpy.int16,
                numpy.uint32,
                numpy.int32
                ]
    _verify_types(A,int_types,function)
    
def _verify_is_bool(A,function):
    _verify_types(A,[numpy.bool],function)

__all__ = [
        'get_structuring_elem',
        'dilate',
        'erode',
        'cwatershed',
        'close_holes',
        ]
    
def get_structuring_elem(A,Bc):
    '''Bc_out = get_structuring_elem(A,Bc_in)

    Bc_in can be either:
        * None: Then Bc_in is taken to be 1

        * an integer: There are two associated semantics:
            - connectivity:
                Bc[y,x] = [[ is |y - 1| + |x - 1| <= Bc_i ]]

            - count:
                Bc.sum() == Bc_i
            This is the more traditional meaning (when one writes that "4-connected", this is what
            one has in mind).

          Fortunately, the value itself allows one to distinguish between the two semantics and, if
          used correctly, no ambiguity should ever occur.

        * An array. This should be of the same nr. of dimensions as A and will be passed through if of the
            right type. Otherwise, it will be cast.

    Bc_out will be of the same type as A
    '''
    if len(A.shape) != 2:
        raise NotImplementedError('morph.get_structuring_elem: Sorry, only 2D morphology for now.')
    if Bc is None:
        Bc=numpy.zeros((3,3),A.dtype)
        Bc[0,1]=1
        Bc[1]=1
        Bc[2,1]=1
        return Bc
    elif type(Bc) is int:
        if Bc == 4:
            return get_structuring_elem(A,None)
        elif Bc == 8:
            return numpy.ones((3,3),A.dtype)
        else:
            raise RuntimeError('morph.get_structuring_elem: Forbidden argument %s' % Bc)
    else:
        if len(A.shape) != len(Bc.shape):
            raise ValueError('morph.get_structuring_elem: Bc does not have the correct number of dimensions.')
        Bc=numpy.asanyarray(Bc,A.dtype)
        if not Bc.flags['C_CONTIGUOUS']:
            return Bc.copy()
        return Bc

def dilate(A,Bc=None):
    _verify_is_bool(A,'dilate')
    _verify_is_integer_type(A,'dilate')
    Bc=get_structuring_elem(A,Bc)
    return _morph.dilate(A,Bc)

def erode(A,Bc=None):
    _verify_is_bool(A,'dilate')
    _verify_is_integer_type(A,'erode')
    Bc=get_structuring_elem(A,Bc)
    return _morph.erode(A,Bc)
    
def cwatershed(surface, markers, Bc=None, return_lines=False):
    '''
    W = cwatershed(surface, markers, Bc=None, return_lines=False)
    W,WL = cwatershed(surface, markers, Bc=None, return_lines=True)

    Seeded Watershed
    
    Parameters
    ----------
        * surface: image
        * markers: initial markers (must be a labeled image)
        * Bc: structuring element (default: 3x3 cross)
        * return_lines: whether to return separating lines
            (in addition to regions)
    '''
    _verify_is_integer_type(surface, 'cwatershed')
    _verify_is_integer_type(markers, 'cwatershed')
    if surface.shape != markers.shape:
        raise ValueError('morph.cwatershed: Markers array should have the same shape as value array.')
    if markers.dtype != surface.dtype:
        markers = markers.astype(surface.dtype)
    Bc = get_structuring_elem(surface, Bc)
    return _morph.cwatershed(surface, markers, Bc, bool(return_lines))

def close_holes(ref, Bc=None):
    '''
    closed = close_holes(ref, Bc=None):

    Close Holes

    Parameters
    ----------
        * ref: Reference image.
        * Bc: structuring element (default: 3x3 cross)
    '''
    if ref.dtype != np.bool:
        if ((ref== 0)|(ref==1)).sum() != ref.size:
            raise ValueError,'morph.close_holes: passed array is not boolean.'
        ref = ref.astype(bool)
    if not ref.flags['C_CONTIGUOUS']:
        ref = ref.copy()
    Bc = get_structuring_elem(ref, Bc)
    return _morph.close_holes(ref, Bc)
