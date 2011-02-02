import numpy
import numpy as np
try:
    import _morph
except:
    print '''Import of _morph implementation module failed.
Please check your installation.'''

def _verify_types(A,allowed_types,function):
    if A.dtype not in allowed_types:
        raise TypeError('%s: Type %s not allowed for this function.' % (function,A.dtype))

def _verify_is_integer_type(A,function):
    int_types=[
                np.bool,
                np.uint8,
                np.int8,
                np.uint16,
                np.int16,
                np.uint32,
                np.int32,
                np.int64,
                np.uint64,
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
        'hitmiss',
        ]
    
def get_structuring_elem(A,Bc):
    '''
    Bc_out = get_structuring_elem(A, Bc)

    Retrieve appropriate structuring element

    Parameters
    ----------
    A : array which will be operated on
    Bc : can be either:
        :None: Then Bc is taken to be 1
        :An integer: There are two associated semantics:
            connectivity
              ``Bc[y,x] = [[ is |y - 1| + |x - 1| <= Bc_i ]]``
            count
              ``Bc.sum() == Bc_i``
              This is the more traditional meaning (when one writes that "4-connected", this is what
              one has in mind).

          Fortunately, the value itself allows one to distinguish between the two semantics and, if
          used correctly, no ambiguity should ever occur.
        :An array: This should be of the same nr. of dimensions as A and will be passed through if of the
            right type. Otherwise, it will be cast.

    Returns
    -------
    Bc_out : will be of the same type as A
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
            raise ValueError('morph.get_structuring_elem: Forbidden argument %s' % Bc)
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
    surface : image
    markers : initial markers (must be a labeled image)
    Bc : structuring element (default: 3x3 cross)
    return_lines : whether to return separating lines
                   (in addition to regions)

    Returns
    -------
    W : Regions image (i.e., W[i,j] == region for pixel (i,j))
    WL : Lines image (`if return_lines==True`)
    '''
    _verify_is_integer_type(surface, 'cwatershed')
    _verify_is_integer_type(markers, 'cwatershed')
    if surface.shape != markers.shape:
        raise ValueError('morph.cwatershed: Markers array should have the same shape as value array.')
    if markers.dtype != surface.dtype:
        markers = markers.astype(surface.dtype)
    Bc = get_structuring_elem(surface, Bc)
    return _morph.cwatershed(surface, markers, Bc, bool(return_lines))

def hitmiss(input, Bc, output=None):
    '''
    output = hitmiss(input, Bc, output=np.zeros_like(input))

    Hit & Miss Transform

    Parameters
    ----------
    input : input ndarray
    Bc : ndarray
        hit & miss template, values must be one of (0, 1, 2)
    output : output array

    Returns
    -------
    output : ndarray
    '''
    _verify_is_integer_type(input, 'hitmiss')
    _verify_is_integer_type(Bc, 'hitmiss')
    if input.dtype != Bc.dtype:
        if input.dtype == np.bool_:
            input = input.view(np.uint8)
            if Bc.dtype == np.bool_:
                Bc = Bc.view(np.uint8)
            else:
                Bc = Bc.astype(np.uint8)
        else:
            Bc = Bc.astype(np.dtype)
    if output is None:
        output = np.empty_like(input)
    else:
        if output.shape != input.shape:
            raise ValueError('mahotas.hitmiss: output must be of same shape as output')
        if output.dtype != input.dtype:
            if output.dtype == np.bool_ and input.dtype == np.uint8:
                output = output.view(np.uint8)
            else:
                raise TypeError('mahotas.hitmiss: output must be of same type as input')
    return _morph.hitmiss(input, Bc, output)

def close_holes(ref, Bc=None):
    '''
    closed = close_holes(ref, Bc=None):

    Close Holes

    Parameters
    ----------
    ref : Reference image.
    Bc : structuring element (default: 3x3 cross)

    Returns
    -------
    closed : superset of `ref` (i.e. with closed holes)
    '''
    if ref.dtype != np.bool:
        if ((ref== 0)|(ref==1)).sum() != ref.size:
            raise ValueError,'morph.close_holes: passed array is not boolean.'
        ref = ref.astype(bool)
    if not ref.flags['C_CONTIGUOUS']:
        ref = ref.copy()
    Bc = get_structuring_elem(ref, Bc)
    return _morph.close_holes(ref, Bc)


def majority_filter(img, N=3, output=None):
    '''
    filtered = majority_filter(img, N=3, output={np.empty(img.shape, np.bool)})

    Majority filter

    filtered[y,x] is positive if the majority of pixels in the squared of size
    `N` centred on (y,x) are positive.

    Parameters
    ----------
      img : input img (currently only 2-D images accepted)
      N : size of filter (must be odd integer)
      output : used for output. Must be Boolean ndarray of same size as `img`

    Returns
    -------
      filtered : boolean image of same size as img.
                 output if passed
    '''
    if img.dtype != np.bool_:
        img = img.astype(bool)
    if output is None or output.dtype != np.bool_ or output.shape != img.shape:
        output = np.empty(img.shape, np.bool_)
    if N <= 1:
        raise ValueError('mahotas.majority_filter: filter size must be positive')
    if not N&1:
        import warnings
        warnings.warn('mahotas.majority_filter: size argument must be odd.')
        N += 1
    return _morph.majority_filter(img, N, output)

