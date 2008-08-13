import numpy
try:
    import _morph
except:
    print '''Import of _morph implementation module failed.
Please check your installation.'''

def _verify_types(A,allowed_types,function):
    if A.dtype not in allowed_types:
        raise RunTimeException('%s: Type %s not allowed for this function.' % (function,A.dtype))
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
def get_structuring_elem(A,Bc):
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
            raise RunTimeException('morph.get_structuring_elem: Forbidden argument %s' % Bc)
    else:
        return numpy.asanyarray(Bc,A.dtype)

def dilate(A,Bc=None):
    _verify_is_integer_type(A,'dilate')
    Bc=get_structuring_elem(A,Bc)
    return _morph.dilate(A,Bc)

def erode(A,Bc=None):
    _verify_is_integer_type(A,'erode')
    Bc=get_structuring_elem(A,Bc)
    return _morph.erode(A,Bc)
    
