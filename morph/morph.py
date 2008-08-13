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
    Bc=get_structuring_elem(A,Bc)
    return _morph.dilate(A,Bc)

def erode(A,Bc=None):
    Bc=get_structuring_elem(A,Bc)
    return _morph.erode(A,Bc)
    
