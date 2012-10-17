# This is from Pymorph.
# Partial copy
import numpy as np

def add4dilate(f, c):
    from numpy import asarray, minimum, maximum, float64

    if not c:
        return f
    y = asarray(f,float64) + c
    k1,k2 = limits(f)
    y = ((f==k1) * k1) + ((f!=k1) * y)
    y = maximum(minimum(y,k2),k1)
    return y.astype(f.dtype)
def mat2set(A):
    from numpy import take, ravel, nonzero, transpose, newaxis

    if len(A.shape) == 1: A = A[newaxis,:]
    offsets = nonzero(ravel(A) - limits(A)[0])[0]
    if len(offsets) == 0: return ([],[])
    h,w = A.shape
    x = [0,1]
    x[0] = offsets//w - (h-1)//2
    x[1] = offsets%w - (w-1)//2
    x = transpose(x)
    return x,take(ravel(A),offsets)

def dilate(f, B):
    from numpy import maximum, newaxis, ones, int32
    h,w = f.shape
    x,v = mat2set(B)
    mh,mw = max(abs(x)[:,0]),max(abs(x)[:,1])
    y = (ones((h+2*mh,w+2*mw),int32) * limits(f)[0]).astype(f.dtype)
    for i in range(x.shape[0]):
        if v[i] > -2147483647:
            y[mh+x[i,0]:mh+x[i,0]+h, mw+x[i,1]:mw+x[i,1]+w] = maximum(
                y[mh+x[i,0]:mh+x[i,0]+h, mw+x[i,1]:mw+x[i,1]+w], add4dilate(f,v[i]))
    y = y[mh:mh+h, mw:mw+w]
    return y

def sereflect(Bi):
    return Bi[::-1, ::-1]
def limits(f):
    from numpy import array, bool, uint8, uint16, int32, int64
    code = f.dtype
    if code == bool: return 0,1
    if code == uint8: return 0,255
    if code == uint16: return 0,65535
    if code == int32: return -2147483647,2147483647
    if code == int64: return -2**63,2**63-1

    raise ValueError('pymorph.limits: does not accept this typecode: %s' % code)

def neg(f):
    y = limits(f)[0] + limits(f)[1] - f
    return y.astype(f.dtype)

def erode(f, b):
    return neg(dilate(neg(f),sereflect(b)))


def cdilate(f, g, Bc, n=1):
    f = np.minimum(f,g)
    for i in range(n):
        prev = f
        f = np.minimum(dilate(f, Bc), g)
        if np.all(f == prev): break
    return f


