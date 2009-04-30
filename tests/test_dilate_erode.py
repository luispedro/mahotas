import numpy
import morph.morph

def test_dilate_erode():
    A=numpy.zeros((100,100))
    Bc=numpy.zeros((3,3))
    Bc[2,1]=1
    Bc[1]=1
    Bc[0,1]=1
    A[30,30]=1
    A=(A!=0)
    orig=A.copy()
    Bc=(Bc!=0)
    for i in xrange(12):
        A=morph.morph.dilate(A!=0,Bc != 0)
    for i in xrange(12):
        A=morph.morph.erode(A!=0,Bc != 0)

