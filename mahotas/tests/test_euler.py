import numpy as np
from mahotas.euler import euler, _euler_lookup4, _euler_lookup8
from nose.tools import raises

def test_lookup():
    Q1 = [np.array(q, np.bool) for q in [[0,0],[1,0]], [[0,0],[0,1]], [[0,1],[0,0]], [[1,0],[0,0]] ]
    Q2 =  [(~q) for q in Q1]
    Q3 = [np.array(q, np.bool) for q in [[0,1],[1,0]], [[1,0],[0,1]] ]

    def _value(q, lookup):
        q = q.ravel()
        value = np.dot(q, (1,2,4,8))
        return lookup[value]

    for q in Q1:
        assert _value(q, _euler_lookup8) == .25
        assert _value(q, _euler_lookup4) == .25
    for q in Q2:
        assert _value(q, _euler_lookup8) == -.25
        assert _value(q, _euler_lookup4) == -.25
    for q in Q3:
        assert _value(q, _euler_lookup8) == -.5
        assert _value(q, _euler_lookup4) ==  .5

def test_euler():
    f = np.zeros((16,16), np.bool)
    f[4:8,4:8] = 1
    assert euler(f) == 1
    assert euler(f, 4) == 1

    f[6:7,5:7] = 0

    assert euler(f) == 0
    assert euler(f, 4) == 0

@raises(ValueError)
def test_euler7():
    f = np.arange(100)
    f = (f % 5) == 1
    f = f.reshape((10,10))
    euler(f, 7)

