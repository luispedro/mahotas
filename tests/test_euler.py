from mahotas.euler import euler, _euler_lookup
import numpy as np

def test_lookup():
    Q1 = [np.array(q, np.bool) for q in [[0,0],[1,0]], [[0,0],[0,1]], [[0,1],[0,0]], [[1,0],[0,0]] ]
    Q2 =  [(~q) for q in Q1]
    Q3 = [np.array(q, np.bool) for q in [[0,1],[1,0]], [[1,0],[0,1]] ]
    
    def _value(q):
        q = q.ravel()
        value = np.dot(q, (1,2,4,8))
        return _euler_lookup[value]

    for q in Q1:
        assert _value(q) == .25
    for q in Q2:
        assert _value(q) == -.25
    for q in Q3:
        assert _value(q) == .5

def test_euler():
    f = np.zeros((16,16), np.bool)
    f[4:8,4:8] = 1
    assert euler(f) == 1

    f[6:7,5:7] = 0

    assert euler(f) == 0

