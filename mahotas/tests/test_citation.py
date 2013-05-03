import mahotas as mh
def test_citation():
    from sys import stdout
    t = mh.citation(False)
    assert len(stdout.getvalue()) == 0
    t2 = mh.citation(True)
    assert len(stdout.getvalue()) != 0
    assert t == t2
