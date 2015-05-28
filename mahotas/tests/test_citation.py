import mahotas as mh
def test_citation():
    from sys import stdout
    t = mh.citation(False)
    assert len(stdout.getvalue()) == 0
    t2 = mh.citation(True)
    assert len(stdout.getvalue()) != 0
    assert t == t2

    ts = mh.citation(False, short=True)
    assert t != ts
