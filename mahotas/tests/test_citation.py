import mahotas as mh
def test_citation(capsys):
    t = mh.citation(False)
    captured = capsys.readouterr()
    assert len(captured.out) == 0
    t2 = mh.citation(True)
    captured = capsys.readouterr()
    assert len(captured.out) != 0
    assert t == t2

    ts = mh.citation(False, short=True)
    assert t != ts
