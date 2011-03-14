def run():
    import nose
    from os import path
    currentdir = path.dirname(__file__)
    updir = path.join(currentdir, '..')
    nose.run('mahotas', argv=['', '--exe', '-w', updir])
