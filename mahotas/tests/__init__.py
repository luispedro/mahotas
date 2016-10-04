def run(verbose=False):
    import nose
    from os import path
    currentdir = path.dirname(__file__)
    updir = path.join(currentdir, '..')
    argv = ['', '--exe', '-w', updir]
    if verbose:
        argv.append('--verbose')
    return nose.run('mahotas', argv=argv)
