def run(verbose=False):
    from os import path
    import pytest
    args = [path.dirname(__file__)]
    if verbose:
        args.append('--verbose')
    return pytest.main(args)
