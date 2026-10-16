def run(verbose=False):
    '''
    Run the mahotas test suite (this requires ``pytest``)

    This function is also available as ``mahotas.test()``.

    Parameters
    ----------
    verbose : bool, optional
        Whether to run pytest in verbose mode (default: False)

    Returns
    -------
    exit_code : int
        The pytest exit code (0 if all tests passed)
    '''
    from os import path
    import pytest
    args = [path.dirname(__file__)]
    if verbose:
        args.append('--verbose')
    return pytest.main(args)
