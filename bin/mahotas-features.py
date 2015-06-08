#!/usr/bin/env python

import sys
import numpy as np
import mahotas as mh
import argparse

RED = '\033[91m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
LIGHT_PURPLE = '\033[94m'
PURPLE = '\033[95m'
END = '\033[0m'

def print_error(text, color=True):
    '''Prints error message

    Arguments
    ---------
    text : str
        Error message
    color : bool, optional
        Whether to print in colour.
    '''
    if color and sys.stderr.isatty():
        sys.stderr.write("{}ERROR: {}{}\n".format(RED, text, END))
    else:
        sys.stderr.write("ERROR: {}\n".format(text))


def read_bw(fname, options):
    '''Read image `fname` as greyscale

    Parameters
    ----------
    fname : str, file-name
    options : argparse result

    Returns
    -------
    image : ndarray
        Two dimensional ndarray
    '''
    im = mh.imread(fname)
    if im.ndim == 2:
        return im
    print_error("{} is not a greyscale image".format(fname), not options.no_color)
    sys.exit(1)

def main():
    sys.stderr.write(mh.citation(print_out=False, short=True))
    sys.stderr.write('\n\n')
    parser = argparse.ArgumentParser(
            description='Compute features using mahotas')
    parser.add_argument(
                    'fnames', metavar='input_file_name', nargs='+', type=str,
                            help='Image files names')
    parser.add_argument(
                    '--output', default=sys.stdout, type=argparse.FileType('w'),
                            help='Output file for feature files')
    parser.add_argument(
                    '--no-color', default=False, action='store_true',
                            help='Do not print in color (for error and warning messages)')
    parser.add_argument(
                    '--haralick', default=False, action='store_true',
                            help='Compute Haralick features')
    args = parser.parse_args()
    if not args.haralick:
        sys.stderr.write('''\
No features selected. Doing nothing.

For example, use --haralick switch to compute Haralick features\n''')
        sys.exit(1)

    features = []
    colnames = []
    first = True
    for fname in args.fnames:
        cur = []
        im = read_bw(fname, args)
        if args.haralick:
            har = mh.features.haralick(im, return_mean_ptp=True)
            cur.append(har)
            if first:
                colnames.extend(mh.features.texture.haralick_labels[:-1])
                colnames.extend(["ptp:{}".format(ell) for ell in mh.features.texture.haralick_labels[:-1]])

        features.append(np.concatenate(cur))
        first = False

    features = np.array(features)
    try:
        import pandas as pd
        features = pd.DataFrame(features, index=args.fnames, columns=colnames)
        features.to_csv(args.output, sep='\t')
    except ImportError:
        np.savetxt(args.output, features)

if __name__ == '__main__':
    main()
