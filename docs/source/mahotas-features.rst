===================
mahotas-features.py
===================

.. versionadded:: 1.4.0
    The mahotas-features.py script appeared in version 1.4.0 (July 2015)

With the installation of mahotas, a script called ``mahotas-features.py`` is
installed, which can be used to compute features from a set of files.

Usage
-----

You call the script with a set of flags specifying which features you want to
compute, followed by a list of files. For example::

    $ mahotas-features.py --haralick --lbp image-file1.tiff image-file2.tiff

This will output to the file ``features.tsv`` (this default can be changed with
the ``--output`` option)

Full Usage Information
----------------------

You can obtain help on all the options by running ``mahotas-features.py
--help``::

    If you use mahotas in a scientific publication, please cite
        Coelho, LP (2013). http://dx.doi.org/10.5334/jors.ac


    usage: mahotas-features.py [-h] [--output OUTPUT] [--clobber]
                               [--convert-to-bw CONVERT_TO_BW] [--no-color]
                               [--haralick] [--lbp] [--lbp-radius LBP_RADIUS]
                               [--lbp-points LBP_POINTS]
                               input_file_name [input_file_name ...]

    Compute features using mahotas

    positional arguments:
      input_file_name       Image files names

    optional arguments:
      -h, --help            show this help message and exit
      --output OUTPUT       Output file for feature files
      --clobber             Overwrite output file (if it exists)
      --convert-to-bw CONVERT_TO_BW
                            Convert color images to greyscale. Acceptable values:
                            no: raises an error (default) max: use max projection
                            yes: use rgb2gray
      --no-color            Do not print in color (for error and warning messages)
      --haralick            Compute Haralick features
      --lbp                 Compute LBP (linear binary patterns) features
      --lbp-radius LBP_RADIUS
                            Radius to use for LBP features
      --lbp-points LBP_POINTS
                            Nr of points to use for LBP features
