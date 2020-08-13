#!/usr/bin/env bash


if test -e $HOME/miniconda/envs/condaenv; then
    echo "condaenv already exists"
else
    conda create  --quiet --yes -n condaenv python=${TRAVIS_PYTHON_VERSION} numpy=${NUMPY_VERSION}
    conda install --quiet --yes -n condaenv scipy matplotlib pillow pytest pip coveralls
fi

source activate condaenv
make debug
