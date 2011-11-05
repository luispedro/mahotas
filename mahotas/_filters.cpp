/* Copyright (C) 2003-2005 Peter J. Verveer
 * Copyright (C) 2010-2011 Luis Pedro Coelho
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above
 *    copyright notice, this list of conditions and the following
 *    disclaimer in the documentation and/or other materials provided
 *    with the distribution.
 *
 * 3. The name of the author may not be used to endorse or promote
 *    products derived from this software without specific prior
 *    written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS
 * OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE
 * GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 * WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <cassert>
#include <memory>
#include "_filters.h"
#include "utils.hpp"

extern "C" {
    #include <Python.h>
    #include <numpy/ndarrayobject.h>
}

npy_intp fix_offset(const ExtendMode mode, npy_intp cc, const npy_intp len, const npy_intp border_flag_value) {
    /* apply boundary conditions, if necessary: */
    switch (mode) {
    case EXTEND_MIRROR:
        if (cc < 0) {
            if (len <= 1) {
                return 0;
            } else {
                int sz2 = 2 * len - 2;
                cc = sz2 * (int)(-cc / sz2) + cc;
                return cc <= 1 - len ? cc + sz2 : -cc;
            }
        } else if (cc >= len) {
            if (len <= 1) {
                return 0;
            } else {
                int sz2 = 2 * len - 2;
                cc -= sz2 * (int)(cc / sz2);
                if (cc >= len)
                    cc = sz2 - cc;
            }
        }
        return cc;

    case EXTEND_REFLECT:
        if (cc < 0) {
            if (len <= 1) {
                return 0;
            } else {
                int sz2 = 2 * len;
                if (cc < -sz2)
                    cc = sz2 * (int)(-cc / sz2) + cc;
                cc = cc < -len ? cc + sz2 : -cc - 1;
            }
        } else if (cc >= len) {
            if (len <= 1) {
                return 0;
            } else {
                int sz2 = 2 * len;
                cc -= sz2 * (int)(cc / sz2);
                if (cc >= len)
                    cc = sz2 - cc - 1;
            }
        }
        return cc;
    case EXTEND_WRAP:
        if (cc < 0) {
            if (len <= 1) {
                return 0;
            } else {
                int sz = len;
                cc += sz * (int)(-cc / sz);
                if (cc < 0)
                    cc += sz;
            }
        } else if (cc >= len) {
            if (len <= 1) {
                return 0;
            } else {
                int sz = len;
                cc -= sz * (int)(cc / sz);
            }
        }
        return cc;
    case EXTEND_NEAREST:
        if (cc < 0) {
            return 0;
        } else if (cc >= len) {
            return len - 1;
        }
        return cc;
    case EXTEND_CONSTANT:
        if (cc < 0 || cc >= len)
            return border_flag_value;
        return cc;
    }
    assert(false); // We should never get here
    return 0;
}

/* Calculate the offsets to the filter points, for all border regions and
     the interior of the array: */
int init_filter_offsets(PyArrayObject *array, bool *footprint,
         const npy_intp * const fshape, npy_intp* origins,
         const ExtendMode mode, npy_intp **offsets, npy_intp *border_flag_value,
         npy_intp **coordinate_offsets)
{
    npy_intp coordinates[NPY_MAXDIMS], position[NPY_MAXDIMS];
    npy_intp forigins[NPY_MAXDIMS];
    const int rank = array->nd;
    const npy_intp* const ashape = array->dimensions;
    const npy_intp* const astrides = array->strides;
    const npy_intp sizeof_element = PyArray_ITEMSIZE(array);

    /* calculate how many sets of offsets must be stored: */
    npy_intp offsets_size = 1;
    for(int ii = 0; ii < rank; ii++)
        offsets_size *= (ashape[ii] < fshape[ii] ? ashape[ii] : fshape[ii]);
    /* the size of the footprint array: */
    npy_intp filter_size = 1;
    for(int i = 0; i < rank; ++i) filter_size *= fshape[i];
    /* calculate the number of non-zero elements in the footprint: */
    npy_intp footprint_size = 0;
    if (footprint) {
        for(int i = 0; i < filter_size; ++i) footprint_size += footprint[i];
    } else {
        footprint_size = filter_size;
    }

    if (int(mode) < 0 || int(mode) > EXTEND_LAST) {
        throw PythonException(PyExc_RuntimeError, "boundary mode not supported");
    }
    try {
        *offsets = 0;
        if (coordinate_offsets) *coordinate_offsets = 0;
        *offsets = new npy_intp[offsets_size * footprint_size];
        if (coordinate_offsets) {
            *coordinate_offsets = new npy_intp[offsets_size * rank * footprint_size];
        }
    } catch (std::bad_alloc&) {
        if (*offsets) delete [] offsets;
        throw;
    }
    // from here on, we cannot fail anymore:

    for(int ii = 0; ii < rank; ii++) {
        forigins[ii] = fshape[ii]/2 + (origins ? *origins++ : 0);
    }

    npy_intp max_size = 0; // maximum ashape[i]
    npy_intp max_stride = 0; // maximum abs( astrides[i] )
    for(int ii = 0; ii < rank; ii++) {
        const npy_intp stride = astrides[ii] < 0 ? -astrides[ii] : astrides[ii];
        if (stride > max_stride)
            max_stride = stride;
        if (ashape[ii] > max_size)
            max_size = ashape[ii];

        /* coordinates for iterating over the kernel elements: */
        coordinates[ii] = 0;
        /* keep track of the kernel position: */
        position[ii] = 0;
    }


    /* the flag to indicate that we are outside the border must have a
         value that is larger than any possible offset: */
    *border_flag_value = max_size * max_stride + 1;
    /* calculate all possible offsets to elements in the filter kernel,
         for all regions in the array (interior and border regions): */

    npy_intp* po = *offsets;
    npy_intp* pc = coordinate_offsets ? *coordinate_offsets : 0;

    /* iterate over all regions: */
    for(int ll = 0; ll < offsets_size; ll++) {
        /* iterate over the elements in the footprint array: */
        for(int kk = 0; kk < filter_size; kk++) {
            npy_intp offset = 0;
            /* only calculate an offset if the footprint is 1: */
            if (!footprint || footprint[kk]) {
                /* find offsets along all axes: */
                for(int ii = 0; ii < rank; ii++) {
                    const npy_intp orgn = forigins[ii];
                    npy_intp cc = coordinates[ii] - orgn + position[ii];
                    cc = fix_offset(mode, cc, ashape[ii], *border_flag_value);

                    /* calculate offset along current axis: */
                    if (cc == *border_flag_value) {
                        /* just flag that we are outside the border */
                        offset = *border_flag_value;
                        if (coordinate_offsets)
                            pc[ii] = 0;
                        break;
                    } else {
                        /* use an offset that is possibly mapped from outside the border: */
                        cc -= position[ii];
                        offset += astrides[ii] * cc;
                        if (coordinate_offsets)
                            pc[ii] = cc;
                    }
                }
                if (offset != *border_flag_value) offset /= sizeof_element;
                /* store the offset */
                *po++ = offset;
                if (coordinate_offsets)
                    pc += rank;
            }
            /* next point in the filter: */
            for(int ii = rank - 1; ii >= 0; ii--) {
                if (coordinates[ii] < fshape[ii] - 1) {
                    coordinates[ii]++;
                    break;
                } else {
                    coordinates[ii] = 0;
                }
            }
        }

        /* move to the next array region: */
        for(int ii = rank - 1; ii >= 0; ii--) {
            const int orgn = forigins[ii];
            if (position[ii] == orgn) {
                position[ii] += ashape[ii] - fshape[ii] + 1;
                if (position[ii] <= orgn)
                    position[ii] = orgn + 1;
            } else {
                position[ii]++;
            }
            if (position[ii] < ashape[ii]) {
                break;
            } else {
                position[ii] = 0;
            }
        }
    }

    return footprint_size;
}

void init_filter_iterator(const int rank, const npy_intp *fshape,
                    const npy_intp filter_size, const npy_intp *ashape,
                    const npy_intp *origins,
                    npy_intp* strides, npy_intp* backstrides,
                    npy_intp* minbound, npy_intp* maxbound)
{
    /* calculate the strides, used to move the offsets pointer through
         the offsets table: */
    if (rank > 0) {
        strides[rank - 1] = filter_size;
        for(int ii = rank - 2; ii >= 0; ii--) {
            const npy_intp step = ashape[ii + 1] < fshape[ii + 1] ? ashape[ii + 1] : fshape[ii + 1];
            strides[ii] = strides[ii + 1] * step;
        }
    }
    for(int ii = 0; ii < rank; ii++) {
        const npy_intp step = ashape[ii] < fshape[ii] ? ashape[ii] : fshape[ii];
        const npy_intp orgn = fshape[ii]/2 + (origins ? *origins++ : 0);
        /* stride for stepping back to previous offsets: */
        backstrides[ii] = (step - 1) * strides[ii];
        /* initialize boundary extension sizes: */
        minbound[ii] = orgn;
        maxbound[ii] = ashape[ii] - fshape[ii] + orgn;
    }
}

