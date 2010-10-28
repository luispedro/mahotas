/* Copyright (C) 2003-2005 Peter J. Verveer
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

#include "_filters.h"

extern "C" {
    #include <Python.h>
    #include <numpy/ndarrayobject.h>
}

namespace {

/* Calculate the offsets to the filter points, for all border regions and
     the interior of the array: */
int NI_InitFilterOffsets(PyArrayObject *array, bool *footprint,
         npy_intp *filter_shape, npy_intp* origins,
         NI_ExtendMode mode, npy_intp **offsets, npy_intp *border_flag_value,
         npy_intp **coordinate_offsets)
{
    int rank, ii;
    npy_intp kk, ll, filter_size = 1, offsets_size = 1, max_size = 0;
    npy_intp max_stride = 0, *ashape = NULL, *astrides = NULL;
    npy_intp footprint_size = 0, coordinates[NPY_MAXDIMS], position[NPY_MAXDIMS];
    npy_intp fshape[NPY_MAXDIMS], forigins[NPY_MAXDIMS], *po, *pc = NULL;

    rank = array->nd;
    ashape = array->dimensions;
    astrides = array->strides;
    for(ii = 0; ii < rank; ii++) {
        fshape[ii] = *filter_shape++;
        forigins[ii] = origins ? *origins++ : 0;
    }
    /* the size of the footprint array: */
    for(ii = 0; ii < rank; ii++)
        filter_size *= fshape[ii];
    /* calculate the number of non-zero elements in the footprint: */
    if (footprint) {
        for(kk = 0; kk < filter_size; kk++)
            if (footprint[kk])
                ++footprint_size;
    } else {
        footprint_size = filter_size;
    }
    /* calculate how many sets of offsets must be stored: */
    for(ii = 0; ii < rank; ii++)
        offsets_size *= (ashape[ii] < fshape[ii] ? ashape[ii] : fshape[ii]);
    /* allocate offsets data: */
    *offsets = (npy_intp*)malloc(offsets_size * footprint_size *
                                                        sizeof(npy_intp));
    if (!*offsets) {
        PyErr_NoMemory();
        goto exit;
    }
    if (coordinate_offsets) {
        *coordinate_offsets = (npy_intp*)malloc(offsets_size * rank *
                                        footprint_size * sizeof(npy_intp));
        if (!*coordinate_offsets) {
            PyErr_NoMemory();
            goto exit;
        }
    }
    for(ii = 0; ii < rank; ii++) {
        npy_intp stride;
        /* find maximum axis size: */
        if (ashape[ii] > max_size)
            max_size = ashape[ii];
        /* find maximum stride: */
        stride = astrides[ii] < 0 ? -astrides[ii] : astrides[ii];
        if (stride > max_stride)
            max_stride = stride;
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
    po = *offsets;
    if (coordinate_offsets) {
        pc = *coordinate_offsets;
    }
    /* iterate over all regions: */
    for(ll = 0; ll < offsets_size; ll++) {
        /* iterate over the elements in the footprint array: */
        for(kk = 0; kk < filter_size; kk++) {
            npy_intp offset = 0;
            /* only calculate an offset if the footprint is 1: */
            if (!footprint || footprint[kk]) {
                /* find offsets along all axes: */
                for(ii = 0; ii < rank; ii++) {
                    npy_intp orgn = fshape[ii] / 2 + forigins[ii];
                    npy_intp cc = coordinates[ii] - orgn + position[ii];
                    npy_intp len = ashape[ii];
                    /* apply boundary conditions, if necessary: */
                    switch (mode) {
                    case NI_EXTEND_MIRROR:
                        if (cc < 0) {
                            if (len <= 1) {
                                cc = 0;
                            } else {
                                int sz2 = 2 * len - 2;
                                cc = sz2 * (int)(-cc / sz2) + cc;
                                cc = cc <= 1 - len ? cc + sz2 : -cc;
                            }
                        } else if (cc >= len) {
                            if (len <= 1) {
                                cc = 0;
                            } else {
                                int sz2 = 2 * len - 2;
                                cc -= sz2 * (int)(cc / sz2);
                                if (cc >= len)
                                    cc = sz2 - cc;
                            }
                        }
                        break;
                    case NI_EXTEND_REFLECT:
                        if (cc < 0) {
                            if (len <= 1) {
                                cc = 0;
                            } else {
                                int sz2 = 2 * len;
                                if (cc < -sz2)
                                    cc = sz2 * (int)(-cc / sz2) + cc;
                                cc = cc < -len ? cc + sz2 : -cc - 1;
                            }
                        } else if (cc >= len) {
                            if (len <= 1) {cc = 0;
                            } else {
                                int sz2 = 2 * len;
                                cc -= sz2 * (int)(cc / sz2);
                                if (cc >= len)
                                    cc = sz2 - cc - 1;
                            }
                        }
                        break;
                    case NI_EXTEND_WRAP:
                        if (cc < 0) {
                            if (len <= 1) {
                                cc = 0;
                            } else {
                                int sz = len;
                                cc += sz * (int)(-cc / sz);
                                if (cc < 0)
                                    cc += sz;
                            }
                        } else if (cc >= len) {
                            if (len <= 1) {
                                cc = 0;
                            } else {
                                int sz = len;
                                cc -= sz * (int)(cc / sz);
                            }
                        }
                        break;
                    case NI_EXTEND_NEAREST:
                        if (cc < 0) {
                            cc = 0;
                        } else if (cc >= len) {
                            cc = len - 1;
                        }
                        break;
                    case NI_EXTEND_CONSTANT:
                        if (cc < 0 || cc >= len)
                            cc = *border_flag_value;
                        break;
                    default:
                    PyErr_SetString(PyExc_RuntimeError,
                                                                                    "boundary mode not supported");
                        goto exit;
                    }

                    /* calculate offset along current axis: */
                    if (cc == *border_flag_value) {
                        /* just flag that we are outside the border */
                        offset = *border_flag_value;
                        if (coordinate_offsets)
                            pc[ii] = 0;
                        break;
                    } else {
                        /* use an offset that is possibly mapped from outside
                           the border: */
                        cc = cc - position[ii];
                        offset += astrides[ii] * cc;
                        if (coordinate_offsets)
                            pc[ii] = cc;
                    }
                }
                /* store the offset */
                *po++ = offset;
                if (coordinate_offsets)
                    pc += rank;
            }
            /* next point in the filter: */
            for(ii = rank - 1; ii >= 0; ii--) {
                if (coordinates[ii] < fshape[ii] - 1) {
                    coordinates[ii]++;
                    break;
                } else {
                    coordinates[ii] = 0;
                }
            }
        }

        /* move to the next array region: */
        for(ii = rank - 1; ii >= 0; ii--) {
            int orgn = fshape[ii] / 2 + forigins[ii];
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

 exit:
    if (PyErr_Occurred()) {
        if (*offsets)
            free(*offsets);
        if (coordinate_offsets && *coordinate_offsets)
            free(*coordinate_offsets);
        return 0;
    } else {
        return 1;
    }
}

}

