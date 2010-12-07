#include "numpypp/array.hpp"
#include "numpypp/dispatch.hpp"
#include "utils.hpp"

#include <vector>
#include <algorithm>
#include <cmath>

extern "C" {
    #include <Python.h>
    #include <numpy/ndarrayobject.h>
}

namespace{

const char TypeErrorMsg[] =
    "Type not understood. "
    "This is caused by either a direct call to _surf (which is dangerous: types are not checked!) or a bug in surf.py.\n";

/* SURF: Speeded-Up Robust Features
 *
 * The implementation here is a port from DLIB, which is in turn influenced by
 * the very well documented OpenSURF library and its corresponding description
 * of how the fast Hessian algorithm functions: "Notes on the OpenSURF Library"
 * by Christopher Evans.
 */

template <typename T>
double sum_rect(numpy::aligned_array<T> integral, int y, int x, const int dy, const int dx, int h, int w) {
    int y0 = y + dy - h/2;
    int x0 = x + dx - w/2;
    int y1 = y0 + h;
    int x1 = x0 + w;
    y0 = std::max<T>(y0, 0);
    y1 = std::max<T>(y1, 0);
    x0 = std::min<T>(x0, integral.dim(1));
    x1 = std::min<T>(x1, integral.dim(1));

    const T A = integral.at(y0,x0);
    const T B = integral.at(y0,x1);
    const T C = integral.at(y1,x0);
    const T D = integral.at(y1,x1);

    // This expression, unlike equivalent alternatives,
    // has no overflows. (D > B) and (C > A) and (D-B) > (C-A)
    return (D - B) - (C - A);
}

int get_border_size(const int octave, const int nr_intervals) {
    const double lobe_size = std::pow(2.0, octave+1.0)*(nr_intervals+1) + 1;
    const double filter_size = 3*lobe_size;

    const int bs = static_cast<int>(std::ceil(filter_size/2.0));
    return bs;
}
int get_step_size(const int initial_step_size, const int octave) {
    return initial_step_size*static_cast<int>(std::pow(2.0, double(octave))+0.5);
}

struct hessian_pyramid {
    typedef std::vector<numpy::aligned_array<double> > pyramid_type;
    pyramid_type pyr;
    double get_laplacian(int o, int i, int r, int c) const {
        return pyr[o].at(i,r,c) < 0 ? -1. : +1.;
    }
    double get_value(int o, int i, int r, int c) const {
        return std::abs(pyr[o].at(i,r,c));
    }
    int nr_intervals() const { return pyr[0].dim(0); }
    int nr_octaves() const { return pyr.size(); }
    int nr(const int o) const { return pyr[o].dim(1); }
    int nc(const int o) const { return pyr[o].dim(2); }
};


inline bool is_maximum_in_region(
    const hessian_pyramid& pyr,
    int o,
    int i,
    int r,
    int c
)
{
    // First check if this point is near the edge of the octave
    // If it is then we say it isn't a maximum as these points are
    // not as reliable.
    if (i <= 0 || i+1 >= pyr.nr_intervals()) return false;
    assert(r > 0);
    assert(c > 0);

    const double val = pyr.get_value(o,i,r,c);

    // now check if there are any bigger values around this guy
    for (int ii = i-1; ii <= i+1; ++ii)
    {
        for (int rr = r-1; rr <= r+1; ++rr)
        {
            for (int cc = c-1; cc <= c+1; ++cc)
            {
                if (pyr.get_value(o,ii,rr,cc) > val)
                    return false;
            }
        }
    }

    return true;
}
struct interest_point
{
    interest_point()
        :c0(0.)
        ,c1(0.)
        ,scale(0)
        ,score(0)
        ,laplacian(0)
        { }

    double c0;
    double c1;
    double scale;
    double score;
    double laplacian;

    bool operator < (const interest_point& p) const { return score < p.score; }
};


inline const interest_point interpolate_point (
    const hessian_pyramid& pyr,
    const int o,
    const int i,
    const int r,
    const int c
)
{
    // The original (dlib) code reads:
    //
    // interpolated_point = -inv(get_hessian_hessian(pyr,o,i,r,c))*
    //                          get_hessian_gradient(pyr,o,i,r,c);
    //
    //  instead of doing this, we are inlining the matrices here
    //  and solving the 3x3 inversion and vector multiplication directly.

    const double val = pyr.get_value(o,i,r,c);

    // get_hessian_hessian:

    const double Dxx = (pyr.get_value(o,i,r,c+1) + pyr.get_value(o,i,r,c-1)) - 2*val;
    const double Dyy = (pyr.get_value(o,i,r+1,c) + pyr.get_value(o,i,r-1,c)) - 2*val;
    const double Dss = (pyr.get_value(o,i+1,r,c) + pyr.get_value(o,i-1,r,c)) - 2*val;

    const double Dxy = (pyr.get_value(o,i,r+1,c+1) + pyr.get_value(o,i,r-1,c-1) -
                  pyr.get_value(o,i,r-1,c+1) - pyr.get_value(o,i,r+1,c-1)) / 4.0;

    const double Dxs = (pyr.get_value(o,i+1,r,c+1) + pyr.get_value(o,i-1,r,c-1) -
                  pyr.get_value(o,i-1,r,c+1) - pyr.get_value(o,i+1,r,c-1)) / 4.0;

    const double Dys = (pyr.get_value(o,i+1,r+1,c) + pyr.get_value(o,i-1,r-1,c) -
                  pyr.get_value(o,i-1,r+1,c) - pyr.get_value(o,i+1,r-1,c)) / 4.0;

    // H  = | Dxx, Dxy, Dxs | = | a d e |
    //      | Dxy, Dyy, Dys |   | d b f |
    //      | Dxs, Dys, Dss |   | e f c |

    const double Ma = Dxx;
    const double Mb = Dyy;
    const double Mc = Dss;
    const double Md = Dxy;
    const double Me = Dxs;
    const double Mf = Dys;

    // get_hessian_gradient:
    const double g0 = (pyr.get_value(o,i,r,c+1) - pyr.get_value(o,i,r,c-1))/2.0;
    const double g1 = (pyr.get_value(o,i,r+1,c) - pyr.get_value(o,i,r-1,c))/2.0;
    const double g2 = (pyr.get_value(o,i+1,r,c) - pyr.get_value(o,i-1,r,c))/2.0;

    // now compute inverse and multiply mat vec
    const double A = Mb*Mc-Mf*Mf;
    const double B = Ma*Mc-Me*Me;
    const double C = Ma*Mb-Md*Md;

    const double D = Me*Mf-Md*Mc;
    const double E = Md*Mf-Me*Mb;
    const double F = Md*Me-Mf*Ma;

    const double L = Ma*A - Md*D + Me*E;
    if (L == 0) throw PythonException(PyExc_RuntimeError, "Determinant is zero.");

    // H^{-1} = 1./L | A D E |
    //               | D B F |
    //               | E F C |

    const double inter0 = ( A/L*g0 + D/L*g1 + E/L*g2 );
    const double inter1 = ( D/L*g0 + B/L*g1 + F/L*g2 );
    const double inter2 = ( E/L*g0 + F/L*g1 + C/L*g2 );

    interest_point res;
    if (std::max(inter0, std::max(inter1, inter2)) < .5) {
        const int initial_step_size = 1;
        const int step = get_step_size(initial_step_size, o);
        const double p0 = i + inter0 * step;
        const double p1 = r + inter1 * step;
        const double p2 = c + inter2 * step;
        const double lobe_size = std::pow(2.0, o+1.0)*(i+p0+1) + 1;
        const double filter_size = 3*lobe_size;
        const double scale = 1.2/9.0 * filter_size;

        res.c0 = r;
        res.c1 = c;
        res.scale = scale;
        res.score = pyr.get_value(o,i,r,c);
        res.laplacian = pyr.get_laplacian(o,i,r,c);
    }
    return res;
}

void get_interest_points (
    const hessian_pyramid& pyr,
    double threshold,
    std::vector<interest_point>& result_points) {
    assert(threshold >= 0);

    result_points.clear();
    const int nr_octaves = pyr.nr_octaves();
    const int nr_intervals = pyr.nr_intervals();

    for (int o = 0; o < nr_octaves; ++o)
    {
        const int border_size = get_border_size(o, nr_intervals);
        const int nr = pyr.nr(o);
        const int nc = pyr.nc(o);

        // do non-maximum suppression on all the intervals in the current octave and
        // accumulate the results in result_points
        for (int i = 1; i < nr_intervals-1;  i += 3)
        {
            for (int r = border_size+1; r < nr - border_size-1; r += 3)
            {
                for (int c = border_size+1; c < nc - border_size-1; c += 3)
                {
                    double max_val = pyr.get_value(o,i,r,c);
                    int max_i = i;
                    int max_r = r;
                    int max_c = c;

                    // loop over this 3x3x3 block and find the largest element
                    for (int ii = i; ii < std::min(i + 3, pyr.nr_intervals()-1); ++ii)
                    {
                        for (int rr = r; rr < std::min(r + 3, nr - border_size - 1); ++rr)
                        {
                            for (int cc = c; cc < std::min(c + 3, nc - border_size - 1); ++cc)
                            {
                                double temp = pyr.get_value(o,ii,rr,cc);
                                if (temp > max_val)
                                {
                                    max_val = temp;
                                    max_i = ii;
                                    max_r = rr;
                                    max_c = cc;
                                }
                            }
                        }
                    }

                    // If the max point we found is really a maximum in its own region and
                    // is big enough then add it to the results.
                    if (max_val > threshold && is_maximum_in_region(pyr, o, max_i, max_r, max_c))
                    {
                        interest_point sp = interpolate_point (pyr, o, max_i, max_r, max_c);
                        if (sp.score > threshold)
                        {
                            result_points.push_back(sp);
                        }
                    }

                }
            }
        }
    }
}

template <typename T>
void build_pyramid(numpy::aligned_array<T> integral,
                hessian_pyramid& hpyramid,
                const int nr_octaves,
                const int nr_intervals,
                const int initial_step_size) {
    assert(nr_octaves > 0);
    assert(nr_intervals > 0);
    assert(initial_step_size > 0);

    hessian_pyramid::pyramid_type& pyramid = hpyramid.pyr;
    const int N0 = integral.dim(0);
    const int N1 = integral.dim(1);
    // allocate space for the pyramid
    pyramid.reserve(nr_octaves);
    for (int o = 0; o < nr_octaves; ++o)
    {
        const int step_size = get_step_size(initial_step_size, o);
        pyramid.push_back(numpy::new_array<double>(nr_intervals, N0/step_size, N1/step_size));
        PyArray_FILLWBYTE(pyramid[o].raw_array(), 0);
    }

    // now fill out the pyramid with data
    for (int o = 0; o < nr_octaves; ++o)
    {
        const int step_size = get_step_size(initial_step_size, o);
        const int border_size = get_border_size(o, nr_intervals)*step_size;
        double *pout = pyramid[o].data();

        for (int i = 0; i < nr_intervals; ++i)
        {
            const int lobe_size = static_cast<int>(std::pow(2.0, o+1.0)+0.5)*(i+1) + 1;
            const double area_inv = 1.0/std::pow(3.0*lobe_size, 2.0);
            const int lobe_offset = lobe_size/2+1;

            for (int y = border_size; y < N0 - border_size; y += step_size)
            {
                for (int x = border_size; x < N1 - border_size; x += step_size)
                {
                    double Dxx =     sum_rect(integral, y, x, 0, 0, lobe_size*3, 2*lobe_size-1) -
                                 3.* sum_rect(integral, y, x, 0, 0, lobe_size,   2*lobe_size-1);

                    double Dyy =     sum_rect(integral, y, x, 0, 0, 2*lobe_size-1, lobe_size*3) -
                                 3.* sum_rect(integral, y, x, 0, 0, 2*lobe_size-1, lobe_size);

                    double Dxy = sum_rect(integral, y, x, -lobe_offset, +lobe_offset, lobe_size, lobe_size) +
                                 sum_rect(integral, y, x, +lobe_offset, -lobe_offset, lobe_size, lobe_size) -
                                 sum_rect(integral, y, x, +lobe_offset, +lobe_offset, lobe_size, lobe_size) -
                                 sum_rect(integral, y, x, -lobe_offset, -lobe_offset, lobe_size, lobe_size);

                    // now we normalize the filter responses
                    Dxx *= area_inv;
                    Dyy *= area_inv;
                    Dxy *= area_inv;

                    const double sign_of_laplacian = (Dxx + Dyy < 0) ? -1 : +1;
                    double determinant = Dxx*Dyy - 0.81*Dxy*Dxy;

                    // If the determinant is negative then just blank it out by setting
                    // it to zero.
                    if (determinant < 0) determinant = 0;

                    // Save the determinant of the Hessian into our image pyramid.  Also
                    // pack the laplacian sign into the value so we can get it out later.
                    *pout++ = sign_of_laplacian*determinant;
                }
            }

        }
    }
}

template <typename T>
void integral(numpy::aligned_array<T> array) {
    gil_release nogil;
    const int N0 = array.dim(0);
    const int N1 = array.dim(1);
    if (N0 == 0 || N1 == 0) return;
    for (int j = 1; j != N1; ++j) {
        array.at(0, j) += array.at(0, j - 1);
    }
    for (int i = 1; i != N0; ++i) {
        array.at(i,0) += array.at(i-1,0);
        for (int j = 1; j != N1; ++j) {
            array.at(i,j) += array.at(i-1, j) + array.at(i, j-1) - array.at(i-1, j-1);
        }
    }
}

PyObject* py_interest_points(PyObject* self, PyObject* args) {
    PyArrayObject* array;
    PyArrayObject* res;
    int nr_octaves;
    int nr_intervals;
    int initial_step_size;
    if (!PyArg_ParseTuple(args,"Oiii", &array, &nr_octaves, &nr_intervals, &initial_step_size)) return NULL;
    if (!PyArray_Check(array) || PyArray_NDIM(array) != 2) {
        PyErr_SetString(PyExc_RuntimeError, TypeErrorMsg);
        return NULL;
    }
    holdref array_ref(array);
    hessian_pyramid pyramid;
    std::vector<interest_point> interest_points;
    try {
        switch(PyArray_TYPE(array)) {
        #define HANDLE(type) {\
            gil_release nogil; \
            build_pyramid<type>(numpy::aligned_array<type>(array), pyramid, nr_octaves, nr_intervals, initial_step_size); \
            get_interest_points(pyramid, 0, interest_points); \
        }

            HANDLE_TYPES();
        #undef HANDLE
            default:
            PyErr_SetString(PyExc_RuntimeError, TypeErrorMsg);
            return NULL;
        }
        numpy::aligned_array<double> arr = numpy::new_array<double>(interest_points.size(), 3);
        for (unsigned int i = 0; i != interest_points.size(); ++i) {
            arr.at(i, 0) = interest_points[i].c0;
            arr.at(i, 1) = interest_points[i].c1;
            arr.at(i, 2) = interest_points[i].score;
        }
        res = arr.raw_array();
    } catch (const std::bad_alloc&) {
        PyErr_NoMemory();
        return NULL;
    } catch (const PythonException& exc) {
        PyErr_SetString(exc.type(), exc.message());
        return NULL;
    }
    Py_INCREF(res);
    return PyArray_Return(res);
}

PyObject* py_pyramid(PyObject* self, PyObject* args) {
    PyArrayObject* array;
    int nr_octaves;
    int nr_intervals;
    int initial_step_size;
    if (!PyArg_ParseTuple(args,"Oiii", &array, &nr_octaves, &nr_intervals, &initial_step_size)) return NULL;
    if (!PyArray_Check(array) || PyArray_NDIM(array) != 2) {
        PyErr_SetString(PyExc_RuntimeError, TypeErrorMsg);
        return NULL;
    }
    holdref array_ref(array);
    hessian_pyramid pyramid;
    try {
        switch(PyArray_TYPE(array)) {
        #define HANDLE(type) \
            build_pyramid<type>(numpy::aligned_array<type>(array), pyramid, nr_octaves, nr_intervals, initial_step_size);

            HANDLE_TYPES();
        #undef HANDLE
            default:
            PyErr_SetString(PyExc_RuntimeError, TypeErrorMsg);
            return NULL;
        }
    } catch (const std::bad_alloc&) {
        PyErr_NoMemory();
        return NULL;
    } catch (const PythonException& exc) {
        PyErr_SetString(exc.type(), exc.message());
        return NULL;
    }
    PyObject* pyramid_list = PyList_New(nr_octaves);
    if (!pyramid_list) return NULL;
    for (int o = 0; o != nr_octaves; ++o) {
        PyObject* arr = reinterpret_cast<PyObject*>(pyramid.pyr.at(o).raw_array());
        Py_INCREF(arr);
        PyList_SetItem(pyramid_list, o, arr);
    }
    return pyramid_list;
}


PyObject* py_integral(PyObject* self, PyObject* args) {
    PyArrayObject* array;
    if (!PyArg_ParseTuple(args,"O", &array)) return NULL;
    if (!PyArray_Check(array) || PyArray_NDIM(array) != 2) {
        PyErr_SetString(PyExc_RuntimeError, TypeErrorMsg);
        return NULL;
    }
    Py_INCREF(array);
    switch(PyArray_TYPE(array)) {
    #define HANDLE(type) \
        integral<type>(numpy::aligned_array<type>(array));

        HANDLE_TYPES();
    #undef HANDLE
        default:
        PyErr_SetString(PyExc_RuntimeError, TypeErrorMsg);
        return NULL;
    }
    return PyArray_Return(array);
}

PyMethodDef methods[] = {
  {"integral",(PyCFunction)py_integral, METH_VARARGS, NULL},
  {"pyramid",(PyCFunction)py_pyramid, METH_VARARGS, NULL},
  {"interest_points",(PyCFunction)py_interest_points, METH_VARARGS, NULL},
  {NULL, NULL,0,NULL},
};

} // namespace
extern "C"
void init_surf()
  {
    import_array();
    (void)Py_InitModule("_surf", methods);
  }

