// This file contains much code ported from dlib
//
// DLIB is under the following copyright and license:
    // Copyright (C) 2009  Davis E. King (davis@dlib.net)
    // License: Boost Software License
    // (See the file LICENSE.BOOST in the mahotas distribution)
//
// Mahotas itself is
// Copyright (C) 2010-2013 Luis Pedro Coelho (luis@luispedro.org)
// License: MIT

#include "../numpypp/array.hpp"
#include "../numpypp/dispatch.hpp"
#include "../utils.hpp"

#include <vector>
#include <algorithm>
#include <cmath>
#include <sstream>
#include <limits>

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

typedef numpy::aligned_array<double> integral_image_type;

template <typename T>
double sum_rect(numpy::aligned_array<T> integral, int y0, int x0, int y1, int x1) {
    y0 = std::max<int>(y0-1, 0);
    x0 = std::max<int>(x0-1, 0);
    y1 = std::min<int>(y1-1, integral.dim(0) - 1);
    x1 = std::min<int>(x1-1, integral.dim(1) - 1);

    const T A = integral.at(y0,x0);
    const T B = integral.at(y0,x1);
    const T C = integral.at(y1,x0);
    const T D = integral.at(y1,x1);

    // This expression, unlike equivalent alternatives,
    // has no overflows. (D > B) and (C > A) and (D-B) > (C-A)
    return (D - B) - (C - A);
}

template <typename T>
double csum_rect(numpy::aligned_array<T> integral, int y, int x, const int dy, const int dx, int h, int w) {
    int y0 = y + dy - h/2;
    int x0 = x + dx - w/2;
    int y1 = y0 + h;
    int x1 = x0 + w;
    return sum_rect(integral, y0, x0, y1, x1);
}

double haar_x(const integral_image_type& integral, int y, int x, const int w) {
    const double left  = sum_rect(integral, y - w/2,  x - w/2, (y - w/2) + w,             x);
    const double right = sum_rect(integral, y - w/2,        x, (y - w/2) + w, (x - w/2) + w);
    return left - right;
}

double haar_y(const integral_image_type& integral, int y, int x, const int w) {
    const double top    = sum_rect(integral, y - w/2,  x - w/2,             y, (x - w/2) + w);
    const double bottom = sum_rect(integral,       y,  x - w/2, (y - w/2) + w, (x - w/2) + w);
    return top - bottom;
}
int round(double f) {
    if (f > 0) return int(f+.5);
    return int(f-.5);
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
    const int o,
    const int i,
    const int r,
    const int c
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
    for (int ii = i-1; ii <= i+1; ++ii) {
        for (int rr = r-1; rr <= r+1; ++rr) {
            for (int cc = c-1; cc <= c+1; ++cc) {
                if (pyr.get_value(o,ii,rr,cc) > val) return false;
            }
        }
    }

    return true;
}


const double pi = 3.1415926535898;
struct double_v2 {

    double_v2() {
        data_[0] = 0.;
        data_[1] = 0.;
    }
    double_v2(double y, double x) {
        data_[0] = y;
        data_[1] = x;
    }
    double& y() { return data_[0];}
    double& x() { return data_[1];}

    double y() const { return data_[0];}
    double x() const { return data_[1];}

    double angle() const { return std::atan2(data_[1], data_[0]); }
    double norm2() const { return data_[0]*data_[0] + data_[1]*data_[1]; }

    double_v2 abs() const { return double_v2(std::abs(data_[0]), std::abs(data_[1])); }

    void clear() {
        data_[0] = 0.;
        data_[1] = 0.;
    }

    double_v2& operator += (const double_v2& rhs) {
        data_[0] += rhs.data_[0];
        data_[1] += rhs.data_[1];
        return *this;
    }

    double_v2& operator -= (const double_v2& rhs) {
        data_[0] -= rhs.data_[0];
        data_[1] -= rhs.data_[1];
        return *this;
    }

    private:
    double data_[2];
};

inline
bool operator < (const double_v2& lhs, const double_v2& rhs) {
    return (lhs.y() == rhs.y()) ? (lhs.x() < rhs.x()) : (lhs.y() < rhs.y());
}

struct interest_point {
    interest_point()
        :scale(0)
        ,score(0)
        ,laplacian(0)
        { }

    double& y() { return center_.y(); }
    double& x() { return center_.x(); }

    double y() const { return center_.y(); }
    double x() const { return center_.x(); }

    double_v2& center() { return center_; }
    const double_v2& center() const { return center_; }

    double_v2 center_;
    double scale;
    double score;
    double laplacian;

    bool operator < (const interest_point& p) const { return score < p.score; }

    static const size_t ndoubles = 2 + 3;
    void dump(double out[ndoubles]) const {
        out[0] = center_.y();
        out[1] = center_.x();
        out[2] = scale;
        out[3] = score;
        out[4] = laplacian;
    }

    static
    interest_point load(const double in[ndoubles]) {
        interest_point res;
        res.center_.y() = in[0];
        res.center_.x() = in[1];
        res.scale = in[2];
        res.score = in[3];
        res.laplacian = in[4];
        return res;
    }
};


inline const interest_point interpolate_point (
    const hessian_pyramid& pyr,
    const int o,
    const int i,
    const int r,
    const int c,
    const int initial_step_size) {
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
    const double g0 = (pyr.get_value(o,i,r+1,c) - pyr.get_value(o,i,r-1,c))/2.0;
    const double g1 = (pyr.get_value(o,i,r,c+1) - pyr.get_value(o,i,r,c-1))/2.0;
    const double g2 = (pyr.get_value(o,i+1,r,c) - pyr.get_value(o,i-1,r,c))/2.0;

    // now compute inverse and multiply mat vec
    const double A = Mb*Mc-Mf*Mf;
    const double B = Ma*Mc-Me*Me;
    const double C = Ma*Mb-Md*Md;

    const double D = Me*Mf-Md*Mc;
    const double E = Md*Mf-Me*Mb;
    const double F = Md*Me-Mf*Ma;

    const double L = Ma*A - Md*D + Me*E;
    // it might be, that we encounter a degraded point. ignore it
    if (L == 0) {
        interest_point res;
        res.score = -std::numeric_limits<double>::max();
        return res;
    }

    // H^{-1} = 1./L | A D E |
    //               | D B F |
    //               | E F C |

    const double inter0 = ( A/L*g0 + D/L*g1 + E/L*g2 );
    const double inter1 = ( D/L*g0 + B/L*g1 + F/L*g2 );
    const double inter2 = ( E/L*g0 + F/L*g1 + C/L*g2 );

    interest_point res;
    if (std::max(std::abs(inter0), std::max(std::abs(inter1), std::abs(inter2))) < .5) {
        const int step = get_step_size(initial_step_size, o);
        const double p0 = (r + inter0) * step;
        const double p1 = (c + inter1) * step;
        const double lobe_size = std::pow(2.0, o+1.0)*(i+inter2+1) + 1;
        const double filter_size = 3*lobe_size;
        const double scale = 1.2/9.0 * filter_size;

        res.y() = p0;
        res.x() = p1;
        res.scale = scale;
        res.score = pyr.get_value(o,i,r,c);
        res.laplacian = pyr.get_laplacian(o,i,r,c);
    }
    return res;
}

void get_interest_points(
    const hessian_pyramid& pyr,
    double threshold,
    std::vector<interest_point>& result_points,
    const int initial_step_size) {
    assert(threshold >= 0);

    result_points.clear();
    const int nr_octaves = pyr.nr_octaves();
    const int nr_intervals = pyr.nr_intervals();

    for (int o = 0; o < nr_octaves; ++o) {
        const int border_size = get_border_size(o, nr_intervals);
        const int nr = pyr.nr(o);
        const int nc = pyr.nc(o);

        // do non-maximum suppression on all the intervals in the current octave and
        // accumulate the results in result_points
        for (int i = 1; i < nr_intervals-1;  i += 3) {
            for (int r = border_size+1; r < nr - border_size-1; r += 3) {
                for (int c = border_size+1; c < nc - border_size-1; c += 3) {
                    double max_val = pyr.get_value(o,i,r,c);
                    int max_i = i;
                    int max_r = r;
                    int max_c = c;

                    // loop over this 3x3x3 block and find the largest element
                    for (int ii = i; ii < std::min(i + 3, pyr.nr_intervals()-1); ++ii) {
                        for (int rr = r; rr < std::min(r + 3, nr - border_size - 1); ++rr) {
                            for (int cc = c; cc < std::min(c + 3, nc - border_size - 1); ++cc) {
                                double temp = pyr.get_value(o,ii,rr,cc);
                                if (temp > max_val) {
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
                    if (max_val > threshold && is_maximum_in_region(pyr, o, max_i, max_r, max_c)) {
                        interest_point sp = interpolate_point(pyr, o, max_i, max_r, max_c, initial_step_size);
                        if (sp.score > threshold) {
                            result_points.push_back(sp);
                        }
                    }
                }
            }
        }
    }
    // sort all the points by how strong their score is
    // We want the highest scoring in front, so we sort on rbegin()/rend()
    std::sort(result_points.rbegin(), result_points.rend());
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
        numpy::aligned_array<double>& cur_data = pyramid[o];

        for (int i = 0; i < nr_intervals; ++i) {
            const int lobe_size = static_cast<int>(std::pow(2.0, o+1.0)+0.5)*(i+1) + 1;
            const double area_inv = 1.0/std::pow(3.0*lobe_size, 2.0);
            const int lobe_offset = lobe_size/2+1;

            for (int y = border_size; y < N0 - border_size; y += step_size) {
                for (int x = border_size; x < N1 - border_size; x += step_size) {

                    double Dxx =     csum_rect(integral, y, x, 0, 0, 2*lobe_size-1, 3*lobe_size) -
                                 3.* csum_rect(integral, y, x, 0, 0, 2*lobe_size-1,   lobe_size);

                    double Dyy =     csum_rect(integral, y, x, 0, 0, 3*lobe_size, 2*lobe_size-1) -
                                 3.* csum_rect(integral, y, x, 0, 0,   lobe_size, 2*lobe_size-1);

                    double Dxy =    csum_rect(integral, y, x, -lobe_offset, +lobe_offset, lobe_size, lobe_size)
                                  + csum_rect(integral, y, x, +lobe_offset, -lobe_offset, lobe_size, lobe_size)
                                  - csum_rect(integral, y, x, +lobe_offset, +lobe_offset, lobe_size, lobe_size)
                                  - csum_rect(integral, y, x, -lobe_offset, -lobe_offset, lobe_size, lobe_size);

                    // now we normalize the filter responses
                    Dxx *= area_inv;
                    Dyy *= area_inv;
                    Dxy *= area_inv;

                    const double sign_of_laplacian = (Dxx + Dyy < 0) ? -1 : +1;
                    // The constant below is the matter of some debate:
                    // In the original papers, the authors use 0.81 (.9^2).
                    // However, some have claimed that 0.36 (.6^2) is better
                    // and the review "Local Invariant Feature Detectors: A
                    // Survey."by Tuytelaars T and Mikolajczyk K.; Foundations
                    // and Trends® in Computer Graphics and Vision.
                    // 2007;3(3):177-280. Available at:
                    // http://www.nowpublishers.com/product.aspx?product=CGV&doi=0600000017.
                    //
                    // Also uses 0.6
                    double determinant = Dxx*Dyy - 0.36*Dxy*Dxy;

                    // If the determinant is negative then just blank it out by setting
                    // it to zero.
                    if (determinant < 0) determinant = 0;

                    // Save the determinant of the Hessian into our image pyramid.  Also
                    // pack the laplacian sign into the value so we can get it out later.
                    cur_data.at(i,y/step_size,x/step_size) = sign_of_laplacian*determinant;
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

struct surf_point {
    interest_point p;
    double angle;
    double des[64];
    static const size_t ndoubles = interest_point::ndoubles + 1 + 64;
    void dump(double out[ndoubles]) const {
        p.dump(out);
        out[interest_point::ndoubles] = angle;
        std::memcpy(out+interest_point::ndoubles + 1, des, 64 * sizeof(double));
    }

};


inline
double gaussian (const double x, const double y, const double sig) {
    return 1.0/(sig*sig*2*pi) * std::exp( -(x*x + y*y)/(2*sig*sig));
}
inline
bool between_angles(const double a1, double a) {
    const double two_pi = 2*pi;
    const double window_size = pi/3;
    const double a2 = a1 + window_size;
    if ((a1 <= a) && (a < a2)) return true;
    a += two_pi;
    if ((a1 <= a) && (a < a2)) return true;
    return false;
}

double compute_dominant_angle(
        const integral_image_type& img,
        const double_v2& center,
        const double scale) {
    std::vector<std::pair<double, double_v2> > samples;

    // accumulate a bunch of angle and vector samples
    double_v2 vect;
    for (int r = -6; r <= 6; ++r) {
        for (int c = -6; c <= 6; ++c) {
            if (r*r + c*c < 36) {
                // compute a Gaussian weighted gradient and the gradient's angle.
                const double gauss = gaussian(c,r, 2.5);
                vect.y() = gauss*haar_y(img, round(scale*r+center.y()), round(scale*c+center.x()), (~1)&static_cast<int>(4*scale+0.5));
                vect.x() = gauss*haar_x(img, round(scale*r+center.y()), round(scale*c+center.x()), (~1)&static_cast<int>(4*scale+0.5));

                samples.push_back(std::make_pair(vect.angle(), vect));
            }
        }
    }

    const int Nsamples = samples.size();
    std::sort(samples.begin(), samples.end());

    // Perform the following:
    //
    // Loop: for i \in 0..Nsamples
    //          vect_i = \sum { a_j } | a_i <= a_j < a_i + pi/3
    //
    // where the angle comparison takes into account circularity (i.e., x == 2 pi + x)
    //
    // However, vect_{i+1} - vect_i = - a_i + ( \sum { a_j } |a_i + pi/3 <= a_j < a_{i+1} + pi/3 )
    //
    // So, we first compute vect_0 and then update it to get vect_i for i > 0

    int j;
    vect = samples[0].second;
    for (j = 1; j != Nsamples && between_angles(samples[0].first, samples[j].first); ++j) {
        vect += samples[j].second;
    }
    // If we got all of the elements in our window, we are done:
    if (j == Nsamples) return vect.angle();

    double max_length = vect.norm2();
    double best_ang = vect.angle();

    for (int i = 1; i < Nsamples; ++i) {
        vect -= samples[i].second;
        while (j != i && between_angles(samples[i].first, samples[j].first)) {
            vect += samples[j].second;
            ++j;
            if (j == Nsamples) j = 0;
        }

        const double cur_length = vect.norm2();
        if (cur_length > max_length) {
            max_length = cur_length;
            best_ang = vect.angle();
        }
    }

    return best_ang;
}

inline
double_v2 rotate_point(const double_v2& p, const double sin_angle, const double cos_angle) {
    return double_v2(
        cos_angle*p.x() - sin_angle*p.y(),
        sin_angle*p.x() + cos_angle*p.y());
}

// ----------------------------------------------------------------------------------------

void compute_surf_descriptor (
    const integral_image_type& img,
    double_v2 center,
    const double scale,
    const double angle,
    double des[64]) {
    assert(scale > 0);

    const double sin_angle = std::sin(angle);
    const double cos_angle = std::cos(angle);

    int count = 0;

    // loop over the 4x4 grid of histogram buckets
    for (int r = -10; r < 10; r += 5) {
        for (int c = -10; c < 10; c += 5) {
            double_v2 vect, abs_vect;

            // now loop over 25 points in this bucket and sum their features
            for (int y = r; y < r+5; ++y) {
                for (int x = c; x < c+5; ++x) {
                    // get the rotated point for this extraction point
                    double_v2 p = rotate_point(double_v2(x*scale, y*scale), sin_angle, cos_angle);
                    p += center;

                    const double gauss = gaussian(x,y, 3.3);
                    double_v2 temp(
                            gauss*haar_x(img, int(p.y()), int(p.x()), static_cast<int>(2*scale+0.5)),
                            gauss*haar_y(img, int(p.y()), int(p.x()), static_cast<int>(2*scale+0.5)));

                    // rotate this vector into alignment with the surf descriptor box
                    // This is a reverse rotation (takes advantage of the fact that
                    // sin(-a) = -sin(a) & cos(-a) = cos(a))
                    temp = rotate_point(temp, -sin_angle, cos_angle);

                    vect += temp;
                    abs_vect += temp.abs();
                }
            }

            des[count++] = vect.y();
            des[count++] = vect.x();
            des[count++] = abs_vect.y();
            des[count++] = abs_vect.x();
        }
    }

    assert(count == 64);

    // Return the length normalized descriptor.  Add a small number
    // to guard against division by zero.
    double len = 1e-7;
    for (int i = 0; i != 64; ++i) len += des[i]*des[i];
    len = std::sqrt(len);
    for (int i = 0; i != 64; ++i) des[i] /= len;
}

std::vector<surf_point> compute_descriptors(
            const integral_image_type& int_img,
            const std::vector<interest_point>& points,
            const int max_points) {
    std::vector<surf_point> spoints;
    const int N0 = int_img.dim(0);
    const int N1 = int_img.dim(1);
    for (unsigned i = 0; i < std::min(size_t(max_points), points.size()); ++i)
    {
        // ignore points that are close to the edge of the image
        const double border = 31;
        const interest_point& p = points[i];
        const unsigned long border_size = static_cast<unsigned long>(border*points[i].scale)/2;
        if (border_size <= p.y() && (p.y() + border_size) < N0 &&
            border_size <= p.x() && (p.x() + border_size) < N1) {
            surf_point sp;

            sp.angle = compute_dominant_angle(int_img, p.center(), p.scale);
            compute_surf_descriptor(int_img, p.center(), p.scale, sp.angle, sp.des);
            sp.p = p;
            spoints.push_back(sp);
        }
    }
    return spoints;
}

template<typename T>
std::vector<surf_point> get_surf_points(const numpy::aligned_array<T>& int_img, const int nr_octaves, const int nr_intervals, const int initial_step_size, const float threshold, const int max_points) {
    assert(max_points > 0);
    hessian_pyramid pyramid;

    gil_release nogil;
    std::vector<interest_point> points;
    build_pyramid<T>(int_img, pyramid, nr_octaves, nr_intervals, initial_step_size);
    get_interest_points(pyramid, threshold, points, initial_step_size);
    // compute descriptors and return
    return compute_descriptors(int_img, points, max_points);
}

PyObject* py_surf(PyObject* self, PyObject* args) {
    PyArrayObject* array;
    PyArrayObject* res;
    int nr_octaves;
    int nr_intervals;
    int initial_step_size;
    float threshold;
    int max_points;
    if (!PyArg_ParseTuple(args,"Oiiifi", &array, &nr_octaves, &nr_intervals, &initial_step_size, &threshold, &max_points)) return NULL;
    if (!PyArray_Check(array) ||
        PyArray_NDIM(array) != 2 ||
        PyArray_TYPE(array) != NPY_DOUBLE) {
        PyErr_SetString(PyExc_RuntimeError, TypeErrorMsg);
        return NULL;
    }
    holdref array_ref(array);
    try {
        std::vector<surf_point> spoints;
        spoints = get_surf_points<double>(
                        numpy::aligned_array<double>(array),
                        nr_octaves,
                        nr_intervals,
                        initial_step_size,
                        threshold,
                        max_points);

        numpy::aligned_array<double> arr = numpy::new_array<double>(spoints.size(), surf_point::ndoubles);
        for (unsigned int i = 0; i != spoints.size(); ++i) {
            spoints[i].dump(arr.data(i));
        }
        res = arr.raw_array();
        Py_INCREF(res);
        return PyArray_Return(res);
    } catch (const std::bad_alloc&) {
        PyErr_NoMemory();
        return NULL;
    } catch (const PythonException& exc) {
        PyErr_SetString(exc.type(), exc.message());
        return NULL;
    }
}

PyObject* py_descriptors(PyObject* self, PyObject* args) {
    PyArrayObject* array;
    PyArrayObject* points_arr;
    PyArrayObject* res;
    if (!PyArg_ParseTuple(args,"OO", &array, &points_arr)) return NULL;
    if (!numpy::are_arrays(array, points_arr) ||
        PyArray_NDIM(array) != 2 ||
        !PyArray_EquivTypenums(PyArray_TYPE(array), NPY_DOUBLE) ||
        !PyArray_EquivTypenums(PyArray_TYPE(points_arr), NPY_DOUBLE)) {
        PyErr_SetString(PyExc_RuntimeError, TypeErrorMsg);
        return NULL;
    }
    if (PyArray_NDIM(points_arr) != 2) {
        PyErr_SetString(PyExc_ValueError, "mahotas.features.surf.descriptors: interestpoints must be a two-dimensional array");
        return NULL;
    }
    if (PyArray_DIM(points_arr,1) != npy_intp(interest_point::ndoubles)) {
        std::ostringstream ss;
        ss << "mahotas.features.surf.descriptors: interestpoints must have " << interest_point::ndoubles
            << " entries per element (" << PyArray_DIM(points_arr, 1) << " were found).";
        PyErr_SetString(PyExc_ValueError, ss.str().c_str());
        return NULL;
    }
    holdref array_ref(array);
    try {

        std::vector<surf_point> spoints;
        { // no gil block
            gil_release nogil;
            numpy::aligned_array<double> points_raw(points_arr);
            const unsigned npoints = points_raw.dim(0);
            std::vector<interest_point> points;
            for (unsigned int i = 0; i != npoints; ++i) {
                points.push_back(interest_point::load(points_raw.data(i)));
            }
            spoints = compute_descriptors(integral_image_type(array), points, npoints);
        }

        numpy::aligned_array<double> arr = numpy::new_array<double>(spoints.size(), surf_point::ndoubles);
        for (unsigned int i = 0; i != spoints.size(); ++i) {
            spoints[i].dump(arr.data(i));
        }
        res = arr.raw_array();
        Py_INCREF(res);
        return PyArray_Return(res);
    } catch (const std::bad_alloc&) {
        PyErr_NoMemory();
        return NULL;
    } catch (const PythonException& exc) {
        PyErr_SetString(exc.type(), exc.message());
        return NULL;
    }
}


PyObject* py_interest_points(PyObject* self, PyObject* args) {
    PyArrayObject* array;
    PyArrayObject* res;
    int nr_octaves;
    int nr_intervals;
    int initial_step_size;
    int max_points;
    float threshold;
    if (!PyArg_ParseTuple(args,"Oiiifi", &array, &nr_octaves, &nr_intervals, &initial_step_size, &threshold, &max_points)) return NULL;
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
            get_interest_points(pyramid, threshold, interest_points, initial_step_size); \
            if (max_points >= 0 && interest_points.size() > unsigned(max_points)) { \
                interest_points.erase( \
                            interest_points.begin() + max_points, \
                            interest_points.end()); \
            } \
        }

            HANDLE_TYPES();
        #undef HANDLE
            default:
            PyErr_SetString(PyExc_RuntimeError, TypeErrorMsg);
            return NULL;
        }
        numpy::aligned_array<double> arr = numpy::new_array<double>(interest_points.size(), interest_point::ndoubles);
        for (unsigned int i = 0; i != interest_points.size(); ++i) {
            interest_points[i].dump(arr.data(i));
        }
        res = arr.raw_array();
        Py_INCREF(res);
    } catch (const std::bad_alloc&) {
        PyErr_NoMemory();
        return NULL;
    } catch (const PythonException& exc) {
        PyErr_SetString(exc.type(), exc.message());
        return NULL;
    }
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
        PyList_SET_ITEM(pyramid_list, o, arr);
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
PyObject* py_sum_rect(PyObject* self, PyObject* args) {
    PyArrayObject* array;
    int y0, x0, y1, x1;
    if (!PyArg_ParseTuple(args,"Oiiii", &array, &y0, &x0, &y1, &x1)) return NULL;
    if (!PyArray_Check(array) || PyArray_NDIM(array) != 2) {
        PyErr_SetString(PyExc_RuntimeError, TypeErrorMsg);
        return NULL;
    }
    holdref a_ref(array);
    double res;
    switch(PyArray_TYPE(array)) {
    #define HANDLE(type) \
        res = sum_rect<type>(numpy::aligned_array<type>(array), y0, x0, y1, x1);

        HANDLE_TYPES();
    #undef HANDLE
        default:
        PyErr_SetString(PyExc_RuntimeError, TypeErrorMsg);
        return NULL;
    }
    return PyFloat_FromDouble(res);
}

PyMethodDef methods[] = {
  {"integral",(PyCFunction)py_integral, METH_VARARGS, NULL},
  {"pyramid",(PyCFunction)py_pyramid, METH_VARARGS, NULL},
  {"interest_points",(PyCFunction)py_interest_points, METH_VARARGS, NULL},
  {"sum_rect",(PyCFunction)py_sum_rect, METH_VARARGS, NULL},
  {"descriptors",(PyCFunction)py_descriptors, METH_VARARGS, NULL},
  {"surf",(PyCFunction)py_surf, METH_VARARGS, NULL},
  {NULL, NULL,0,NULL},
};

} // namespace
DECLARE_MODULE(_surf)
