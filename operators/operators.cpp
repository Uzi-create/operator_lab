// C++17, contiguous grayscale float32 images in [0, 1].
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>
#include "advanced.hpp"
#include "geometry.hpp"
#include "grasp_depth.hpp"
#include "metal_ridge.hpp"
#include "local_stats.hpp"
#include "perception_raster.hpp"
#include "nearest_neighbor.hpp"
#include "depth_components.hpp"
#include "voxel_components.hpp"

extern "C" {
// mode: 0 threshold, 1 fast mean, 2 adaptive smoothing, 3 naive mean.
// mode: 4 self-guided, 5 CLAHE, 6 erode, 7 dilate, 8 open, 9 close.
// Window operators use clipped borders; CLAHE interpolates tile-center maps.
// Buffers must be distinct and hold width * height floats.
int run_operator(const float* src, float* dst, int width, int height,
                 int radius, int mode, float parameter) noexcept {
    if (!src || !dst || src == dst || width <= 0 || height <= 0 || radius < 0 ||
        mode < 0 || mode > 9 || !std::isfinite(parameter) ||
        ((mode == 2 || mode == 4) && parameter <= 0) ||
        (mode == 5 && (radius < 2 || parameter < 1))) return 1;
    const size_t w = width, h = height;
    if (w > std::numeric_limits<size_t>::max() / h) return 1;
    const size_t n = w * h;
    for (size_t i = 0; i < n; ++i)
        if (!std::isfinite(src[i]) || src[i] < 0 || src[i] > 1) return 1;
    try {
        if (mode == 0) {
            for (size_t i = 0; i < n; ++i) dst[i] = src[i] > parameter ? 1.f : 0.f;
            return 0;
        }
        const size_t r = std::min<size_t>(radius, std::max(w, h));
        if (mode == 4) {
            advanced::guided(src,dst,w,h,r,parameter);
            return 0;
        }
        if (mode == 5) {
            advanced::clahe(src,dst,w,h,size_t(radius),parameter);
            return 0;
        }
        if (mode >= 6) {
            if (mode <= 7) advanced::morphology(src,dst,w,h,r,mode == 7);
            else {
                std::vector<float> intermediate(n);
                advanced::morphology(src,intermediate.data(),w,h,r,mode == 9);
                advanced::morphology(intermediate.data(),dst,w,h,r,mode == 8);
            }
            return 0;
        }
        if (mode == 3) {
            for (size_t y = 0; y < h; ++y) for (size_t x = 0; x < w; ++x) {
                const size_t x0 = x > r ? x-r : 0, y0 = y > r ? y-r : 0;
                const size_t x1 = std::min(w, x+r+1), y1 = std::min(h, y+r+1);
                double sum = 0;
                for (size_t yy = y0; yy < y1; ++yy)
                    for (size_t xx = x0; xx < x1; ++xx) sum += src[yy*w+xx];
                dst[y*w+x] = float(sum / ((x1-x0)*(y1-y0)));
            }
            return 0;
        }
        // Integral images: each rectangular sum needs four table lookups.
        const size_t stride = w+1;
        if (stride > std::numeric_limits<size_t>::max() / (h+1)) return 1;
        std::vector<double> sums(stride*(h+1), 0);
        std::vector<double> squares(mode == 2 ? sums.size() : 0, 0);
        for (size_t y = 0; y < h; ++y) {
            double row_sum = 0, row_square = 0;
            for (size_t x = 0; x < w; ++x) {
                const double v = src[y*w+x];
                row_sum += v;
                sums[(y+1)*stride+x+1] = sums[y*stride+x+1] + row_sum;
                if (mode == 2) {
                    row_square += v*v;
                    squares[(y+1)*stride+x+1] = squares[y*stride+x+1] + row_square;
                }
            }
        }
        auto rectangle = [stride](const std::vector<double>& a, size_t x0,
                                  size_t y0, size_t x1, size_t y1) {
            return a[y1*stride+x1]-a[y0*stride+x1]-a[y1*stride+x0]+a[y0*stride+x0];
        };
        for (size_t y = 0; y < h; ++y) for (size_t x = 0; x < w; ++x) {
            const size_t x0 = x > r ? x-r : 0, y0 = y > r ? y-r : 0;
            const size_t x1 = std::min(w, x+r+1), y1 = std::min(h, y+r+1);
            const double count = double((x1-x0)*(y1-y0));
            const double mean = rectangle(sums, x0, y0, x1, y1) / count;
            double out = mean;
            if (mode == 2) {
                const double variance = std::max(0., rectangle(squares, x0, y0, x1, y1)/count-mean*mean);
                const double weight = variance / (variance + double(parameter)*parameter);
                out = mean + weight*(src[y*w+x]-mean);
            }
            dst[y*w+x] = float(std::clamp(out, 0., 1.));
        }
        return 0;
    } catch (...) { return 2; }
}
}
