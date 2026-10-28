#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace operator_lab::rgbd {

struct Point3 { double x = 0, y = 0, z = 0; };
struct Pixel { std::size_t u = 0, v = 0; };
struct Rect { std::size_t x = 0, y = 0, width = 0, height = 0; };
struct Intrinsics {
    std::size_t width = 0, height = 0;
    double fx = 0, fy = 0, cx = 0, cy = 0;
};

inline bool finite(Point3 p) {
    return std::isfinite(p.x) && std::isfinite(p.y) && std::isfinite(p.z);
}

inline void validate(const Intrinsics& k) {
    if (!k.width || !k.height || k.width > 16384 || k.height > 16384 ||
        !std::isfinite(k.fx) || !std::isfinite(k.fy) || k.fx <= 0 || k.fy <= 0 ||
        !std::isfinite(k.cx) || !std::isfinite(k.cy))
        throw std::invalid_argument("invalid pinhole intrinsics");
}

inline void validate(Rect roi, std::size_t width, std::size_t height) {
    if (!roi.width || !roi.height || roi.x >= width || roi.y >= height ||
        roi.width > width - roi.x || roi.height > height - roi.y)
        throw std::invalid_argument("ROI must be nonempty and fully inside the depth image");
}

// The input must already be undistorted. Output is in the optical frame:
// +X right, +Y down, +Z forward; depth is optical Z in METERS, not ray length.
inline Point3 backproject(const Intrinsics& k, double u, double v, double depth_m) {
    validate(k);
    if (!std::isfinite(u) || !std::isfinite(v) || u < 0 || v < 0 ||
        u > static_cast<double>(k.width - 1) || v > static_cast<double>(k.height - 1) ||
        !std::isfinite(depth_m) || depth_m <= 0)
        throw std::invalid_argument("invalid pixel or metric optical depth");
    const Point3 point{((u - k.cx) / k.fx) * depth_m,
                       ((v - k.cy) / k.fy) * depth_m, depth_m};
    if (!finite(point)) throw std::overflow_error("backprojection overflow");
    return point;
}

// Row-major R and translation encode T_destination_from_source.
struct RigidTransform {
    std::array<double, 9> rotation{1, 0, 0, 0, 1, 0, 0, 0, 1};
    Point3 translation_m{};
};

inline void validate(const RigidTransform& t) {
    if (!finite(t.translation_m)) throw std::invalid_argument("nonfinite translation");
    const auto& r = t.rotation;
    for (double value : r)
        if (!std::isfinite(value)) throw std::invalid_argument("nonfinite rotation");
    for (std::size_t i = 0; i < 3; ++i)
        for (std::size_t j = 0; j < 3; ++j) {
            double dot = 0;
            for (std::size_t k = 0; k < 3; ++k) dot += r[i * 3 + k] * r[j * 3 + k];
            if (!std::isfinite(dot) || std::abs(dot - (i == j ? 1. : 0.)) > 1e-6)
                throw std::invalid_argument("rotation must be orthonormal");
        }
    const double determinant = r[0] * (r[4] * r[8] - r[5] * r[7]) -
        r[1] * (r[3] * r[8] - r[5] * r[6]) + r[2] * (r[3] * r[7] - r[4] * r[6]);
    if (!std::isfinite(determinant) || std::abs(determinant - 1.) > 1e-6)
        throw std::invalid_argument("rotation must have determinant +1");
}

inline Point3 transform_point(Point3 point, const RigidTransform& t) {
    validate(t);
    if (!finite(point)) throw std::invalid_argument("nonfinite source point");
    const auto& r = t.rotation;
    const Point3 out{r[0]*point.x + r[1]*point.y + r[2]*point.z + t.translation_m.x,
                     r[3]*point.x + r[4]*point.y + r[5]*point.z + t.translation_m.y,
                     r[6]*point.x + r[7]*point.y + r[8]*point.z + t.translation_m.z};
    if (!finite(out)) throw std::overflow_error("rigid transform overflow");
    return out;
}

// Non-owning view; caller retains the buffer for the duration of every call.
// row_stride is measured in ELEMENTS. elements is the accessible buffer length.
// scale_to_m is explicit: e.g. uint16 millimeters -> 0.001, float meters -> 1.
template<class T> class DepthView {
    static_assert(std::is_same_v<T, std::uint16_t> || std::is_same_v<T, float> ||
                  std::is_same_v<T, double>, "supported depths: uint16_t, float, double");
public:
    DepthView(const T* data, std::size_t elements, std::size_t width, std::size_t height,
              double scale_to_m, std::size_t row_stride = 0)
        : data_(data), width_(width), height_(height),
          stride_(row_stride ? row_stride : width), scale_(scale_to_m) {
        if (!data || !width || !height || width > 16384 || height > 16384 ||
            width > 16000000 / height || stride_ < width ||
            !std::isfinite(scale_) || scale_ <= 0 || elements < width ||
            height - 1 > (elements - width) / stride_)
            throw std::invalid_argument("invalid depth buffer, extent, stride or scale");
    }
    std::size_t width() const { return width_; }
    std::size_t height() const { return height_; }
    double scale_to_m() const { return scale_; }
    T raw_at(std::size_t u, std::size_t v) const {
        if (u >= width_ || v >= height_) throw std::out_of_range("depth pixel outside image");
        return data_[v * stride_ + u];
    }
    double at_m(std::size_t u, std::size_t v) const {
        if (u >= width_ || v >= height_) throw std::out_of_range("depth pixel outside image");
        const double z = static_cast<double>(data_[v * stride_ + u]) * scale_;
        return std::isfinite(z) && z > 0 ? z : std::numeric_limits<double>::quiet_NaN();
    }
private:
    const T* data_;
    std::size_t width_, height_, stride_;
    double scale_;
};

struct PointSample { Pixel pixel; Point3 point_m; };

template<class T>
inline std::vector<PointSample> depth_to_points(const DepthView<T>& depth, const Intrinsics& k,
    Rect roi, double min_depth_m = 0.1, double max_depth_m = 8., std::size_t sampling_step = 1) {
    validate(k); validate(roi, depth.width(), depth.height());
    if (k.width != depth.width() || k.height != depth.height() || !sampling_step ||
        sampling_step > 16384 || !std::isfinite(min_depth_m) || min_depth_m < 0 ||
        !std::isfinite(max_depth_m) || max_depth_m <= min_depth_m)
        throw std::invalid_argument("invalid point-cloud configuration");
    std::vector<PointSample> out;
    out.reserve(((roi.width - 1) / sampling_step + 1) * ((roi.height - 1) / sampling_step + 1));
    for (std::size_t v = roi.y; v < roi.y + roi.height; v += sampling_step)
        for (std::size_t u = roi.x; u < roi.x + roi.width; u += sampling_step) {
            const double z = depth.at_m(u, v);
            if (std::isfinite(z) && z > min_depth_m && z <= max_depth_m)
                out.push_back({{u, v}, backproject(k, static_cast<double>(u), static_cast<double>(v), z)});
        }
    return out;
}

} // namespace operator_lab::rgbd
