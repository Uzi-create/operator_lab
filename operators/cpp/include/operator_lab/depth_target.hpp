#pragma once

#include "rgbd_geometry.hpp"
#include <algorithm>
#include <optional>

namespace operator_lab::rgbd {

struct DepthTargetConfig {
    double min_depth_m = 0.2, max_depth_m = 8.;
    double band_width_m = 0.12; // FULL span, not a radius or connectivity threshold
    std::size_t min_samples = 12;
    double min_valid_fraction = 0.25, min_support_fraction = 0.35;
    double ambiguity_ratio = 0.8, max_mad_m = 0.06;
};

// A caller-supplied prediction of CURRENT optical Z, with an uncertainty gate.
// It is not an identity cue. Ego motion and prediction age are caller concerns.
struct DepthPrior { double depth_m = 0, tolerance_m = 0; };
enum class DepthState { accepted, insufficient_valid, no_prior_support,
                        insufficient_support, ambiguous, excessive_spread };

struct DepthTargetResult {
    DepthState state = DepthState::insufficient_valid;
    std::size_t roi_area = 0, valid_count = 0, support_count = 0;
    double valid_fraction = 0, support_fraction = 0, runner_ratio = 0;
    std::optional<double> depth_m, mad_m;
    std::optional<Point3> point_m;
    std::vector<Pixel> support_pixels;
    bool accepted() const { return state == DepthState::accepted; }
};

inline void validate(const DepthTargetConfig& c) {
    for (double v : {c.min_depth_m, c.max_depth_m, c.band_width_m,
                     c.min_valid_fraction, c.min_support_fraction, c.ambiguity_ratio, c.max_mad_m})
        if (!std::isfinite(v)) throw std::invalid_argument("nonfinite depth-target configuration");
    if (c.min_depth_m < 0 || c.max_depth_m <= c.min_depth_m || c.band_width_m <= 0 ||
        !c.min_samples || c.min_valid_fraction < 0 || c.min_valid_fraction > 1 ||
        c.min_support_fraction < 0 || c.min_support_fraction > 1 ||
        c.ambiguity_ratio <= 0 || c.ambiguity_ratio > 1 || c.max_mad_m < 0)
        throw std::invalid_argument("invalid depth-target configuration");
}

namespace detail {
struct DepthPixel { double z; Pixel pixel; };
struct Window { std::size_t begin = 0, end = 0; std::size_t size() const { return end - begin; } };
inline Window densest_window(const std::vector<DepthPixel>& samples, std::size_t begin,
                            std::size_t end, double span) {
    Window best{begin, begin};
    std::size_t right = begin;
    for (std::size_t left = begin; left < end; ++left) {
        while (right < end && samples[right].z - samples[left].z <= span) ++right;
        if (right - left > best.size()) best = {left, right};
    }
    return best;
}
inline double sorted_median(const std::vector<double>& values) {
    const std::size_t n = values.size();
    return n % 2 ? values[n / 2] : values[n / 2 - 1] / 2 + values[n / 2] / 2;
}
} // namespace detail

// No segmentation, identity association or motion command is implicit here.
// All pixels in ROI are considered. RGB ROI and depth MUST already be registered.
// A prior restricts candidate depths before peak selection. Two strong disjoint
// bands cause rejection, rather than silently choosing the nearer/larger person.
template<class T>
inline DepthTargetResult estimate_depth_target(const DepthView<T>& depth, const Intrinsics& k,
    Rect roi, const DepthTargetConfig& config = {}, std::optional<DepthPrior> prior = std::nullopt) {
    validate(k); validate(config); validate(roi, depth.width(), depth.height());
    if (k.width != depth.width() || k.height != depth.height())
        throw std::invalid_argument("depth and intrinsics dimensions differ");
    if (prior && (!std::isfinite(prior->depth_m) || prior->depth_m <= 0 ||
                  !std::isfinite(prior->tolerance_m) || prior->tolerance_m < 0))
        throw std::invalid_argument("invalid depth prior");
    DepthTargetResult result;
    result.roi_area = roi.width * roi.height;
    std::vector<detail::DepthPixel> candidates;
    candidates.reserve(result.roi_area);
    for (std::size_t v = roi.y; v < roi.y + roi.height; ++v)
        for (std::size_t u = roi.x; u < roi.x + roi.width; ++u) {
            const double z = depth.at_m(u, v);
            if (!std::isfinite(z) || z <= config.min_depth_m || z > config.max_depth_m) continue;
            ++result.valid_count;
            if (!prior || std::abs(z - prior->depth_m) <= prior->tolerance_m)
                candidates.push_back({z, {u, v}});
        }
    result.valid_fraction = static_cast<double>(result.valid_count) / result.roi_area;
    if (result.valid_count < config.min_samples || result.valid_fraction < config.min_valid_fraction)
        return result;
    if (candidates.empty()) { result.state = DepthState::no_prior_support; return result; }
    std::stable_sort(candidates.begin(), candidates.end(),
                     [](const auto& a, const auto& b) { return a.z < b.z; });
    const auto best = detail::densest_window(candidates, 0, candidates.size(), config.band_width_m);
    const auto left = detail::densest_window(candidates, 0, best.begin, config.band_width_m);
    const auto right = detail::densest_window(candidates, best.end, candidates.size(), config.band_width_m);
    result.support_count = best.size();
    result.support_fraction = static_cast<double>(best.size()) / result.valid_count;
    result.runner_ratio = static_cast<double>(std::max(left.size(), right.size())) / best.size();
    if (best.size() < config.min_samples || result.support_fraction < config.min_support_fraction) {
        result.state = DepthState::insufficient_support; return result;
    }
    if (result.runner_ratio >= config.ambiguity_ratio) {
        result.state = DepthState::ambiguous; return result;
    }
    std::vector<double> z;
    z.reserve(best.size());
    double mean_u = 0, mean_v = 0;
    for (std::size_t i = best.begin; i < best.end; ++i) {
        z.push_back(candidates[i].z);
        const double count = static_cast<double>(i - best.begin + 1);
        mean_u += (static_cast<double>(candidates[i].pixel.u) - mean_u) / count;
        mean_v += (static_cast<double>(candidates[i].pixel.v) - mean_v) / count;
    }
    const double median = detail::sorted_median(z);
    for (double& value : z) value = std::abs(value - median);
    auto middle = z.begin() + static_cast<std::ptrdiff_t>(z.size() / 2);
    std::nth_element(z.begin(), middle, z.end());
    const double mad = z.size() % 2 ? *middle : *std::max_element(z.begin(), middle) / 2 + *middle / 2;
    if (mad > config.max_mad_m) { result.state = DepthState::excessive_spread; return result; }
    result.point_m = backproject(k, mean_u, mean_v, median);
    result.depth_m = median; result.mad_m = mad;
    result.support_pixels.reserve(best.size());
    for (std::size_t i = best.begin; i < best.end; ++i)
        result.support_pixels.push_back(candidates[i].pixel);
    result.state = DepthState::accepted;
    return result;
}

} // namespace operator_lab::rgbd
