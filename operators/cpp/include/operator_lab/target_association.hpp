#pragma once
#include "rgbd_geometry.hpp"
#include <algorithm>
#include <optional>

namespace operator_lab::rgbd {
enum class ChannelOrder { rgb, bgr };
class RgbView {
public:
    RgbView(const std::uint8_t* data, std::size_t bytes, std::size_t width, std::size_t height,
            std::size_t stride_bytes = 0, ChannelOrder order = ChannelOrder::rgb)
        : data_(data), width_(width), height_(height),
          stride_(stride_bytes ? stride_bytes : width * 3), order_(order) {
        if (!data || !width || !height || width > 16384 || height > 16384 ||
            width > 16000000 / height || stride_ < width * 3 || bytes < width * 3 ||
            height - 1 > (bytes - width * 3) / stride_ ||
            (order != ChannelOrder::rgb && order != ChannelOrder::bgr))
            throw std::invalid_argument("invalid RGB buffer or stride");
    }
    std::size_t width() const { return width_; }
    std::size_t height() const { return height_; }
    std::array<std::uint8_t,3> at(std::size_t u, std::size_t v) const {
        if (u >= width_ || v >= height_) throw std::out_of_range("RGB pixel outside image");
        const auto* pixel = data_ + v * stride_ + u * 3;
        return order_ == ChannelOrder::rgb ? std::array<std::uint8_t,3>{pixel[0],pixel[1],pixel[2]}
                                          : std::array<std::uint8_t,3>{pixel[2],pixel[1],pixel[0]};
    }
private:
    const std::uint8_t* data_;
    std::size_t width_, height_, stride_;
    ChannelOrder order_;
};

// Interpretable appearance baseline; replaceable by externally computed embeddings.
// Samples only the spatial region, not its full rectangular background.
inline std::vector<double> color_descriptor(const RgbView& image, const std::vector<Pixel>& pixels,
                                            std::size_t bins_per_channel = 4) {
    if (pixels.empty() || bins_per_channel < 1 || bins_per_channel > 16)
        throw std::invalid_argument("invalid histogram support or bin count");
    std::vector<double> histogram(bins_per_channel * bins_per_channel * bins_per_channel, 0);
    for (Pixel pixel : pixels) {
        const auto rgb = image.at(pixel.u, pixel.v);
        const auto r = rgb[0] * bins_per_channel / 256;
        const auto g = rgb[1] * bins_per_channel / 256;
        const auto b = rgb[2] * bins_per_channel / 256;
        ++histogram[(r * bins_per_channel + g) * bins_per_channel + b];
    }
    for (auto& value : histogram) value /= static_cast<double>(pixels.size());
    return histogram;
}
struct TrackCandidate {
    std::size_t candidate_id = 0;
    Point3 position_m;
    std::vector<double> appearance;
};
struct AssociationQuery {
    Point3 predicted_position_m;
    std::array<double,3> position_variance_m2{};
    std::vector<double> reference_appearance;
};
struct AssociationConfig {
    double measurement_std_m = .08, max_mahalanobis_squared = 16.;
    double max_appearance_distance = .3;
    double geometry_weight = 1., appearance_weight = 1., min_score_margin = .1;
};
enum class AssociationState { matched, no_match, ambiguous };
struct AssociationResult {
    AssociationState state = AssociationState::no_match;
    std::optional<std::size_t> candidate_index;
    double best_score = std::numeric_limits<double>::infinity();
    double second_score = std::numeric_limits<double>::infinity();
};
inline void validate(const AssociationConfig& c) {
    for (double value : {c.measurement_std_m,c.max_mahalanobis_squared,c.max_appearance_distance,
                        c.geometry_weight,c.appearance_weight,c.min_score_margin})
        if (!std::isfinite(value) || value < 0 || value > 1e6)
            throw std::invalid_argument("invalid association configuration");
    if (c.measurement_std_m < 1e-9 || c.max_mahalanobis_squared <= 0 ||
        c.max_appearance_distance <= 0 || c.max_appearance_distance > 2 ||
        c.geometry_weight + c.appearance_weight <= 0)
        throw std::invalid_argument("invalid association scale");
}
namespace association_detail {
inline std::optional<std::vector<double>> unit_descriptor(const std::vector<double>& descriptor) {
    double scale = 0;
    for (double value : descriptor) {
        if (!std::isfinite(value)) return std::nullopt;
        scale = std::max(scale, std::abs(value));
    }
    if (scale == 0) return std::nullopt;
    std::vector<double> unit = descriptor;
    double norm = 0;
    for (double& value : unit) { value /= scale; norm += value * value; }
    norm = std::sqrt(norm);
    for (double& value : unit) value /= norm;
    return unit;
}
}

// Invalid candidate observations are skipped. Invalid query/configuration throws.
// Distances and a score margin are gates, not a calibrated probability of identity.
inline AssociationResult associate_target(const std::vector<TrackCandidate>& candidates,
    const AssociationQuery& query, const AssociationConfig& config = {}) {
    validate(config);
    const auto reference = association_detail::unit_descriptor(query.reference_appearance);
    if (!finite(query.predicted_position_m) || !reference)
        throw std::invalid_argument("invalid association query");
    for (double variance : query.position_variance_m2)
        if (!std::isfinite(variance) || variance < 0)
            throw std::invalid_argument("invalid predicted variance");
    AssociationResult result;
    std::optional<std::size_t> best_index;
    for (std::size_t index = 0; index < candidates.size(); ++index) {
        const auto& candidate = candidates[index];
        if (!finite(candidate.position_m) || candidate.appearance.size() != reference->size()) continue;
        const auto appearance = association_detail::unit_descriptor(candidate.appearance);
        if (!appearance) continue;
        double dot = 0;
        for (std::size_t j = 0; j < reference->size(); ++j) dot += (*reference)[j] * (*appearance)[j];
        const double appearance_distance = 1 - std::clamp(dot, -1., 1.);
        if (appearance_distance > config.max_appearance_distance) continue;
        const std::array<double,3> delta{candidate.position_m.x - query.predicted_position_m.x,
            candidate.position_m.y - query.predicted_position_m.y,
            candidate.position_m.z - query.predicted_position_m.z};
        double geometry_distance = 0;
        for (std::size_t j = 0; j < 3; ++j) {
            const double d = delta[j] / std::sqrt(query.position_variance_m2[j] +
                                                 config.measurement_std_m * config.measurement_std_m);
            geometry_distance += d * d;
        }
        if (!std::isfinite(geometry_distance) || geometry_distance > config.max_mahalanobis_squared) continue;
        const double score = (config.geometry_weight * geometry_distance / config.max_mahalanobis_squared +
            config.appearance_weight * appearance_distance / config.max_appearance_distance) /
            (config.geometry_weight + config.appearance_weight);
        if (score < result.best_score) {
            result.second_score = result.best_score; result.best_score = score; best_index = index;
        } else if (score < result.second_score) result.second_score = score;
    }
    if (!best_index) return result;
    if (result.second_score - result.best_score < config.min_score_margin ||
        result.second_score == result.best_score) {
        result.state = AssociationState::ambiguous; return result;
    }
    result.state = AssociationState::matched; result.candidate_index = best_index;
    return result;
}
} // namespace operator_lab::rgbd
