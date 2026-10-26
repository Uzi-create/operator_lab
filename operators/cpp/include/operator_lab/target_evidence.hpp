#pragma once

#include "rgbd_geometry.hpp"
#include <optional>

namespace operator_lab::rgbd {

struct EvidenceConfig {
    std::size_t confirm_frames = 3;
    double min_duration_s = 0.06, max_age_s = 0.2, max_gap_s = 0.25, hold_timeout_s = 0.6;
    double max_speed_mps = 3., position_slack_m = 0.05;
};
struct TargetObservation {
    std::uint64_t target_key = 0;
    Point3 point_m;
    double stamp_s = 0;
};
enum class EvidenceState { lost, confirming, confirmed, missing, stale,
                           future_frame, out_of_order, different_target, position_jump };
struct EvidenceResult {
    EvidenceState state = EvidenceState::lost;
    bool ready = false;
    std::size_t confirmations = 0;
    std::optional<Point3> last_point_m;
};

inline void validate(const EvidenceConfig& c) {
    for (double v : {c.min_duration_s, c.max_age_s, c.max_gap_s, c.hold_timeout_s,
                     c.max_speed_mps, c.position_slack_m})
        if (!std::isfinite(v)) throw std::invalid_argument("nonfinite evidence configuration");
    if (!c.confirm_frames || c.min_duration_s < 0 || c.max_age_s <= 0 ||
        c.max_gap_s <= 0 || c.hold_timeout_s < c.max_gap_s ||
        c.max_speed_mps < 0 || c.position_slack_m < 0)
        throw std::invalid_argument("invalid evidence configuration");
}

// This is a confidence gate for ONE ALREADY ASSOCIATED target, not a tracker,
// re-identification algorithm, obstacle checker, predictor or safety controller.
// Points must share one metric frame across timestamps (e.g. odom). The caller
// must compensate camera motion BEFORE calling this class. All times use one
// monotonic clock; reset() on clock/mission/frame changes or deliberate reselection.
class TargetEvidence {
public:
    explicit TargetEvidence(EvidenceConfig config = {}) : config_(config) { validate(config_); }
    void reset() {
        last_now_.reset(); last_input_stamp_.reset(); observation_.reset();
        count_ = 0; started_s_ = 0;
    }
    EvidenceResult observe(std::optional<TargetObservation> observation, double now_s) {
        if (observation && (!std::isfinite(observation->stamp_s) || !finite(observation->point_m)))
            throw std::invalid_argument("nonfinite observation");
        advance(now_s);
        if (!observation) return reject(EvidenceState::missing);
        const auto& next = *observation;
        if (next.stamp_s > now_s) return reject(EvidenceState::future_frame);
        if (last_input_stamp_ && next.stamp_s <= *last_input_stamp_)
            return reject(EvidenceState::out_of_order);
        last_input_stamp_ = next.stamp_s;
        if (now_s - next.stamp_s > config_.max_age_s) return reject(EvidenceState::stale);
        if (observation_) {
            if (next.target_key != observation_->target_key)
                return reject(EvidenceState::different_target);
            const double dt = next.stamp_s - observation_->stamp_s;
            const Point3 delta{next.point_m.x - observation_->point_m.x,
                               next.point_m.y - observation_->point_m.y,
                               next.point_m.z - observation_->point_m.z};
            const double displacement = std::hypot(delta.x, delta.y, delta.z);
            const double allowed = config_.position_slack_m + config_.max_speed_mps * dt;
            if (!std::isfinite(displacement) || !std::isfinite(allowed) || displacement > allowed)
                return reject(EvidenceState::position_jump);
        }
        if (!observation_ || next.stamp_s - observation_->stamp_s > config_.max_gap_s || !count_) {
            count_ = 0; started_s_ = next.stamp_s;
        }
        observation_ = next;
        // Saturation preserves the "at least N" condition in long-lived streams.
        if (count_ < std::numeric_limits<std::size_t>::max()) ++count_;
        return current();
    }
    // Call on EVERY control tick, including when no new sensor callback arrives.
    // A returned result is only a snapshot; an old ready=true must not be cached.
    EvidenceResult poll(double now_s) {
        advance(now_s);
        if (!observation_) return {};
        if (now_s - observation_->stamp_s > config_.max_age_s) return reject(EvidenceState::stale);
        return current();
    }
private:
    EvidenceConfig config_;
    std::optional<double> last_now_, last_input_stamp_;
    std::optional<TargetObservation> observation_;
    std::size_t count_ = 0;
    double started_s_ = 0;

    void advance(double now_s) {
        if (!std::isfinite(now_s)) throw std::invalid_argument("nonfinite current time");
        if (last_now_ && now_s < *last_now_) {
            reset(); throw std::invalid_argument("clock moved backwards; evidence reset");
        }
        last_now_ = now_s;
        if (observation_ && now_s - observation_->stamp_s > config_.hold_timeout_s) {
            observation_.reset(); count_ = 0; started_s_ = 0;
        }
    }
    EvidenceResult reject(EvidenceState reason) {
        count_ = 0; started_s_ = 0;
        return {observation_ ? reason : EvidenceState::lost, false, 0,
                observation_ ? std::optional<Point3>(observation_->point_m) : std::nullopt};
    }
    EvidenceResult current() const {
        if (!observation_) return {};
        const bool ready = count_ >= config_.confirm_frames &&
            observation_->stamp_s - started_s_ >= config_.min_duration_s;
        return {ready ? EvidenceState::confirmed : EvidenceState::confirming, ready, count_,
                observation_->point_m};
    }
};

} // namespace operator_lab::rgbd
