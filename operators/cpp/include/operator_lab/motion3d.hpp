#pragma once
#include "rgbd_geometry.hpp"
#include <optional>

namespace operator_lab::rgbd {
struct MotionConfig {
    double measurement_std_m = .05, acceleration_std_mps2 = 1.;
    double initial_velocity_std_mps = 1., max_prediction_s = .6;
    double max_innovation_squared = 16.;
};
struct MotionPrediction {
    Point3 position_m, velocity_mps;
    std::array<double,3> variance_m2{};
    double stamp_s = 0, age_s = 0;
};
inline void validate(const MotionConfig& c) {
    for (double value : {c.measurement_std_m, c.acceleration_std_mps2,
                        c.initial_velocity_std_mps, c.max_prediction_s, c.max_innovation_squared})
        if (!std::isfinite(value) || value < 0 || value > 1e6)
            throw std::invalid_argument("invalid motion configuration");
    if (c.measurement_std_m < 1e-9 || c.initial_velocity_std_mps <= 0 ||
        c.max_prediction_s <= 0 || c.max_innovation_squared <= 0)
        throw std::invalid_argument("motion scale must be positive");
}

// Independent position/velocity Kalman axes. Points MUST share a fixed metric
// frame. Prediction is read-only and never prolongs the last measurement age.
class Motion3D {
public:
    explicit Motion3D(MotionConfig config = {}) : config_(config) { validate(config_); }
    void reset() { stamp_.reset(); axes_ = {}; }
    void initialize(Point3 point, double stamp_s) {
        if (!finite(point) || !std::isfinite(stamp_s)) throw std::invalid_argument("invalid motion initialization");
        const auto values = coordinates(point);
        for (std::size_t i = 0; i < 3; ++i)
            axes_[i] = {values[i], 0, square(config_.measurement_std_m), 0,
                         square(config_.initial_velocity_std_mps)};
        stamp_ = stamp_s;
    }
    std::optional<MotionPrediction> predict(double stamp_s) const {
        if (!std::isfinite(stamp_s)) throw std::invalid_argument("invalid prediction time");
        if (!stamp_) return std::nullopt;
        if (stamp_s < *stamp_) throw std::invalid_argument("prediction before last measurement");
        const double dt = stamp_s - *stamp_;
        if (dt > config_.max_prediction_s) return std::nullopt;
        const auto predicted = propagate(dt);
        MotionPrediction result;
        result.position_m = {predicted[0].p, predicted[1].p, predicted[2].p};
        result.velocity_mps = {predicted[0].v, predicted[1].v, predicted[2].v};
        result.variance_m2 = {predicted[0].pp, predicted[1].pp, predicted[2].pp};
        result.stamp_s = stamp_s; result.age_s = dt;
        return result;
    }
    bool update(Point3 point, double stamp_s) {
        if (!finite(point) || !std::isfinite(stamp_s)) throw std::invalid_argument("invalid motion observation");
        if (!stamp_) return false;
        if (stamp_s <= *stamp_) throw std::invalid_argument("motion timestamps must increase");
        const double dt = stamp_s - *stamp_;
        if (dt > config_.max_prediction_s) return false;
        auto predicted = propagate(dt);
        const auto observed = coordinates(point);
        const double r = square(config_.measurement_std_m);
        double innovation_squared = 0;
        for (std::size_t i = 0; i < 3; ++i) {
            const double standardized = (observed[i] - predicted[i].p) / std::sqrt(predicted[i].pp + r);
            innovation_squared += standardized * standardized;
        }
        if (!std::isfinite(innovation_squared) || innovation_squared > config_.max_innovation_squared) return false;
        for (std::size_t i = 0; i < 3; ++i) {
            const Axis a = predicted[i];
            const double kp = a.pp / (a.pp + r), kv = a.pv / (a.pp + r);
            const double residual = observed[i] - a.p;
            // Joseph covariance update reduces cancellation in long streams.
            predicted[i] = {a.p + kp * residual, a.v + kv * residual,
                square(1 - kp) * a.pp + square(kp) * r,
                (1 - kp) * (a.pv - kv * a.pp) + kp * kv * r,
                a.vv - 2 * kv * a.pv + square(kv) * (a.pp + r)};
            check(predicted[i]);
        }
        axes_ = predicted; stamp_ = stamp_s;
        return true;
    }
private:
    struct Axis { double p = 0, v = 0, pp = 0, pv = 0, vv = 0; };
    MotionConfig config_;
    std::optional<double> stamp_;
    std::array<Axis,3> axes_{};
    static double square(double value) { return value * value; }
    static std::array<double,3> coordinates(Point3 p) { return {p.x,p.y,p.z}; }
    static void check(const Axis& a) {
        for (double value : {a.p,a.v,a.pp,a.pv,a.vv})
            if (!std::isfinite(value)) throw std::overflow_error("motion state overflow");
        if (a.pp < 0 || a.vv < 0 || a.pv * a.pv > a.pp * a.vv * (1 + 1e-8) + 1e-18)
            throw std::runtime_error("invalid motion covariance");
    }
    std::array<Axis,3> propagate(double dt) const {
        auto result = axes_;
        const double q = square(config_.acceleration_std_mps2);
        const double dt2 = dt * dt;
        for (auto& a : result) {
            a = {a.p + a.v * dt, a.v,
                 a.pp + 2 * dt * a.pv + dt2 * a.vv + q * dt2 * dt2 / 4,
                 a.pv + dt * a.vv + q * dt2 * dt / 2,
                 a.vv + q * dt2};
            check(a);
        }
        return result;
    }
};
} // namespace operator_lab::rgbd
