#include <operator_lab/depth_target.hpp>
#include <operator_lab/target_evidence.hpp>
#include <algorithm>
#include <iostream>
#include <random>
#include <string>

using namespace operator_lab::rgbd;
int checks = 0;
#define CHECK(condition) do { ++checks; if (!(condition)) throw std::runtime_error( \
    std::string(__FILE__) + ":" + std::to_string(__LINE__) + " " #condition); } while (false)
template<class F> void throws(F function) {
    bool caught = false;
    try { function(); } catch (const std::exception&) { caught = true; }
    CHECK(caught);
}
bool near(double a, double b, double tolerance = 1e-10) { return std::abs(a - b) <= tolerance; }
bool linked_geometry_check(); // Defined in a second TU: detect header ODR violations.

void geometry_tests() {
    const Intrinsics k{4, 3, 2, 4, 1, 1};
    const auto p = backproject(k, 3, 2, 2);
    CHECK(near(p.x, 2) && near(p.y, .5) && near(p.z, 2));
    RigidTransform t{{0, -1, 0, 1, 0, 0, 0, 0, 1}, {3, 4, 5}};
    const auto q = transform_point(p, t);
    CHECK(near(q.x, 2.5) && near(q.y, 6) && near(q.z, 7));
    RigidTransform inverse{{0, 1, 0, -1, 0, 0, 0, 0, 1}, {-4, 3, -5}};
    const auto recovered = transform_point(q, inverse);
    CHECK(near(recovered.x, p.x) && near(recovered.y, p.y) && near(recovered.z, p.z));
    throws([&] { backproject(k, -1, 0, 1); });
    throws([&] { backproject(k, 4, 0, 1); });
    throws([&] { backproject(k, 0, 0, 0); });
    throws([&] { backproject({4, 3, 0, 4, 1, 1}, 0, 0, 1); });
    auto invalid = t; invalid.rotation[0] = 2;
    throws([&] { transform_point(p, invalid); });
    invalid = {}; invalid.rotation[0] = -1;
    throws([&] { transform_point(p, invalid); }); // Reflection is not a rigid rotation.
    invalid = {}; invalid.translation_m.x = std::numeric_limits<double>::infinity();
    throws([&] { transform_point(p, invalid); });
    const double huge = std::numeric_limits<double>::max();
    throws([&] { backproject({2, 1, 1e-300, 1, 0, 0}, 1, 0, huge); });
    throws([&] { transform_point({huge, 0, 0}, RigidTransform{{1,0,0,0,1,0,0,0,1}, {huge,0,0}}); });

    // Padding must not be interpreted as image pixels; the final row has no padding.
    std::vector<std::uint16_t> mm{1000, 2000, 9000, 9000, 3000, 0};
    DepthView<std::uint16_t> view(mm.data(), mm.size(), 2, 2, .001, 4);
    CHECK(near(view.at_m(1, 0), 2) && near(view.at_m(0, 1), 3));
    CHECK(std::isnan(view.at_m(1, 1)));
    auto cloud = depth_to_points(view, {2, 2, 2, 2, 0, 0}, {0, 0, 2, 2}, 1, 3);
    CHECK(cloud.size() == 2); // Strict lower bound and inclusive upper bound.
    CHECK(cloud[0].pixel.u == 1 && cloud[0].pixel.v == 0 && near(cloud[0].point_m.x, 1));
    CHECK(cloud[1].pixel.u == 0 && cloud[1].pixel.v == 1 && near(cloud[1].point_m.y, 1.5));
    cloud = depth_to_points(view, {2, 2, 2, 2, 0, 0}, {1, 0, 1, 2}, 0, 3, 2);
    CHECK(cloud.size() == 1 && cloud[0].pixel.u == 1);
    throws([&] { DepthView<std::uint16_t>(mm.data(), 5, 2, 2, .001, 4); });
    throws([&] { DepthView<std::uint16_t>(mm.data(), mm.size(), 2, 2, .001, 1); });
    throws([&] { DepthView<std::uint16_t>(nullptr, 4, 2, 2, .001); });
    throws([&] { DepthView<std::uint16_t>(mm.data(), mm.size(), 2, 2, 0); });
    throws([&] { view.at_m(2, 0); });
    throws([&] { depth_to_points(view, k, {0, 0, 2, 2}); });
    throws([&] { depth_to_points(view, {2,2,1,1,0,0}, {1,0,2,2}); });
    throws([&] { validate(Rect{std::numeric_limits<std::size_t>::max(),0,1,1}, 2, 2); });

    const float nan = std::numeric_limits<float>::quiet_NaN();
    std::vector<float> floating{1, nan, -1, std::numeric_limits<float>::infinity()};
    auto sparse = depth_to_points(DepthView<float>(floating.data(), 4, 2, 2, 1),
                                 {2,2,1,1,0,0}, {0,0,2,2});
    CHECK(sparse.size() == 1);

    // Independent closed-form reference over randomized pixels and depths.
    std::mt19937 random(42);
    for (int iteration = 0; iteration < 150; ++iteration) {
        const double z = 0.5 + (random() % 5000) / 1000.;
        const double u = random() % 640, v = random() % 480;
        const auto point = backproject({640,480,500,400,320,240}, u, v, z);
        CHECK(near(point.x * 500 / point.z + 320, u));
        CHECK(near(point.y * 400 / point.z + 240, v));
    }
    CHECK(linked_geometry_check());
}

DepthTargetResult estimate(std::vector<double>& values, DepthTargetConfig config = {},
                           std::optional<DepthPrior> prior = std::nullopt) {
    return estimate_depth_target(DepthView<double>(values.data(), values.size(), 10, 10, 1),
                                 {10,10,100,100,4.5,4.5}, {0,0,10,10}, config, prior);
}

void depth_tests() {
    std::vector<double> values(100, 2.);
    auto result = estimate(values);
    CHECK(result.accepted() && result.valid_count == 100 && result.support_count == 100);
    CHECK(near(*result.depth_m, 2) && near(*result.mad_m, 0));
    CHECK(near(result.point_m->x, 0) && near(result.point_m->y, 0));
    CHECK(result.support_pixels.size() == 100);
    std::fill(values.begin(), values.begin() + 20, 4.);
    result = estimate(values);
    CHECK(result.accepted() && near(*result.depth_m, 2) && result.support_count == 80);
    CHECK(near(result.runner_ratio, .25));
    std::fill(values.begin(), values.begin() + 50, 1.);
    result = estimate(values);
    CHECK(result.state == DepthState::ambiguous);
    CHECK(!result.point_m && !result.depth_m && result.support_pixels.empty());
    result = estimate(values, {}, DepthPrior{2, .2});
    CHECK(result.accepted() && near(*result.depth_m, 2) && result.support_count == 50);
    result = estimate(values, {}, DepthPrior{3, .1});
    CHECK(result.state == DepthState::no_prior_support && !result.point_m);

    // Dominant foreground is not automatically the original target. An explicit
    // current-depth prior can retain the smaller target band, if enough support remains.
    std::fill(values.begin(), values.begin() + 60, 1.);
    result = estimate(values, {}, DepthPrior{2, .1});
    CHECK(result.accepted() && near(*result.depth_m, 2) && result.support_count == 40);
    std::fill(values.begin(), values.begin() + 70, 1.);
    result = estimate(values, {}, DepthPrior{2, .1});
    CHECK(result.state == DepthState::insufficient_support);
    std::fill(values.begin(), values.end(), 0.);
    CHECK(estimate(values).state == DepthState::insufficient_valid);
    std::fill(values.begin(), values.begin() + 20, 2.);
    result = estimate(values);
    CHECK(result.state == DepthState::insufficient_valid && near(result.valid_fraction, .2));
    values.assign(100, std::numeric_limits<double>::quiet_NaN());
    CHECK(!estimate(values).accepted());
    values.assign(100, std::numeric_limits<double>::infinity());
    CHECK(!estimate(values).accepted());

    values.assign(100, 2.); std::fill(values.begin(), values.begin() + 50, 2.1);
    DepthTargetConfig config; config.max_mad_m = .01;
    CHECK(estimate(values, config).state == DepthState::excessive_spread);
    config = {}; config.band_width_m = 0;
    throws([&] { estimate(values, config); });
    throws([&] { estimate(values, {}, DepthPrior{2, -1}); });
    throws([&] { estimate_depth_target(DepthView<double>(values.data(),100,10,10,1),
        {10,10,100,100,5,5}, {9,9,2,2}); });

    // Full-span windows must not merge a long gradually changing ramp.
    for (std::size_t i = 0; i < values.size(); ++i) values[i] = 1 + i * .01;
    config = {}; config.band_width_m = .051; config.min_samples = 2;
    config.min_support_fraction = 0;
    CHECK(estimate(values, config).state == DepthState::ambiguous);

    // Exhaustive O(N^2) reference for densest-band support, including ties.
    std::mt19937 random(7);
    for (int iteration = 0; iteration < 120; ++iteration) {
        for (auto& value : values) value = .5 + (random() % 50) * .05;
        config = {}; config.band_width_m = .151; config.min_samples = 1;
        config.min_support_fraction = 0; config.max_mad_m = 1;
        auto sorted = values; std::sort(sorted.begin(), sorted.end());
        std::size_t best = 0;
        for (double low : sorted) {
            std::size_t count = 0;
            for (double value : sorted) if (value >= low && value - low <= config.band_width_m) ++count;
            best = std::max(best, count);
        }
        CHECK(estimate(values, config).support_count == best);
    }

    // Unit conversion and non-contiguous rows must produce the same geometry.
    std::vector<std::uint16_t> mm(120, 9999);
    for (std::size_t v = 0; v < 10; ++v)
        for (std::size_t u = 0; u < 10; ++u) mm[v * 12 + u] = 2000;
    result = estimate_depth_target(DepthView<std::uint16_t>(mm.data(),mm.size(),10,10,.001,12),
        {10,10,100,100,4.5,4.5}, {2,3,4,5});
    CHECK(result.accepted() && result.support_count == 20 && near(*result.depth_m, 2));
    CHECK(result.support_pixels.front().u == 2 && result.support_pixels.front().v == 3);
    CHECK(near(result.point_m->x, -.02) && near(result.point_m->y, .01));
}

TargetObservation observation(double time, double x = 0, std::uint64_t key = 7) {
    return {key, {x, 0, 2}, time};
}

void evidence_tests() {
    TargetEvidence gate;
    CHECK(!gate.poll(0).ready);
    CHECK(!gate.observe(observation(0), 0).ready);
    CHECK(!gate.observe(observation(.1, .1), .1).ready);
    auto result = gate.observe(observation(.2, .2), .2);
    CHECK(result.ready && result.confirmations == 3);
    CHECK(gate.poll(.21).ready);
    CHECK(!gate.poll(.41).ready); // No callback needed for timeout protection.
    CHECK(!gate.observe(observation(.42, .22), .42).ready);
    CHECK(!gate.observe(observation(.52, .24), .52).ready);
    CHECK(gate.observe(observation(.62, .26), .62).ready);
    result = gate.observe(observation(.72, .27, 8), .72);
    CHECK(result.state == EvidenceState::different_target && !result.ready);
    result = gate.observe(observation(.82, 10), .82);
    CHECK(result.state == EvidenceState::position_jump && !result.ready);
    result = gate.observe(std::nullopt, .85);
    CHECK(result.state == EvidenceState::missing && result.last_point_m && !result.ready);
    CHECK(gate.poll(1.3).state == EvidenceState::lost);
    CHECK(!gate.observe(observation(1.31, 10, 8), 1.31).ready); // New identity after expiry.

    gate.reset();
    gate.observe(observation(1), 1);
    result = gate.observe(observation(1), 1.1);
    CHECK(result.state == EvidenceState::out_of_order && result.confirmations == 0);
    gate.observe(observation(1.2), 1.2);
    result = gate.observe(observation(1.1), 1.3);
    CHECK(result.state == EvidenceState::out_of_order && !result.ready);
    result = gate.observe(observation(2), 1.4);
    CHECK(result.state == EvidenceState::future_frame && !result.ready);
    CHECK(!gate.observe(observation(1.41), 1.41).ready); // Future frame did not poison watermark.
    result = gate.observe(observation(1.5), 1.75);
    CHECK(result.state == EvidenceState::stale && !result.ready);
    throws([&] { gate.poll(1.); });
    CHECK(gate.poll(0).state == EvidenceState::lost); // Clock rewind cleared all state.
    throws([&] { gate.poll(std::numeric_limits<double>::quiet_NaN()); });
    throws([&] { gate.observe({TargetObservation{1, {0,0,std::numeric_limits<double>::infinity()}, 0}}, 0); });
    EvidenceConfig invalid; invalid.max_gap_s = 1;
    throws([&] { TargetEvidence bad(invalid); });

    // Equivalent motion at different frame rates uses dt, not a fixed distance/frame.
    for (double dt : {.02, .05, .1}) {
        EvidenceConfig c; c.min_duration_s = 0; c.max_speed_mps = 1; c.position_slack_m = .001;
        TargetEvidence moving(c);
        for (int frame = 0; frame < 20; ++frame) {
            const double stamp = frame * dt;
            result = moving.observe(observation(stamp, .8 * stamp), stamp);
            CHECK(result.ready == (frame >= 2));
        }
    }
    // A camera translating toward a stationary world target must be transformed
    // before gating. This checks the geometry + temporal contract together.
    EvidenceConfig stationary; stationary.max_speed_mps = 0; stationary.position_slack_m = 1e-8;
    TargetEvidence world_gate(stationary);
    for (int frame = 0; frame < 5; ++frame) {
        const double camera_x = frame * .2, stamp = frame * .1;
        const Point3 camera_point{1 - camera_x, 0, 2};
        const auto world_point = transform_point(camera_point,
            RigidTransform{{1,0,0,0,1,0,0,0,1}, {camera_x,0,0}});
        result = world_gate.observe(TargetObservation{7, world_point, stamp}, stamp);
        CHECK(result.ready == (frame >= 2));
    }
    // A missed frame immediately breaks confirmation even while the last point is retained.
    result = world_gate.observe(std::nullopt, .45);
    CHECK(!result.ready && result.last_point_m);
    CHECK(!world_gate.poll(.46).ready);
}

int main() {
    try {
        geometry_tests(); depth_tests(); evidence_tests();
        std::cout << "PASS: " << checks << " C++ RGB-D checks (geometry, depth bands, temporal evidence)\n";
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n'; return 1;
    }
}
