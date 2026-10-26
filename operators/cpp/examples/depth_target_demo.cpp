#include <operator_lab/depth_target.hpp>
#include <operator_lab/target_evidence.hpp>
#include <iostream>

int main() {
    using namespace operator_lab::rgbd;
    // Synthetic registered depth ROI: half foreground at 1 m, half target at 2 m.
    std::vector<std::uint16_t> depth_mm(100, 2000);
    std::fill(depth_mm.begin(), depth_mm.begin() + 50, 1000);
    const DepthView<std::uint16_t> depth(depth_mm.data(), depth_mm.size(), 10, 10, .001);
    const Intrinsics camera{10,10,100,100,4.5,4.5};
    const auto ambiguous = estimate_depth_target(depth, camera, {0,0,10,10});
    const auto selected = estimate_depth_target(depth, camera, {0,0,10,10}, {}, DepthPrior{2., .2});
    if (ambiguous.state != DepthState::ambiguous || !selected.accepted()) return 1;
    std::cout << "No prior: ambiguous, no target point returned\n"
              << "Explicit current-depth prior: " << *selected.depth_m << " m, "
              << selected.support_count << " supporting pixels\n";
    // Already associated observations in one fixed metric frame, not raw camera frames.
    TargetEvidence evidence;
    for (int frame = 0; frame < 3; ++frame) {
        const double time = frame * .1;
        const auto state = evidence.observe(TargetObservation{42, {frame * .05, 0, 2}, time}, time);
        std::cout << "frame " << frame << ": ready=" << state.ready << '\n';
    }
    const auto stale = evidence.poll(.5);
    std::cout << "Sensor silence: ready=" << stale.ready << '\n';
    return stale.ready ? 1 : 0;
}
