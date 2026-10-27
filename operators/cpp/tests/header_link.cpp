#include <operator_lab/depth_target.hpp>
#include <operator_lab/target_evidence.hpp>
#include <operator_lab/rgbd_tracker.hpp>
#include <operator_lab/replay_io.hpp>
#include <operator_lab/tracking_navigation.hpp>
#include <operator_lab/laser_scan_io.hpp>

bool linked_geometry_check() {
    using namespace operator_lab::rgbd;
    const auto point = backproject({1,1,1,1,0,0}, 0, 0, 2);
    TargetEvidence gate;
    return point.z == 2 && !gate.poll(0).ready;
}
