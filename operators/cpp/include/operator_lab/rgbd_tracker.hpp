#pragma once
#include "depth_regions.hpp"
#include "motion3d.hpp"
#include "target_association.hpp"
#include "target_evidence.hpp"
#include <string>

namespace operator_lab::rgbd {
struct TrackerConfig {
    DepthRegionConfig regions;
    MotionConfig motion;
    AssociationConfig association;
    EvidenceConfig evidence;
    double max_frame_age_s = .2, max_rgb_depth_skew_s = .03, max_pose_skew_s = .03;
    double max_rgb_neighbor_distance = 120.; // L2 distance in 8-bit RGB units.
};
struct FrameTimes {
    double rgb_stamp_s = 0, depth_stamp_s = 0, pose_stamp_s = 0, now_s = 0;
};
enum class TrackState { waiting_selection, confirming, tracking, occluded, lost, rejected_frame };
inline const char* state_name(TrackState state) {
    switch (state) {
    case TrackState::waiting_selection: return "waiting_selection";
    case TrackState::confirming: return "confirming";
    case TrackState::tracking: return "tracking";
    case TrackState::occluded: return "occluded";
    case TrackState::lost: return "lost";
    case TrackState::rejected_frame: return "rejected_frame";
    }
    return "unknown";
}
struct TrackerResult {
    TrackState state = TrackState::waiting_selection;
    bool confirmed = false, fresh_observation = false;
    std::optional<Point3> observed_position_m;
    std::optional<MotionPrediction> prediction;
    std::optional<Rect> bounds;
    std::optional<Pixel> representative_pixel;
    std::size_t region_count = 0;
    std::string reason;
};

// Classical RGB-D baseline: spatial depth regions + anchored color appearance +
// 3D motion association. No human detector, learned re-ID or motion control.
class RgbdTracker {
public:
    explicit RgbdTracker(TrackerConfig config = {})
        : config_(config), motion_(config.motion), evidence_(config.evidence) {
        validate(config_.regions); validate(config_.association);
        for (double value : {config_.max_frame_age_s,config_.max_rgb_depth_skew_s,config_.max_pose_skew_s})
            if (!std::isfinite(value) || value < 0) throw std::invalid_argument("invalid frame timing tolerance");
        if (config_.max_frame_age_s <= 0) throw std::invalid_argument("frame age must be positive");
        if (!std::isfinite(config_.max_rgb_neighbor_distance) || config_.max_rgb_neighbor_distance < 0 ||
            config_.max_rgb_neighbor_distance > 442.) throw std::invalid_argument("invalid RGB boundary threshold");
    }
    void reset() {
        motion_.reset(); evidence_.reset(); reference_.clear(); selected_ = false;
        last_frame_stamp_.reset(); last_now_.reset();
    }
    TrackerResult poll(double now_s) {
        check_now(now_s);
        if (!selected_) return {};
        TrackerResult out;
        out.prediction = motion_.predict(now_s);
        const auto evidence = evidence_.poll(now_s);
        out.confirmed = evidence.ready && out.prediction.has_value();
        out.state = !out.prediction ? TrackState::lost :
                    (out.confirmed ? TrackState::tracking : TrackState::occluded);
        out.reason = out.confirmed ? "recent_confirmed_observation" :
                     (out.prediction ? "no_fresh_confirmed_observation" : "prediction_expired_reselect_required");
        return out;
    }

    template<class T>
    TrackerResult process(const RgbView& rgb, const DepthView<T>& depth, const Intrinsics& k,
        const RigidTransform& fixed_from_camera, FrameTimes time,
        std::optional<Pixel> select_pixel = std::nullopt,
        std::optional<MaskView> candidate_mask = std::nullopt) {
        validate(k); validate(fixed_from_camera);
        if (rgb.width() != depth.width() || rgb.height() != depth.height())
            throw std::invalid_argument("registered RGB/depth dimensions must match");
        check_now(time.now_s);
        for (double stamp : {time.rgb_stamp_s,time.depth_stamp_s,time.pose_stamp_s})
            if (!std::isfinite(stamp)) return rejected(time.now_s, "nonfinite_timestamp");
        if (time.rgb_stamp_s > time.now_s || time.depth_stamp_s > time.now_s || time.pose_stamp_s > time.now_s)
            return rejected(time.now_s, "future_frame_or_pose");
        if (time.now_s - std::min(time.rgb_stamp_s, time.depth_stamp_s) > config_.max_frame_age_s)
            return rejected(time.now_s, "stale_frame");
        if (std::abs(time.rgb_stamp_s - time.depth_stamp_s) > config_.max_rgb_depth_skew_s)
            return rejected(time.now_s, "rgb_depth_skew");
        if (std::abs(time.pose_stamp_s - time.depth_stamp_s) > config_.max_pose_skew_s)
            return rejected(time.now_s, "pose_time_skew");
        if (last_frame_stamp_ && time.depth_stamp_s <= *last_frame_stamp_)
            return rejected(time.now_s, "out_of_order_frame");
        last_frame_stamp_ = time.depth_stamp_s;
        const auto color_connected = [&](Pixel a,Pixel b) {
            const auto first=rgb.at(a.u,a.v),second=rgb.at(b.u,b.v);
            double distance_squared=0;
            for(std::size_t c=0;c<3;++c) {
                const double difference=static_cast<int>(first[c])-static_cast<int>(second[c]);
                distance_squared+=difference*difference;
            }
            return distance_squared<=config_.max_rgb_neighbor_distance*config_.max_rgb_neighbor_distance;
        };
        const auto regions = extract_depth_regions(depth, k, {0,0,depth.width(),depth.height()},
                                                   config_.regions, candidate_mask,color_connected,&region_workspace_);
        std::vector<TrackCandidate> candidates;
        candidates.reserve(regions.size());
        for (const auto& region : regions)
            candidates.push_back({region.label, transform_point(region.point_m, fixed_from_camera),
                                  color_descriptor(rgb, region.pixels)});
        if (select_pixel) {
            motion_.reset(); evidence_.reset(); reference_.clear(); selected_ = false;
            for (std::size_t index = 0; index < regions.size(); ++index)
                for (Pixel pixel : regions[index].pixels)
                    if (pixel.u == select_pixel->u && pixel.v == select_pixel->v) {
                        selected_ = true; reference_ = candidates[index].appearance;
                        motion_.initialize(candidates[index].position_m, time.depth_stamp_s);
                        const auto evidence = evidence_.observe(TargetObservation{1,candidates[index].position_m,
                                                                                 time.depth_stamp_s},time.now_s);
                        return observed(regions, index, candidates[index].position_m, evidence, time.depth_stamp_s);
                    }
            TrackerResult out; out.region_count = regions.size(); out.reason = "selection_has_no_valid_region";
            return out;
        }
        if (!selected_) { TrackerResult out; out.region_count = regions.size(); out.reason = "select_target_pixel"; return out; }
        const auto prediction = motion_.predict(time.depth_stamp_s);
        if (!prediction) return missing(time, regions.size(), "prediction_expired_reselect_required");
        const auto match = associate_target(candidates,
            {prediction->position_m,prediction->variance_m2,reference_},config_.association);
        if (match.state != AssociationState::matched)
            return missing(time, regions.size(), match.state == AssociationState::ambiguous ?
                           "ambiguous_candidates" : "no_compatible_candidate");
        const auto index = *match.candidate_index;
        auto updated = motion_;
        if (!updated.update(candidates[index].position_m, time.depth_stamp_s))
            return missing(time, regions.size(), "motion_innovation_rejected");
        const auto evidence = evidence_.observe(TargetObservation{1,candidates[index].position_m,time.depth_stamp_s},time.now_s);
        if (evidence.state == EvidenceState::position_jump || evidence.state == EvidenceState::different_target ||
            evidence.state == EvidenceState::stale || evidence.state == EvidenceState::out_of_order)
            return missing(time, regions.size(), "temporal_evidence_rejected");
        motion_ = updated;
        return observed(regions,index,candidates[index].position_m,evidence,time.depth_stamp_s);
    }
private:
    DepthRegionWorkspace region_workspace_;
    TrackerConfig config_;
    Motion3D motion_;
    TargetEvidence evidence_;
    std::vector<double> reference_;
    bool selected_ = false;
    std::optional<double> last_frame_stamp_, last_now_;
    void check_now(double now_s) {
        if (!std::isfinite(now_s)) throw std::invalid_argument("invalid current time");
        if (last_now_ && now_s < *last_now_) { reset(); throw std::invalid_argument("clock rewind; tracker reset"); }
        last_now_ = now_s;
    }
    TrackerResult rejected(double now_s, const char* reason) {
        evidence_.observe(std::nullopt,now_s);
        TrackerResult out; out.state = TrackState::rejected_frame; out.reason = reason;
        return out;
    }
    TrackerResult missing(FrameTimes time, std::size_t regions, const char* reason) {
        evidence_.observe(std::nullopt,time.now_s);
        TrackerResult out; out.prediction = motion_.predict(time.depth_stamp_s);
        out.state = out.prediction ? TrackState::occluded : TrackState::lost;
        out.region_count = regions; out.reason = reason;
        return out;
    }
    TrackerResult observed(const std::vector<DepthRegion>& regions, std::size_t index, Point3 point,
                           const EvidenceResult& evidence, double stamp) {
        TrackerResult out; out.state = evidence.ready ? TrackState::tracking : TrackState::confirming;
        out.confirmed = evidence.ready; out.fresh_observation = true; out.observed_position_m = point;
        out.bounds = regions[index].bounds; out.representative_pixel = regions[index].representative_pixel;
        out.prediction = motion_.predict(stamp); out.region_count = regions.size();
        out.reason = evidence.ready ? "associated_and_confirmed" : "confirming_association";
        return out;
    }
};
} // namespace operator_lab::rgbd
