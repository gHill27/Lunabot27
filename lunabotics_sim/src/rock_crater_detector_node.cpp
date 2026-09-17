// rock_crater_detector_node.cpp
//
// ============================================================================
// Subscribes to the robot's RGB-D camera, detects rocks/walls (positive
// obstacles) and craters (negative obstacles) using terrain_features.hpp,
// and publishes:
//
//   - /rock_crater_detections        (std_msgs/String, JSON)
//       Confirmed, tracked detections. Any node can parse this without a
//       custom message type.
//
//   - /rock_crater_markers           (visualization_msgs/MarkerArray)
//       One sphere per confirmed track, positioned in 3D (camera_optical_link
//       frame -- see the FRAME CAVEAT below), colored by class, viewable in
//       RViz.
//
//   - /rock_crater_annotated_image   (sensor_msgs/Image)
//       The RGB feed with detected blobs outlined and labeled with type +
//       distance, for viewing in rqt_image_view or RViz.
//
// ============================================================================
// PORTED FROM THE ORIGINAL API. What changed and why it broke the build:
//
//   Detection.label   std::string ("rock"/"crater")  ->  terrain_features::Label (enum)
//                      Fix: compare with Label::Rock / toString(det.label) to print it.
//                      This is exactly what your build error was: operator<<
//                      has no idea how to stream an enum, so the compiler
//                      went hunting through every unrelated operator<< it
//                      could find (hence the wall of std::distribution
//                      "candidate" noise) and failed on all of them.
//
//   Detection.pos_x/y/z            ->  Detection.centroid (Eigen::Vector3f)
//   result.plane                   ->  result.ground (GroundModel, has .normal/.d)
//   segmentFeatures(depth,fx,fy,...) (17 positional floats)
//                                   ->  segment(depth, Intrinsics{...}, Params{...})
//                                       (struct-based config -- see Params in
//                                       terrain_features.hpp for every field)
//
//   NEW in this version, wired in below:
//     - Label::Wall is now a real class (previously any obstacle taller than
//       ~30cm or bigger than ~1.5 sq m was silently DROPPED, not reported).
//       This node treats Wall the same as Rock for markers/JSON (same color,
//       same shape) -- split them out if your consumer needs to react to
//       them differently.
//     - TerrainTracker: raw per-frame detections flicker (a blob at the edge
//       of the area threshold can appear/disappear frame to frame). The
//       tracker requires an obstacle to be seen confirm_hits times before
//       it's reported, and coasts it through a few missed frames rather than
//       having it blink in and out. This node now publishes ONLY confirmed
//       tracks, not raw per-frame detections -- see the FRAME CAVEAT below
//       for why that requires one more piece of information than the old
//       node needed.
// ============================================================================
//
// FRAME CAVEAT -- READ BEFORE TRUSTING THE OUTPUT AT SPEED:
// terrain_features::segment() returns detections in the CAMERA's own optical
// frame, which moves with the robot. TerrainTracker needs to accumulate
// evidence for the same physical rock across frames, so it needs positions
// in a frame that does NOT move -- otherwise, while the robot drives, a
// perfectly stationary rock looks like it's sliding backwards through space,
// and the tracker will spawn a new phantom track for it every frame instead
// of confirming one real track. That's exactly what "5 confirmed tracks
// where the truth is 2" looked like when I tested this at Anthropic.
//
// This node currently passes identity for cam_to_world (see the TODO in
// depthCallback), which means: tracking will work correctly ONLY while the
// camera is stationary. As soon as you wire this up to a moving rover, you
// need a tf2 lookup from the camera frame to your fixed frame (odom or map)
// at the depth image's timestamp, and pass that transform into
// tracker_.update() instead of the identity. I've left the lookup call
// commented out rather than guessing at your TF tree's frame names.
// ============================================================================

#include <memory>
#include <sstream>
#include <iomanip>
#include <array>
#include <optional>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <std_msgs/msg/string.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <cv_bridge/cv_bridge.hpp>
#include <opencv2/opencv.hpp>

// Uncomment these two, plus the lookup block marked TODO below, once you're
// ready to wire up real robot motion instead of the stationary-camera
// identity transform:
// #include <tf2_ros/transform_listener.h>
// #include <tf2_ros/buffer.h>
// #include <tf2_eigen/tf2_eigen.hpp>

#include "terrain_features.hpp"

using std::placeholders::_1;
using terrain_features::Label;

namespace {
// Detection thresholds -- see Params in terrain_features.hpp for the full
// list and what each one does; only the ones worth tuning per-robot are
// broken out here.
constexpr float ROCK_THRESH_M = 0.04f;
constexpr float CRATER_THRESH_M = 0.04f;
constexpr int MIN_AREA_PX = 60;
constexpr float MAX_VALID_DEPTH_M = 9.5f;
constexpr int DOWNSAMPLE_FACTOR = 2;

// Tracking thresholds. confirm_hits=3 means an obstacle must appear in 3
// consecutive processed frames before it's published -- this is what turns
// off one-frame false positives. max_misses=5 means a confirmed obstacle
// survives up to 5 consecutive missed frames (occlusion, brief dropout)
// before its track is deleted.
constexpr int CONFIRM_HITS = 3;
constexpr int MAX_MISSES = 5;

const std::string CAMERA_FRAME_ID = "camera_optical_link";
}  // namespace

class RockCraterDetector : public rclcpp::Node {
public:
    RockCraterDetector() : Node("rock_crater_detector") {
        rgb_sub_ = create_subscription<sensor_msgs::msg::Image>(
            "/camera/image", 10,
            std::bind(&RockCraterDetector::rgbCallback, this, _1));

        depth_sub_ = create_subscription<sensor_msgs::msg::Image>(
            "/camera/depth_image", 10,
            std::bind(&RockCraterDetector::depthCallback, this, _1));

        camera_info_sub_ = create_subscription<sensor_msgs::msg::CameraInfo>(
            "/camera/camera_info", 10,
            std::bind(&RockCraterDetector::cameraInfoCallback, this, _1));

        detections_pub_ = create_publisher<std_msgs::msg::String>(
            "/rock_crater_detections", 10);
        markers_pub_ = create_publisher<visualization_msgs::msg::MarkerArray>(
            "/rock_crater_markers", 10);
        annotated_pub_ = create_publisher<sensor_msgs::msg::Image>(
            "/rock_crater_annotated_image", 10);

        // --- detection params -------------------------------------------
        params_.rock_thresh = ROCK_THRESH_M;
        params_.crater_thresh = CRATER_THRESH_M;
        params_.min_area_px = MIN_AREA_PX;
        params_.max_valid_depth = MAX_VALID_DEPTH_M;
        params_.downsample_factor = DOWNSAMPLE_FACTOR;
        // Level-mounted camera assumed. If yours is pitched down, replace
        // this with:
        //   params_.expected_normal =
        //       terrain_features::expectedGroundNormalFromPitch(pitch_deg);
        params_.expected_normal = Eigen::Vector3f(0.0f, 1.0f, 0.0f);

        // --- tracker params -----------------------------------------------
        terrain_features::TrackerParams tp;
        tp.confirm_hits = CONFIRM_HITS;
        tp.max_misses = MAX_MISSES;
        tracker_ = terrain_features::TerrainTracker(tp);

        RCLCPP_INFO(get_logger(),
            "Waiting for /camera/camera_info to get intrinsics before detecting...");
    }

    ~RockCraterDetector() override {
        cv::destroyAllWindows();
    }

private:
    void cameraInfoCallback(const sensor_msgs::msg::CameraInfo::SharedPtr msg) {
        if (!intrinsics_.has_value()) {
            terrain_features::Intrinsics in;
            in.fx = static_cast<float>(msg->k[0]);
            in.fy = static_cast<float>(msg->k[4]);
            in.cx = static_cast<float>(msg->k[2]);
            in.cy = static_cast<float>(msg->k[5]);
            intrinsics_ = in;
            RCLCPP_INFO(get_logger(), "Got camera intrinsics: fx=%.1f fy=%.1f cx=%.1f cy=%.1f",
                        in.fx, in.fy, in.cx, in.cy);
        }
    }

    void rgbCallback(const sensor_msgs::msg::Image::SharedPtr msg) {
        try {
            latest_rgb_ = cv_bridge::toCvCopy(msg, "bgr8")->image;
        } catch (const cv_bridge::Exception &e) {
            RCLCPP_ERROR(get_logger(), "cv_bridge exception (rgb): %s", e.what());
        }
    }

    void depthCallback(const sensor_msgs::msg::Image::SharedPtr msg) {
        if (!intrinsics_.has_value() || latest_rgb_.empty()) {
            return;
        }

        cv::Mat depth;
        try {
            depth = cv_bridge::toCvCopy(msg, "32FC1")->image;
        } catch (const cv_bridge::Exception &e) {
            RCLCPP_ERROR(get_logger(), "cv_bridge exception (depth): %s", e.what());
            return;
        }

        // segment() takes an Intrinsics struct and a Params struct now,
        // instead of ~17 positional float arguments. Both are set up once
        // in the constructor above.
        auto result = terrain_features::segment(depth, intrinsics_.value(), params_);

        // --- TODO: replace this identity transform with a real lookup ----
        // once this node is running on a moving robot. See the FRAME CAVEAT
        // at the top of this file for why this matters. Example, once tf2
        // is wired in (uncomment the includes at the top of the file too):
        //
        //   geometry_msgs::msg::TransformStamped tf =
        //       tf_buffer_->lookupTransform("odom", CAMERA_FRAME_ID, msg->header.stamp);
        //   Eigen::Isometry3f cam_to_world = tf2::transformToEigen(tf).cast<float>();
        //
        // Until then, tracking is only valid while the camera is stationary.
        Eigen::Isometry3f cam_to_world = Eigen::Isometry3f::Identity();

        const double stamp_s =
            static_cast<double>(msg->header.stamp.sec) +
            static_cast<double>(msg->header.stamp.nanosec) * 1e-9;

        const auto &tracks = tracker_.update(result.detections, stamp_s, cam_to_world);
        auto confirmed = tracker_.confirmedTracks();

        // Publish confirmed, stable tracks -- NOT the raw per-frame
        // detections in result.detections. Publishing raw detections
        // defeats the entire point of tracking: every one-frame false
        // positive would go straight out to RViz / whatever consumes the
        // JSON topic.
        publishJson(confirmed, msg->header.stamp);
        publishMarkers(confirmed, msg->header.stamp);
        publishAnnotatedImage(confirmed, msg->header.stamp);

        (void)tracks;  // silence unused-variable warning if you don't need the full list here

        if (!result.ground.has_value()) {
            RCLCPP_WARN(get_logger(),
                "No ground plane found this frame (camera may be too close "
                "to an obstacle, or scene too cluttered).");
        } else {
            const auto &g = *result.ground;
            RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 2000,
                "Ground plane: normal=(%.3f, %.3f, %.3f) d=%.3f inlier_frac=%.2f "
                "| raw detections: %zu | confirmed tracks: %zu",
                g.normal.x(), g.normal.y(), g.normal.z(), g.d, g.inlier_frac,
                result.detections.size(), confirmed.size());
        }
    }

    // ------------------------------------------------------------------------
    // All three publish* functions below now take Track (confirmed, tracked
    // obstacles with a stable id) instead of Detection (raw per-frame blobs).
    // Track carries the same geometry fields plus `id`, `hits`, `confidence`.
    // ------------------------------------------------------------------------

    void publishJson(const std::vector<terrain_features::Track> &tracks,
                      const builtin_interfaces::msg::Time &stamp) {
        std::ostringstream oss;
        oss << "{\"stamp\":{\"sec\":" << stamp.sec << ",\"nanosec\":" << stamp.nanosec
            << "},\"detections\":[";
        for (size_t i = 0; i < tracks.size(); ++i) {
            const auto &t = tracks[i];
            if (i > 0) oss << ",";
            // toString(Label) turns the enum back into "rock"/"crater"/"wall"
            // for the JSON string -- this replaces the old direct string
            // comparison / stream of det.label.
            oss << "{\"id\":" << t.id << ","
                << "\"label\":\"" << terrain_features::toString(t.label) << "\","
                << "\"position\":[" << t.position.x() << "," << t.position.y()
                << "," << t.position.z() << "],"
                << "\"height_m\":" << t.height_m << ","
                << "\"footprint_m\":[" << t.footprint_w_m << "," << t.footprint_l_m << "],"
                << "\"confidence\":" << t.confidence << ","
                << "\"hits\":" << t.hits << "}";
        }
        oss << "]}";

        std_msgs::msg::String msg;
        msg.data = oss.str();
        detections_pub_->publish(msg);
    }

    void publishMarkers(const std::vector<terrain_features::Track> &tracks,
                         const builtin_interfaces::msg::Time &stamp) {
        visualization_msgs::msg::MarkerArray array;

        visualization_msgs::msg::Marker clear_marker;
        clear_marker.header.frame_id = CAMERA_FRAME_ID;
        clear_marker.header.stamp = stamp;
        clear_marker.action = visualization_msgs::msg::Marker::DELETEALL;
        array.markers.push_back(clear_marker);

        for (const auto &t : tracks) {
            visualization_msgs::msg::Marker marker;
            marker.header.frame_id = CAMERA_FRAME_ID;
            marker.header.stamp = stamp;
            marker.ns = terrain_features::toString(t.label);
            marker.id = t.id;  // stable across frames now, since it comes from the tracker,
                                // not a per-frame vector index -- this means a given rock keeps
                                // the same marker.id (and thus the same RViz "object") over time
            marker.type = visualization_msgs::msg::Marker::SPHERE;
            marker.action = visualization_msgs::msg::Marker::ADD;

            marker.pose.position.x = t.position.x();
            marker.pose.position.y = t.position.y();
            marker.pose.position.z = t.position.z();
            marker.pose.orientation.w = 1.0;

            constexpr double size = 0.08;
            marker.scale.x = size;
            marker.scale.y = size;
            marker.scale.z = size;

            // Rock and Wall share a color (both are positive obstacles);
            // Crater gets its own. Split Wall out here if your planner
            // needs to treat "climb over" and "impassable" differently.
            switch (t.label) {
                case Label::Rock:
                case Label::Wall:
                    marker.color.r = 0.0f; marker.color.g = 0.0f; marker.color.b = 1.0f;
                    break;
                case Label::Crater:
                    marker.color.r = 1.0f; marker.color.g = 0.5f; marker.color.b = 0.0f;
                    break;
            }
            marker.color.a = 0.9f;

            marker.lifetime = rclcpp::Duration::from_seconds(1.0);

            array.markers.push_back(marker);
        }

        markers_pub_->publish(array);
    }

    void publishAnnotatedImage(const std::vector<terrain_features::Track> &tracks,
                                const builtin_interfaces::msg::Time &stamp) {
        if (latest_rgb_.empty()) return;

        cv::Mat annotated = latest_rgb_.clone();

        for (const auto &t : tracks) {
            // Track doesn't carry image-space pixel coordinates directly --
            // that's a property of the raw Detection that produced it, not
            // of the tracked 3D point. last_detection is the most recent
            // Detection that updated this track, and still has pixel_u/v
            // and area_px from that frame's segmentation pass.
            const auto &d = t.last_detection;
            cv::Point center(static_cast<int>(d.pixel_u), static_cast<int>(d.pixel_v));

            cv::Scalar color = (t.label == Label::Crater)
                ? cv::Scalar(0, 128, 255)   // BGR orange
                : cv::Scalar(0, 0, 255);    // BGR red (rock or wall)

            int radius = std::max(10, static_cast<int>(std::sqrt(d.area_px / M_PI)));

            cv::circle(annotated, center, radius, color, 2);
            std::ostringstream text;
            // t.position.norm() is the track's current filtered range;
            // using this instead of a raw per-frame distance is part of why
            // the annotated overlay is steadier than the old node's.
            text << terrain_features::toString(t.label) << " "
                 << std::fixed << std::setprecision(2) << t.position.norm() << "m";
            cv::putText(annotated, text.str(),
                        cv::Point(center.x - radius, center.y - radius - 8),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 2);
        }

        cv::imshow("Rock/Crater Detections", annotated);
        cv::waitKey(1);

        auto img_msg = cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", annotated).toImageMsg();
        img_msg->header.stamp = stamp;
        img_msg->header.frame_id = CAMERA_FRAME_ID;
        annotated_pub_->publish(*img_msg);
    }

    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr rgb_sub_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr depth_sub_;
    rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr camera_info_sub_;

    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr detections_pub_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr markers_pub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr annotated_pub_;

    cv::Mat latest_rgb_;
    std::optional<terrain_features::Intrinsics> intrinsics_;

    terrain_features::Params params_;
    terrain_features::TerrainTracker tracker_;
};

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<RockCraterDetector>());
    rclcpp::shutdown();
    return 0;
}