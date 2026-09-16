// rock_crater_detector_node.cpp
//
// Subscribes to the robot's RGB-D camera, detects rocks (positive
// obstacles) and craters (negative obstacles) using ground-plane-relative
// depth segmentation (see terrain_features.hpp), and publishes:
//
//   - /rock_crater_detections        (std_msgs/String, JSON)
//       Simple structured output any other node can parse without needing
//       a custom message type.
//
//   - /rock_crater_markers           (visualization_msgs/MarkerArray)
//       One sphere per detection, positioned in 3D (camera_optical_link
//       frame), colored red for rocks / orange for craters, viewable in
//       RViz.
//
//   - /rock_crater_annotated_image   (sensor_msgs/Image)
//       The RGB feed with detected blobs outlined and labeled with type +
//       distance, for viewing in rqt_image_view or RViz.

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

#include "terrain_features.hpp"

using std::placeholders::_1;

namespace {
constexpr float ROCK_THRESH_M = 0.04f;
constexpr float CRATER_THRESH_M = 0.04f;
constexpr int MIN_AREA_PX = 60;
constexpr int MAX_AREA_PX = 6000;   // reject anything bigger (walls, columns) as structural, not a rock/crater
constexpr float MAX_VALID_DEPTH_M = 9.5f;
constexpr int RANSAC_ITERATIONS = 150;
constexpr float RANSAC_DIST_THRESH_M = 0.03f;
constexpr int MAX_RANSAC_POINTS = 4000;  // subsample for the RANSAC fit itself; huge speedup, no accuracy loss
constexpr int DEPTH_BLUR_KSIZE = 5;      // median blur kernel on depth before processing (0/1 disables)
constexpr int DEPTH_BLUR_PASSES = 2;     // repeat the blur for a stronger effective smooth (CV_32F
                                          // only supports ksize 3/5 per pass, so stack passes instead
                                          // of raising ksize further)
constexpr int MORPH_KERNEL_SIZE = 5;     // cleanup kernel on the binary masks; bump up for noisier scenes
constexpr int DOWNSAMPLE_FACTOR = 2;     // process at 1/factor resolution; lower to 1 if small/distant
                                          // rocks are being missed, at the cost of higher CPU use
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

        RCLCPP_INFO(get_logger(),
            "Waiting for /camera/camera_info to get intrinsics before detecting...");
    }

    ~RockCraterDetector() override {
        cv::destroyAllWindows();
    }

private:
    void cameraInfoCallback(const sensor_msgs::msg::CameraInfo::SharedPtr msg) {
        if (!intrinsics_.has_value()) {
            float fx = static_cast<float>(msg->k[0]);
            float fy = static_cast<float>(msg->k[4]);
            float cx = static_cast<float>(msg->k[2]);
            float cy = static_cast<float>(msg->k[5]);
            intrinsics_ = std::array<float, 4>{fx, fy, cx, cy};
            RCLCPP_INFO(get_logger(), "Got camera intrinsics: fx=%.1f fy=%.1f cx=%.1f cy=%.1f",
                        fx, fy, cx, cy);
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

        auto [fx, fy, cx, cy] = intrinsics_.value();

        auto result = terrain_features::segmentFeatures(
            depth, fx, fy, cx, cy,
            ROCK_THRESH_M, CRATER_THRESH_M, MIN_AREA_PX,
            RANSAC_ITERATIONS, RANSAC_DIST_THRESH_M, MAX_VALID_DEPTH_M,
            Eigen::Vector3f(0.0f, 1.0f, 0.0f), 30.0f,
            MAX_RANSAC_POINTS, MAX_AREA_PX,
            DEPTH_BLUR_KSIZE, DEPTH_BLUR_PASSES, DOWNSAMPLE_FACTOR,
            MORPH_KERNEL_SIZE);

        publishJson(result.detections, msg->header.stamp);
        publishMarkers(result.detections, msg->header.stamp);
        publishAnnotatedImage(result.detections, msg->header.stamp);

        if (!result.plane.has_value()) {
            RCLCPP_WARN(get_logger(),
                "No ground plane found this frame (camera may be too close "
                "to an obstacle, or scene too cluttered).");
        } else {
            // Throttled diagnostic: confirm the fitted plane is actually
            // close to the expected "camera looking roughly level at
            // flat ground" orientation. If normal.y is not close to 1.0,
            // or d looks wildly off from the camera's real mount height,
            // that's a sign the plane fit locked onto the wrong surface
            // (e.g. a nearby wall) even after the angle constraint.
            RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 2000,
                "Ground plane: normal=(%.3f, %.3f, %.3f) d=%.3f | detections: %zu",
                result.plane->normal.x(), result.plane->normal.y(),
                result.plane->normal.z(), result.plane->d,
                result.detections.size());
        }
    }

    void publishJson(const std::vector<terrain_features::Detection> &detections,
                      const builtin_interfaces::msg::Time &stamp) {
        std::ostringstream oss;
        oss << "{\"stamp\":{\"sec\":" << stamp.sec << ",\"nanosec\":" << stamp.nanosec
            << "},\"detections\":[";
        for (size_t i = 0; i < detections.size(); ++i) {
            const auto &det = detections[i];
            if (i > 0) oss << ",";
            oss << "{\"label\":\"" << det.label << "\","
                << "\"pixel\":[" << det.pixel_u << "," << det.pixel_v << "],"
                << "\"position\":[" << det.pos_x << "," << det.pos_y << "," << det.pos_z << "],"
                << "\"distance\":" << det.distance << ","
                << "\"area_px\":" << det.area_px << "}";
        }
        oss << "]}";

        std_msgs::msg::String msg;
        msg.data = oss.str();
        detections_pub_->publish(msg);
    }

    void publishMarkers(const std::vector<terrain_features::Detection> &detections,
                         const builtin_interfaces::msg::Time &stamp) {
        visualization_msgs::msg::MarkerArray array;

        // Clear previous markers first so stale detections don't linger.
        visualization_msgs::msg::Marker clear_marker;
        clear_marker.header.frame_id = CAMERA_FRAME_ID;
        clear_marker.header.stamp = stamp;
        clear_marker.action = visualization_msgs::msg::Marker::DELETEALL;
        array.markers.push_back(clear_marker);

        for (size_t i = 0; i < detections.size(); ++i) {
            const auto &det = detections[i];
            visualization_msgs::msg::Marker marker;
            marker.header.frame_id = CAMERA_FRAME_ID;
            marker.header.stamp = stamp;
            marker.ns = det.label;
            marker.id = static_cast<int>(i);
            marker.type = visualization_msgs::msg::Marker::SPHERE;
            marker.action = visualization_msgs::msg::Marker::ADD;

            marker.pose.position.x = det.pos_x;
            marker.pose.position.y = det.pos_y;
            marker.pose.position.z = det.pos_z;
            marker.pose.orientation.w = 1.0;

            constexpr double size = 0.08;
            marker.scale.x = size;
            marker.scale.y = size;
            marker.scale.z = size;

            if (det.label == "rock") {
                marker.color.r = 0.0f; marker.color.g = 0.0f; marker.color.b = 1.0f;
            } else {
                marker.color.r = 1.0f; marker.color.g = 0.5f; marker.color.b = 0.0f;
            }
            marker.color.a = 0.9f;

            marker.lifetime = rclcpp::Duration::from_seconds(1.0);

            array.markers.push_back(marker);
        }

        markers_pub_->publish(array);
    }

    void publishAnnotatedImage(const std::vector<terrain_features::Detection> &detections,
                                const builtin_interfaces::msg::Time &stamp) {
        if (latest_rgb_.empty()) return;

        cv::Mat annotated = latest_rgb_.clone();

        for (const auto &det : detections) {
            cv::Point center(static_cast<int>(det.pixel_u), static_cast<int>(det.pixel_v));
            cv::Scalar color = (det.label == "rock")
                ? cv::Scalar(0, 0, 255)      // BGR red
                : cv::Scalar(0, 128, 255);   // BGR orange
            int radius = std::max(10, static_cast<int>(std::sqrt(det.area_px / M_PI)));

            cv::circle(annotated, center, radius, color, 2);
            std::ostringstream text;
            text << det.label << " " << std::fixed << std::setprecision(2)
                 << det.distance << "m";
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
    std::optional<std::array<float, 4>> intrinsics_;  // fx, fy, cx, cy
};

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<RockCraterDetector>());
    rclcpp::shutdown();
    return 0;
}
