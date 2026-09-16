// terrain_features.hpp
//
// Pure geometry/CV logic for detecting rocks (positive obstacles, sticking
// up above the ground) and craters (negative obstacles, dipping below the
// ground) from a depth image, using RANSAC ground-plane fitting + residual
// thresholding. No ROS dependency here on purpose, so this can be unit
// tested in isolation and reused outside a ROS node if needed.
//
// Camera convention assumed: standard optical frame (X right, Y down, Z
// forward), matching sensor_msgs/Image + CameraInfo conventions.

#pragma once

#include <vector>
#include <optional>
#include <random>
#include <string>
#include <cmath>

#include <opencv2/opencv.hpp>
#include <Eigen/Dense>

namespace terrain_features {

struct Detection {
    std::string label;       // "rock" or "crater"
    float pixel_u, pixel_v;
    float pos_x, pos_y, pos_z;  // 3D position, camera optical frame
    float distance;
    int area_px;
};

struct Plane {
    Eigen::Vector3f normal;
    float d;
};

struct SegmentResult {
    std::optional<Plane> plane;
    cv::Mat residuals;    // CV_32F, NaN where invalid
    cv::Mat rock_mask;    // CV_8U, 0/255
    cv::Mat crater_mask;  // CV_8U, 0/255
    std::vector<Detection> detections;
};

// Back-project a depth image (CV_32F, meters) into a 3D point cloud.
// Returns a CV_32FC3 Mat of the same size; invalid pixels (depth <= 0,
// non-finite) are set to NaN in all three channels.
inline cv::Mat backprojectDepth(const cv::Mat &depth, float fx, float fy,
                                 float cx, float cy) {
    CV_Assert(depth.type() == CV_32F);
    cv::Mat points(depth.size(), CV_32FC3);

    for (int v = 0; v < depth.rows; ++v) {
        const float *depth_row = depth.ptr<float>(v);
        cv::Vec3f *pts_row = points.ptr<cv::Vec3f>(v);
        for (int u = 0; u < depth.cols; ++u) {
            float z = depth_row[u];
            if (!std::isfinite(z) || z <= 0.0f) {
                pts_row[u] = cv::Vec3f(NAN, NAN, NAN);
                continue;
            }
            float x = (static_cast<float>(u) - cx) / fx * z;
            float y = (static_cast<float>(v) - cy) / fy * z;
            pts_row[u] = cv::Vec3f(x, y, z);
        }
    }
    return points;
}

// RANSAC ground-plane fit, constrained to stay near expected_normal.
//
// valid_points: Nx3, no NaNs.
// expected_normal / max_normal_angle_deg: candidate planes whose normal
// deviates from expected_normal by more than this angle are rejected
// outright, even if they'd otherwise get more inliers. This is what keeps
// RANSAC from locking onto a nearby wall or a large flat far-clip
// "no-return" region instead of the actual ground -- we know the camera is
// mounted roughly level, so the true ground plane's normal should stay
// close to straight down in the camera optical frame (0,1,0) regardless of
// what else is in the scene.
//
// The returned normal/d are oriented so the camera origin (0,0,0) always
// has a NEGATIVE signed residual (origin . normal - d < 0):
//     residual = point . normal - d
//     residual << 0  -> point is on the camera's side of the ground
//                        (sticking up towards the camera) -> ROCK
//     residual >> 0  -> point is further from the camera than the ground
//                        would be at that ray -> CRATER
//     residual ~= 0  -> point lies on the fitted ground plane
inline std::optional<Plane> fitGroundPlaneRansac(
    const std::vector<Eigen::Vector3f> &points,
    int iterations = 150,
    float dist_thresh = 0.03f,
    float min_inlier_frac = 0.15f,
    Eigen::Vector3f expected_normal = Eigen::Vector3f(0.0f, 1.0f, 0.0f),
    float max_normal_angle_deg = 30.0f,
    unsigned int seed = 42) {

    const size_t n_points = points.size();
    if (n_points < 50) {
        return std::nullopt;
    }

    expected_normal.normalize();
    const float cos_thresh = std::cos(max_normal_angle_deg * static_cast<float>(M_PI) / 180.0f);

    std::mt19937 rng(seed);
    std::uniform_int_distribution<size_t> dist(0, n_points - 1);

    std::vector<char> best_inliers;
    size_t best_count = 0;

    for (int iter = 0; iter < iterations; ++iter) {
        size_t i1 = dist(rng), i2 = dist(rng), i3 = dist(rng);
        if (i1 == i2 || i2 == i3 || i1 == i3) continue;

        const Eigen::Vector3f &p1 = points[i1];
        const Eigen::Vector3f &p2 = points[i2];
        const Eigen::Vector3f &p3 = points[i3];

        Eigen::Vector3f v1 = p2 - p1;
        Eigen::Vector3f v2 = p3 - p1;
        Eigen::Vector3f normal = v1.cross(v2);
        float norm_len = normal.norm();
        if (norm_len < 1e-8f) continue;
        normal /= norm_len;

        float alignment = std::abs(normal.dot(expected_normal));
        if (alignment < cos_thresh) continue;  // reject: not roughly ground-like

        float d = normal.dot(p1);

        size_t count = 0;
        std::vector<char> inliers(n_points, 0);
        for (size_t i = 0; i < n_points; ++i) {
            float residual = points[i].dot(normal) - d;
            if (std::abs(residual) < dist_thresh) {
                inliers[i] = 1;
                ++count;
            }
        }

        if (count > best_count) {
            best_count = count;
            best_inliers = std::move(inliers);
        }
    }

    if (best_inliers.empty() ||
        static_cast<float>(best_count) < min_inlier_frac * static_cast<float>(n_points)) {
        return std::nullopt;
    }

    // Refine using PCA (eigenvector of smallest eigenvalue of the
    // covariance matrix) over all inliers -- equivalent to least-squares
    // plane fit, cheaper than full SVD.
    Eigen::Vector3f centroid = Eigen::Vector3f::Zero();
    size_t inlier_count = 0;
    for (size_t i = 0; i < n_points; ++i) {
        if (best_inliers[i]) {
            centroid += points[i];
            ++inlier_count;
        }
    }
    centroid /= static_cast<float>(inlier_count);

    Eigen::Matrix3f cov = Eigen::Matrix3f::Zero();
    for (size_t i = 0; i < n_points; ++i) {
        if (best_inliers[i]) {
            Eigen::Vector3f centered = points[i] - centroid;
            cov += centered * centered.transpose();
        }
    }

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> solver(cov);
    // Eigenvalues are sorted ascending; the smallest corresponds to the
    // plane normal (least variance direction).
    Eigen::Vector3f normal = solver.eigenvectors().col(0);
    normal.normalize();
    float d = normal.dot(centroid);

    // Orient so the camera origin has a negative residual.
    float origin_residual = -d;  // (0,0,0) . normal - d
    if (origin_residual > 0) {
        normal = -normal;
        d = -d;
    }

    return Plane{normal, d};
}

// Full pipeline: backproject depth -> fit ground plane -> threshold
// residuals -> connected components -> per-blob 3D position + distance.
//
// max_valid_depth: pixels at or beyond this range are treated as no-return
// / open sky rather than real terrain, and excluded from both plane
// fitting and rock/crater classification. Should be set a bit below the
// sensor's actual far clip (e.g. 9.5 for a 10.0 far clip) so a large flat
// region of identical far-clip depth values doesn't get mistaken for a
// real flat surface.
inline SegmentResult segmentFeatures(
    const cv::Mat &depth_in, float fx, float fy, float cx, float cy,
    float rock_thresh = 0.05f, float crater_thresh = 0.05f,
    int min_area_px = 80, int ransac_iterations = 150,
    float ransac_dist_thresh = 0.03f, float max_valid_depth = 9.5f,
    Eigen::Vector3f expected_normal = Eigen::Vector3f(0.0f, 1.0f, 0.0f),
    float max_normal_angle_deg = 30.0f,
    int max_ransac_points = 4000,
    int max_area_px = 6000,
    int depth_blur_ksize = 5,
    int depth_blur_passes = 1,
    int downsample_factor = 2,
    int morph_kernel_size = 3,
    float max_rock_height_m = 0.3f,
    float max_crater_depth_m = 0.3f) {

    // --- Blur: cheap, removes single-pixel depth noise/speckle before it
    // can pollute plane fitting or create spurious small blobs. ksize must
    // be odd; 0 or 1 disables.
    //
    // NOTE: cv::medianBlur only supports ksize 3 or 5 for CV_32F inputs
    // (larger kernels are CV_8U-only in OpenCV). To get a STRONGER blur
    // than a single 5x5 pass without violating that, apply it multiple
    // times via depth_blur_passes instead of raising ksize further. ---
    cv::Mat depth = depth_in;
    if (depth_blur_ksize >= 3) {
        for (int pass = 0; pass < std::max(1, depth_blur_passes); ++pass) {
            cv::Mat blurred;
            cv::medianBlur(depth, blurred, depth_blur_ksize);
            depth = blurred;
        }
    }

    // --- Downsample: the single biggest lever for cost, since
    // backprojection + residual computation + connected components are
    // all O(H*W). Halving each dimension cuts that work 4x. Detections'
    // pixel coordinates and area are rescaled back to the ORIGINAL
    // resolution before returning, so callers can overlay them directly
    // on a full-res RGB image without tracking the factor themselves.
    // NOTE: aggressive downsampling can cause small/distant rocks to drop
    // below min_area_px entirely -- if you need to detect small rocks at
    // long range, lower downsample_factor (or set it to 1 to disable).
    float scale = 1.0f;
    float eff_fx = fx, eff_fy = fy, eff_cx = cx, eff_cy = cy;
    if (downsample_factor > 1) {
        scale = 1.0f / static_cast<float>(downsample_factor);
        cv::Mat resized;
        cv::resize(depth, resized, cv::Size(), scale, scale, cv::INTER_NEAREST);
        depth = resized;
        eff_fx *= scale; eff_fy *= scale; eff_cx *= scale; eff_cy *= scale;
    }

    SegmentResult result;
    const int h = depth.rows, w = depth.cols;

    cv::Mat points = backprojectDepth(depth, eff_fx, eff_fy, eff_cx, eff_cy);

    // Treat far-clip / no-return pixels as invalid, same as NaN/inf.
    for (int v = 0; v < h; ++v) {
        const float *depth_row = depth.ptr<float>(v);
        cv::Vec3f *pts_row = points.ptr<cv::Vec3f>(v);
        for (int u = 0; u < w; ++u) {
            if (depth_row[u] >= max_valid_depth) {
                pts_row[u] = cv::Vec3f(NAN, NAN, NAN);
            }
        }
    }

    // Gather valid points for plane fitting. For RANSAC we only need a
    // representative sample, not every pixel -- evaluating residuals
    // against the full cloud on every one of ransac_iterations is by far
    // the most expensive part of this pipeline. A uniform stride sample
    // down to ~max_ransac_points keeps the fit quality essentially
    // identical while cutting RANSAC cost by orders of magnitude on a
    // typical 640x480 frame (~300k valid points -> a few thousand).
    std::vector<Eigen::Vector3f> valid_points;
    valid_points.reserve(static_cast<size_t>(h) * w);
    for (int v = 0; v < h; ++v) {
        const cv::Vec3f *pts_row = points.ptr<cv::Vec3f>(v);
        for (int u = 0; u < w; ++u) {
            const cv::Vec3f &p = pts_row[u];
            if (std::isfinite(p[0]) && std::isfinite(p[1]) && std::isfinite(p[2])) {
                valid_points.emplace_back(p[0], p[1], p[2]);
            }
        }
    }

    std::vector<Eigen::Vector3f> ransac_points;
    if (static_cast<int>(valid_points.size()) > max_ransac_points && max_ransac_points > 0) {
        int stride = static_cast<int>(valid_points.size()) / max_ransac_points;
        ransac_points.reserve(max_ransac_points + 1);
        for (size_t i = 0; i < valid_points.size(); i += stride) {
            ransac_points.push_back(valid_points[i]);
        }
    } else {
        ransac_points = valid_points;
    }

    result.residuals = cv::Mat(h, w, CV_32F, cv::Scalar(NAN));
    result.rock_mask = cv::Mat::zeros(h, w, CV_8U);
    result.crater_mask = cv::Mat::zeros(h, w, CV_8U);

    auto plane_opt = fitGroundPlaneRansac(
        ransac_points, ransac_iterations, ransac_dist_thresh, 0.15f,
        expected_normal, max_normal_angle_deg);

    if (!plane_opt.has_value()) {
        return result;
    }
    result.plane = plane_opt;
    const Eigen::Vector3f &normal = plane_opt->normal;
    const float d = plane_opt->d;

    for (int v = 0; v < h; ++v) {
        const cv::Vec3f *pts_row = points.ptr<cv::Vec3f>(v);
        float *res_row = result.residuals.ptr<float>(v);
        uchar *rock_row = result.rock_mask.ptr<uchar>(v);
        uchar *crater_row = result.crater_mask.ptr<uchar>(v);
        for (int u = 0; u < w; ++u) {
            const cv::Vec3f &p = pts_row[u];
            if (!std::isfinite(p[0])) continue;
            float residual = p[0] * normal.x() + p[1] * normal.y() + p[2] * normal.z() - d;
            res_row[u] = residual;
            if (residual < -rock_thresh && residual > -max_rock_height_m) rock_row[u] = 255;
            if (residual > crater_thresh && residual < max_crater_depth_m) crater_row[u] = 255;
        }
    }

    // Morphological open to clean up noise. Increase morph_kernel_size for
    // more aggressive cleanup of noisy/fragmented blobs.
    cv::Mat kernel = cv::getStructuringElement(
        cv::MORPH_RECT, cv::Size(morph_kernel_size, morph_kernel_size));
    cv::morphologyEx(result.rock_mask, result.rock_mask, cv::MORPH_OPEN, kernel);
    cv::morphologyEx(result.crater_mask, result.crater_mask, cv::MORPH_OPEN, kernel);

    // min_area_px / max_area_px are specified in ORIGINAL (full-resolution)
    // pixel units for a stable, resolution-independent API; scale them into
    // downsampled-pixel units to match what connectedComponentsWithStats
    // will actually report below.
    const int min_area_scaled = std::max(1, static_cast<int>(min_area_px * scale * scale));
    const int max_area_scaled = (max_area_px > 0)
        ? std::max(1, static_cast<int>(max_area_px * scale * scale))
        : -1;

    for (const auto &item : {std::make_pair(std::string("rock"), &result.rock_mask),
                              std::make_pair(std::string("crater"), &result.crater_mask)}) {
        const std::string &label = item.first;
        cv::Mat *mask = item.second;

        cv::Mat labels, stats, centroids;
        int num_labels = cv::connectedComponentsWithStats(*mask, labels, stats, centroids, 8);

        for (int i = 1; i < num_labels; ++i) {  // skip background label 0
            int area = stats.at<int>(i, cv::CC_STAT_AREA);
            if (area < min_area_scaled) continue;
            if (max_area_scaled > 0 && area > max_area_scaled) continue;  // e.g. a wall, not a rock

            double cx_px = centroids.at<double>(i, 0);
            double cy_px = centroids.at<double>(i, 1);

            // Gather this blob's 3D points to compute a robust (median)
            // position estimate.
            std::vector<float> xs, ys, zs;
            for (int v = 0; v < h; ++v) {
                const int *label_row = labels.ptr<int>(v);
                const cv::Vec3f *pts_row = points.ptr<cv::Vec3f>(v);
                for (int u = 0; u < w; ++u) {
                    if (label_row[u] == i && std::isfinite(pts_row[u][0])) {
                        xs.push_back(pts_row[u][0]);
                        ys.push_back(pts_row[u][1]);
                        zs.push_back(pts_row[u][2]);
                    }
                }
            }
            if (xs.empty()) continue;

            auto median = [](std::vector<float> &v) {
                std::nth_element(v.begin(), v.begin() + v.size() / 2, v.end());
                return v[v.size() / 2];
            };
            float px = median(xs), py = median(ys), pz = median(zs);
            float distance = std::sqrt(px * px + py * py + pz * pz);

            Detection det;
            det.label = label;
            det.pixel_u = static_cast<float>(cx_px) / scale;
            det.pixel_v = static_cast<float>(cy_px) / scale;
            det.pos_x = px;
            det.pos_y = py;
            det.pos_z = pz;
            det.distance = distance;
            det.area_px = static_cast<int>(area / (scale * scale));
            result.detections.push_back(det);
        }
    }

    return result;
}

}  // namespace terrain_features
