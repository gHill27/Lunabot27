// test_terrain_features.cpp
//
// Synthetic validation tests, mirroring test_terrain_features.py, to
// confirm the C++ port of the algorithm behaves identically to the
// validated Python version.

#include <iostream>
#include <cmath>
#include "terrain_features.hpp"

using namespace terrain_features;

const int W = 640, H = 480;
const float FOV_H = 1.211f;
const float FX = W / (2.0f * std::tan(FOV_H / 2.0f));
const float FY = FX;
const float CX = W / 2.0f, CY = H / 2.0f;
const float CAMERA_HEIGHT = 0.22f;

// Build the synthetic flat-ground depth image (no rock/crater yet).
cv::Mat makeGroundDepth() {
    cv::Mat depth(H, W, CV_32F);
    for (int v = 0; v < H; ++v) {
        for (int u = 0; u < W; ++u) {
            float ray_y = (v - CY) / FY;
            float d;
            if (ray_y <= 0) {
                d = 10.0f;  // above horizon -> far clip placeholder
            } else {
                d = CAMERA_HEIGHT / ray_y;
                if (d > 10.0f) d = 10.0f;
            }
            depth.at<float>(v, u) = d;
        }
    }
    return depth;
}

bool testRockAndCrater() {
    cv::Mat depth = makeGroundDepth();

    // Rock: physically ROCK_HEIGHT meters above ground.
    const int rock_u = 340, rock_v = 300, rock_radius = 25;
    const float ROCK_HEIGHT = 0.15f;
    for (int v = 0; v < H; ++v) {
        for (int u = 0; u < W; ++u) {
            int du = u - rock_u, dv = v - rock_v;
            if (du * du + dv * dv < rock_radius * rock_radius) {
                float ray_y = (v - CY) / FY;
                float target_y = CAMERA_HEIGHT - ROCK_HEIGHT;
                depth.at<float>(v, u) = target_y / ray_y;
            }
        }
    }

    // Crater: physically CRATER_DEPTH meters below ground.
    const int crater_u = 220, crater_v = 350, crater_radius = 30;
    const float CRATER_DEPTH = 0.20f;
    for (int v = 0; v < H; ++v) {
        for (int u = 0; u < W; ++u) {
            int du = u - crater_u, dv = v - crater_v;
            if (du * du + dv * dv < crater_radius * crater_radius) {
                float ray_y = (v - CY) / FY;
                float target_y = CAMERA_HEIGHT + CRATER_DEPTH;
                depth.at<float>(v, u) = target_y / ray_y;
            }
        }
    }

    // Sensor noise.
    cv::Mat noise(H, W, CV_32F);
    cv::randn(noise, 0.0, 0.005);
    depth += noise;
    cv::max(depth, 0.05f, depth);
    cv::min(depth, 10.0f, depth);

    SegmentResult result = segmentFeatures(depth, FX, FY, CX, CY, 0.05f, 0.05f, 80);

    std::cout << "Plane found: " << (result.plane.has_value() ? "yes" : "no") << "\n";
    std::cout << "Number of detections: " << result.detections.size() << "\n";

    int rock_count = 0, crater_count = 0;
    float rock_px_err = -1, crater_px_err = -1;

    for (const auto &det : result.detections) {
        std::cout << "  " << det.label << " pixel=(" << det.pixel_u << "," << det.pixel_v
                  << ") pos=(" << det.pos_x << "," << det.pos_y << "," << det.pos_z
                  << ") distance=" << det.distance << "m area=" << det.area_px << "px\n";

        if (det.label == "rock") {
            rock_count++;
            rock_px_err = std::hypot(det.pixel_u - rock_u, det.pixel_v - rock_v);
        } else if (det.label == "crater") {
            crater_count++;
            crater_px_err = std::hypot(det.pixel_u - crater_u, det.pixel_v - crater_v);
        }
    }

    bool ok = true;
    if (!result.plane.has_value()) { std::cout << "FAIL: no plane found\n"; ok = false; }
    if (rock_count != 1) { std::cout << "FAIL: expected 1 rock, got " << rock_count << "\n"; ok = false; }
    if (crater_count != 1) { std::cout << "FAIL: expected 1 crater, got " << crater_count << "\n"; ok = false; }
    if (rock_px_err >= 0 && rock_px_err > 5) { std::cout << "FAIL: rock centroid off by " << rock_px_err << "px\n"; ok = false; }
    if (crater_px_err >= 0 && crater_px_err > 5) { std::cout << "FAIL: crater centroid off by " << crater_px_err << "px\n"; ok = false; }

    if (ok) std::cout << "PASSED: rock+crater test\n";
    return ok;
}

bool testWallRobustness() {
    cv::Mat depth = makeGroundDepth();

    // Replace the "above horizon" placeholder with a large flat wall at
    // 2m, filling most of the upper half -- this should NOT get picked as
    // the ground plane.
    const float wall_z = 2.0f;
    for (int v = 0; v < H; ++v) {
        for (int u = 0; u < W; ++u) {
            float ray_y = (v - CY) / FY;
            if (ray_y <= 0) {
                depth.at<float>(v, u) = wall_z;
            }
        }
    }

    cv::Mat noise(H, W, CV_32F);
    cv::randn(noise, 0.0, 0.005);
    depth += noise;
    cv::max(depth, 0.05f, depth);
    cv::min(depth, 10.0f, depth);

    SegmentResult result = segmentFeatures(depth, FX, FY, CX, CY, 0.05f, 0.05f, 80);

    if (!result.plane.has_value()) {
        std::cout << "FAIL: no plane found in wall-robustness test\n";
        return false;
    }

    Eigen::Vector3f expected(0.0f, 1.0f, 0.0f);
    float alignment = std::abs(result.plane->normal.dot(expected));
    float angle_deg = std::acos(std::min(1.0f, alignment)) * 180.0f / static_cast<float>(M_PI);

    std::cout << "Fitted normal: (" << result.plane->normal.x() << ", "
              << result.plane->normal.y() << ", " << result.plane->normal.z() << ")\n";
    std::cout << "Angle from true ground normal: " << angle_deg << " deg\n";

    if (angle_deg > 10.0f) {
        std::cout << "FAIL: RANSAC picked the wall instead of the ground!\n";
        return false;
    }
    std::cout << "PASSED: wall-robustness test\n";
    return true;
}

int main() {
    bool ok1 = testRockAndCrater();
    std::cout << "\n";
    bool ok2 = testWallRobustness();

    if (ok1 && ok2) {
        std::cout << "\nAll C++ tests passed.\n";
        return 0;
    }
    std::cout << "\nSOME TESTS FAILED.\n";
    return 1;
}
