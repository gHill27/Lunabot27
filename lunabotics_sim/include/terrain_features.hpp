// terrain_features.hpp
//
// Pure geometry/CV logic for detecting positive obstacles (rocks, walls)
// and negative obstacles (craters, holes, drop-offs) from a depth image.
//
// Pipeline:
//   depth -> validity gating -> valid-aware decimate/denoise
//         -> back-projection to an organised point cloud
//         -> robust ground-surface fit (MSAC + Huber IRLS, optional
//            quadratic correction for sloped/curved terrain)
//         -> range-adaptive residual thresholding
//         -> occlusion-gap ("range shadow") detection for negative obstacles
//         -> morphology (close then open)
//         -> connected components
//         -> metric blob filtering + per-blob geometry
//
// No ROS dependency on purpose, so this can be unit tested in isolation
// and reused outside a ROS node.
//
// Camera convention: standard optical frame (X right, Y down, Z forward),
// matching sensor_msgs/Image + CameraInfo.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <random>
#include <vector>

#include <Eigen/Dense>
#include <opencv2/opencv.hpp>

namespace terrain_features {

// ---------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------

enum class Label : uint8_t {
    Rock,    // positive obstacle, small enough to be a discrete object
    Crater,  // negative obstacle: hole, ditch, drop-off
    Wall     // positive obstacle too tall or too wide to be a rock
             // (berm, boulder field, building, the rover's own hardware).
             // NOTE: this is deliberately *reported*, not discarded. The
             // previous version silently dropped anything taller than
             // max_rock_height_m or larger than max_area_px, which meant
             // the biggest hazards in the scene produced zero detections.
};

inline const char *toString(Label l) {
    switch (l) {
        case Label::Rock:   return "rock";
        case Label::Crater: return "crater";
        case Label::Wall:   return "wall";
    }
    return "unknown";
}

struct Intrinsics {
    float fx = 0.0f, fy = 0.0f, cx = 0.0f, cy = 0.0f;
};

struct Detection {
    Label label = Label::Rock;
    float confidence = 0.0f;   // heuristic in [0,1], see computeConfidence()

    // Image space, rescaled back to ORIGINAL full-resolution pixel units
    // so callers can overlay on a full-res RGB frame directly.
    float pixel_u = 0.0f, pixel_v = 0.0f;
    int bbox_x = 0, bbox_y = 0, bbox_w = 0, bbox_h = 0;
    int area_px = 0;      // full-res equivalent
    int n_points = 0;     // number of valid 3D points backing this blob

    // 3D, camera optical frame, metres.
    Eigen::Vector3f centroid{0, 0, 0};  // per-axis median (robust)
    Eigen::Vector3f nearest{0, 0, 0};   // closest point to the camera
    float distance = 0.0f;              // ||nearest||  <- use this for braking
    float centroid_distance = 0.0f;

    // Terrain-relative geometry.
    float height_m = 0.0f;        // signed extreme residual: <0 up, >0 down
    float footprint_w_m = 0.0f;   // extent along the in-plane ex axis
    float footprint_l_m = 0.0f;   // extent along the in-plane ey axis

    // True when the blob is mostly a range shadow (no returns) and its
    // position was inferred by intersecting the centroid ray with the
    // ground surface rather than measured directly.
    bool from_gap = false;
};

// Ground surface model.
//
// The plane normal/d are oriented so the camera origin (0,0,0) always has
// a NEGATIVE signed residual:
//     residual = p . normal - d  (minus optional polynomial correction)
//     residual << 0 -> p sits between the camera and the ground -> ROCK
//     residual >> 0 -> p sits beyond where the ground should be -> CRATER
//     residual ~= 0 -> p lies on the fitted ground surface
struct GroundModel {
    Eigen::Vector3f normal{0, 1, 0};
    float d = 0.0f;

    // Orthonormal in-plane basis, used for the polynomial correction and
    // for measuring blob footprints in metres.
    Eigen::Vector3f ex{1, 0, 0}, ey{0, 0, 1};

    // Optional quadratic height correction h(s,t) fitted over ground
    // inliers, with s = p.ex and t = p.ey. Lets a gently sloping or
    // crowned trail be modelled without the slope leaking into the
    // residuals as fake rocks/craters. Zero when order < 2.
    Eigen::Matrix<float, 6, 1> poly = Eigen::Matrix<float, 6, 1>::Zero();
    int order = 1;
    float max_correction_m = 1.0f;  // clamp, so the quadratic can't blow up
                                    // when extrapolated past the fit region

    // Fit quality, useful for deciding whether to trust a frame at all.
    float inlier_frac = 0.0f;
    float rms_m = 0.0f;

    inline float residual(const Eigen::Vector3f &p) const {
        const float base = p.dot(normal) - d;
        if (order < 2) return base;
        const float s = p.dot(ex), t = p.dot(ey);
        float c = poly[0] + poly[1] * s + poly[2] * t +
                  poly[3] * s * s + poly[4] * t * t + poly[5] * s * t;
        c = std::max(-max_correction_m, std::min(max_correction_m, c));
        return base - c;
    }
};

struct Params {
    // --- validity gating -------------------------------------------------
    // Pixels outside [min_valid_depth, max_valid_depth] are treated as
    // no-return, exactly like NaN/inf. max_valid_depth should sit a bit
    // below the sensor's far clip (e.g. 9.5 for a 10.0 clip) so a large
    // flat slab of identical far-clip values isn't mistaken for ground.
    float min_valid_depth = 0.2f;
    float max_valid_depth = 9.5f;

    // --- preprocessing ---------------------------------------------------
    // Decimation is the single biggest cost lever: back-projection,
    // residuals and connected components are all O(H*W), so factor 2 cuts
    // the work 4x. It is fused with a block median, which denoises at the
    // same time. Aggressive decimation drops small/distant rocks below
    // min_area_px -- set factor 1 if you need long-range small-rock recall.
    int downsample_factor = 2;
    // Valid-aware median. Unlike cv::medianBlur this never mixes NaN or
    // zero-depth pixels into the median, which previously dragged values
    // near dropout regions toward zero and manufactured phantom near-field
    // surfaces. ksize is unrestricted here (cv::medianBlur only allows
    // 3 or 5 on CV_32F). <3 disables.
    int median_ksize = 5;
    int median_passes = 1;
    // Silhouette / mixed-pixel rejection. See removeFlyingPixels() for the
    // constraint these values have to satisfy. 0 disables.
    float flying_pixel_rel = 0.08f;
    float flying_pixel_abs = 0.08f;

    // --- ground model ----------------------------------------------------
    int ransac_iterations = 200;
    float ransac_dist_thresh = 0.03f;  // also the Huber knee for IRLS
    float min_inlier_frac = 0.15f;
    int max_ransac_points = 3000;
    // Candidate planes whose normal deviates from expected_normal by more
    // than max_normal_angle_deg are rejected outright. This is what keeps
    // MSAC off a nearby wall. Use expectedGroundNormalFromPitch() if the
    // camera is mounted pitched down -- with a 25 deg downward mount and
    // the default (0,1,0) the true ground plane is already 25 deg off and
    // eats most of your 30 deg budget.
    Eigen::Vector3f expected_normal{0.0f, 1.0f, 0.0f};
    float max_normal_angle_deg = 30.0f;
    // Restrict ground sampling to the bottom fraction of image rows. The
    // near ground is almost always in the lower image; 0.6 is a good value
    // when tall structures fill the upper half. 1.0 disables.
    float ground_sample_bottom_frac = 1.0f;
    int irls_iterations = 4;
    int ground_model_order = 1;  // 1 = plane, 2 = + quadratic correction
    unsigned int seed = 42;      // fixed -> deterministic, replayable

    // --- classification --------------------------------------------------
    float rock_thresh = 0.05f;    // floor on the rock threshold, metres
    float crater_thresh = 0.05f;  // floor on the crater threshold, metres
    // Stereo/ToF RANGE noise grows roughly as sigma_z ~= c * z^2. But range
    // noise is not what a residual threshold sees -- what matters is how
    // much of it lands along the ground normal:
    //
    //   residual = p . n - d,  p = z * (dx, dy, 1)
    //   d(residual)/dz = (dx, dy, 1) . n
    //   sigma_residual = |(p/z) . n| * sigma_z
    //
    // For a level camera that sensitivity is h_cam/z, so the height noise
    // is c * h_cam * z -- LINEAR in z, not quadratic. Thresholding on a raw
    // z^2 term (the obvious thing to do, and what I first wrote here)
    // over-inflates the far-field threshold by a factor of z/h_cam and
    // throws away real long-range detections.
    //
    // A consequence worth knowing: at grazing incidence the geometric term
    // is small enough that for a decent sensor this correction rarely
    // overtakes the 5 cm floor, and the base threshold is what is really
    // doing the work. The term earns its keep on downward-tilted mounts,
    // where |(p/z).n| approaches 1 and the full range noise lands in the
    // residual. Characterise depth_noise_coeff by pointing the camera at a
    // flat wall at several ranges and taking the std-dev of the residual.
    float depth_noise_coeff = 0.0015f;
    float noise_sigmas = 2.0f;
    float max_crater_depth_m = 2.0f;  // sanity bound on residuals

    // --- negative obstacles ----------------------------------------------
    // A hole with steep walls, viewed at the grazing angle a forward-facing
    // camera actually has, is very nearly INVISIBLE in the residual field:
    // the interior is shadowed, and the ground reappears past the far rim at
    // exactly the range flat ground would have had. Below-ground residuals
    // alone therefore under-report negative obstacles badly -- you only see
    // a thin sliver of far wall, if anything.
    //
    // The reliable cue is the range shadow itself: a run of no-return pixels
    // bounded BELOW (nearer) by real ground and ABOVE (farther) by a valid
    // return, spanning a meaningful distance across the ground.
    //
    // Caveat worth knowing before you enable this on a real sensor: passive
    // stereo drops out on textureless ground for reasons that have nothing
    // to do with holes. Gap detections are deliberately given lower
    // confidence, and min/max_gap_span_m bound how big a dropout can be
    // before it stops being plausible as a hole. If your ground is smooth
    // asphalt or fine dust, expect to raise min_gap_span_m or turn this off.
    bool detect_gaps = true;
    int min_gap_px = 2;  // cheap pre-filter, DOWNSAMPLED rows
    // Gap extent measured along the ground surface, via ray-plane
    // intersection at the near and far rim. Metric, so it means the same
    // thing at 2 m and at 8 m -- unlike a pixel-count threshold.
    float min_gap_span_m = 0.15f;
    float max_gap_span_m = 3.00f;
    int rock_shadow_dilate_px = 3;

    // --- blob extraction -------------------------------------------------
    // Close BEFORE open. Closing first merges a blob that speckle has
    // fragmented; opening first shatters marginal blobs beyond recovery.
    // 5 rather than 3 because a crater's range shadow breaks up across
    // columns and otherwise reports as two adjacent half-craters.
    int morph_close_px = 5;
    int morph_open_px = 3;
    int min_area_px = 40;    // full-res units; cheap pre-filter only
    int min_points = 12;     // minimum valid 3D points backing a blob
    // The real gate is metric, not pixel area: 200 px is a pebble at 1 m
    // and a boulder at 8 m, so a pixel-area threshold silently changes
    // meaning with range.
    float min_obstacle_size_m = 0.05f;
    float max_obstacle_size_m = 1.50f;  // larger footprint -> Wall
    float wall_height_m = 0.30f;        // taller than this -> Wall
};

struct SegmentResult {
    std::optional<GroundModel> ground;

    // All image outputs are at the DOWNSAMPLED resolution; `scale` and
    // `intrinsics_used` tell you how to map back. Detections are already
    // expressed in full-res pixel units.
    cv::Mat depth_used;    // CV_32F metres, NaN where invalid
    cv::Mat residuals;     // CV_32F, NaN where invalid
    cv::Mat rock_mask;     // CV_8U 0/255
    cv::Mat crater_mask;   // CV_8U 0/255 (includes merged gaps)
    cv::Mat ground_mask;   // CV_8U 0/255
    cv::Mat gap_mask;      // CV_8U 0/255, no-return regions judged to be holes

    std::vector<Detection> detections;  // sorted by `distance`, nearest first

    float scale = 1.0f;  // downsampled / original
    Intrinsics intrinsics_used;
};

// ---------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------

namespace detail {

inline float kNaN() { return std::numeric_limits<float>::quiet_NaN(); }

inline float medianInPlace(std::vector<float> &v) {
    const size_t k = v.size() / 2;
    std::nth_element(v.begin(), v.begin() + k, v.end());
    return v[k];
}

// Order statistic by quantile. Percentiles rather than min/max matter a lot
// here: a single mixed pixel at a depth discontinuity is enough to turn a
// 0.3 m rock into a 1.1 m one if you measure extent with min/max.
inline float percentileInPlace(std::vector<float> &v, float q) {
    if (v.empty()) return 0.0f;
    size_t k = static_cast<size_t>(q * static_cast<float>(v.size() - 1) + 0.5f);
    k = std::min(k, v.size() - 1);
    std::nth_element(v.begin(), v.begin() + k, v.end());
    return v[k];
}

}  // namespace detail

// Build an orthonormal basis spanning the plane with the given normal.
inline void makePlaneBasis(const Eigen::Vector3f &n, Eigen::Vector3f &ex,
                           Eigen::Vector3f &ey) {
    const Eigen::Vector3f a = (std::abs(n.z()) < 0.9f)
                                  ? Eigen::Vector3f(0.0f, 0.0f, 1.0f)
                                  : Eigen::Vector3f(1.0f, 0.0f, 0.0f);
    ex = (a - n * a.dot(n)).normalized();
    ey = n.cross(ex).normalized();
}

// Expected ground normal in the optical frame for a camera pitched down by
// `pitch_down_deg`. At 0 deg this is (0,1,0) (straight down in the image);
// at 90 deg (camera staring at its feet) it becomes (0,0,1).
inline Eigen::Vector3f expectedGroundNormalFromPitch(float pitch_down_deg) {
    const float t = pitch_down_deg * static_cast<float>(M_PI) / 180.0f;
    return Eigen::Vector3f(0.0f, std::cos(t), std::sin(t));
}

// Convert a depth image to CV_32F metres. 16-bit inputs are assumed to be
// integer units of `unit_scale` metres (0.001 = millimetres, the RealSense
// and OpenNI default).
inline cv::Mat toMeters(const cv::Mat &depth_in, float unit_scale = 0.001f) {
    if (depth_in.type() == CV_32F) return depth_in;
    cv::Mat out;
    const bool integral =
        (depth_in.depth() == CV_16U || depth_in.depth() == CV_16S ||
         depth_in.depth() == CV_32S);
    depth_in.convertTo(out, CV_32F, integral ? unit_scale : 1.0);
    return out;
}

// Decimate by an integer factor while taking the median of the VALID
// pixels in each block. Out-of-range and non-finite inputs become NaN.
// factor <= 1 just applies the validity gate.
inline cv::Mat decimateValidMedian(const cv::Mat &depth, int factor, float zmin,
                                   float zmax) {
    CV_Assert(depth.type() == CV_32F);
    auto valid = [&](float z) {
        return std::isfinite(z) && z >= zmin && z <= zmax;
    };

    if (factor <= 1) {
        cv::Mat out(depth.size(), CV_32F);
        for (int v = 0; v < depth.rows; ++v) {
            const float *src = depth.ptr<float>(v);
            float *dst = out.ptr<float>(v);
            for (int u = 0; u < depth.cols; ++u)
                dst[u] = valid(src[u]) ? src[u] : detail::kNaN();
        }
        return out;
    }

    // Trailing rows/cols that don't fill a whole block are dropped.
    const int oh = depth.rows / factor, ow = depth.cols / factor;
    cv::Mat out(std::max(oh, 1), std::max(ow, 1), CV_32F,
                cv::Scalar(detail::kNaN()));
    if (oh < 1 || ow < 1) return out;

    std::vector<float> buf;
    buf.reserve(static_cast<size_t>(factor) * factor);
    for (int v = 0; v < oh; ++v) {
        float *dst = out.ptr<float>(v);
        for (int u = 0; u < ow; ++u) {
            buf.clear();
            for (int dv = 0; dv < factor; ++dv) {
                const float *src = depth.ptr<float>(v * factor + dv);
                for (int du = 0; du < factor; ++du) {
                    const float z = src[u * factor + du];
                    if (valid(z)) buf.push_back(z);
                }
            }
            if (!buf.empty()) dst[u] = detail::medianInPlace(buf);
        }
    }
    return out;
}

// Median filter that ignores NaN neighbours entirely and blanks any pixel
// with fewer than `min_valid` finite neighbours (kills isolated speckle).
//
// It deliberately never fills an invalid pixel. An inpainting filter would
// quietly eat the range shadows that negative-obstacle detection depends
// on: a 5-row hole shrinks to 1 row after one pass and the crater vanishes.
// Smooth what was measured; never invent a return.
inline cv::Mat medianFilterValid(const cv::Mat &src, int ksize, int min_valid) {
    CV_Assert(src.type() == CV_32F);
    if (ksize < 3) return src.clone();
    const int r = ksize / 2;
    cv::Mat dst(src.size(), CV_32F, cv::Scalar(detail::kNaN()));

    std::vector<float> buf;
    buf.reserve(static_cast<size_t>(ksize) * ksize);
    for (int v = 0; v < src.rows; ++v) {
        float *drow = dst.ptr<float>(v);
        const float *srow0 = src.ptr<float>(v);
        for (int u = 0; u < src.cols; ++u) {
            if (!std::isfinite(srow0[u])) continue;  // never inpaint
            buf.clear();
            for (int dv = -r; dv <= r; ++dv) {
                const int vv = v + dv;
                if (vv < 0 || vv >= src.rows) continue;
                const float *srow = src.ptr<float>(vv);
                for (int du = -r; du <= r; ++du) {
                    const int uu = u + du;
                    if (uu < 0 || uu >= src.cols) continue;
                    const float z = srow[uu];
                    if (std::isfinite(z)) buf.push_back(z);
                }
            }
            if (static_cast<int>(buf.size()) < min_valid) continue;
            drow[u] = detail::medianInPlace(buf);
        }
    }
    return dst;
}

// Drop "flying pixels": returns that straddle a depth discontinuity and land
// in empty space between the foreground and the background. Every depth
// sensor produces them at object silhouettes, and they are poison for extent
// measurement and for plane fitting.
//
// Tuning note: the threshold must stay ABOVE the natural row-to-row range
// gradient of the ground itself, which grows as z^2 and doubles with every
// decimation step. At 9 m, fy=525, camera 0.8 m up, decimated by 2, adjacent
// rows legitimately differ by ~0.39 m -- a threshold of 0.08*z = 0.72 m
// clears that comfortably, but 0.03*z would delete your entire far field.
inline cv::Mat removeFlyingPixels(const cv::Mat &depth, float rel_thresh,
                                  float abs_thresh) {
    if (rel_thresh <= 0.0f && abs_thresh <= 0.0f) return depth;
    cv::Mat out = depth.clone();
    static const int dv[4] = {-1, 1, 0, 0};
    static const int du[4] = {0, 0, -1, 1};
    for (int v = 0; v < depth.rows; ++v) {
        for (int u = 0; u < depth.cols; ++u) {
            const float z = depth.at<float>(v, u);
            if (!std::isfinite(z)) continue;
            const float lim = std::max(abs_thresh, rel_thresh * z);
            for (int k = 0; k < 4; ++k) {
                const int vv = v + dv[k], uu = u + du[k];
                if (vv < 0 || vv >= depth.rows || uu < 0 || uu >= depth.cols)
                    continue;
                const float zn = depth.at<float>(vv, uu);
                if (std::isfinite(zn) && std::abs(zn - z) > lim) {
                    out.at<float>(v, u) = detail::kNaN();
                    break;
                }
            }
        }
    }
    return out;
}

// Back-project a CV_32F depth image (metres, NaN where invalid) into an
// organised CV_32FC3 cloud. Invalid pixels become NaN in all 3 channels.
inline cv::Mat backprojectDepth(const cv::Mat &depth, const Intrinsics &in) {
    CV_Assert(depth.type() == CV_32F);
    cv::Mat points(depth.size(), CV_32FC3);
    const float nan = detail::kNaN();
    for (int v = 0; v < depth.rows; ++v) {
        const float *drow = depth.ptr<float>(v);
        cv::Vec3f *prow = points.ptr<cv::Vec3f>(v);
        for (int u = 0; u < depth.cols; ++u) {
            const float z = drow[u];
            if (!std::isfinite(z) || z <= 0.0f) {
                prow[u] = cv::Vec3f(nan, nan, nan);
                continue;
            }
            prow[u] = cv::Vec3f((static_cast<float>(u) - in.cx) / in.fx * z,
                                (static_cast<float>(v) - in.cy) / in.fy * z, z);
        }
    }
    return points;
}

// Intersect the ray through pixel (u,v) with the ground plane. Returns
// nullopt when the ray is parallel to or points away from the plane.
inline std::optional<Eigen::Vector3f> rayGroundIntersect(float u, float v,
                                                         const Intrinsics &in,
                                                         const GroundModel &g) {
    const Eigen::Vector3f dir((u - in.cx) / in.fx, (v - in.cy) / in.fy, 1.0f);
    const float denom = dir.dot(g.normal);
    if (std::abs(denom) < 1e-6f) return std::nullopt;
    const float t = g.d / denom;
    if (!std::isfinite(t) || t <= 0.0f) return std::nullopt;
    return Eigen::Vector3f(t * dir);
}

// ---------------------------------------------------------------------
// Ground fitting: MSAC + Huber IRLS + optional quadratic correction
// ---------------------------------------------------------------------
//
// MSAC (M-estimator SAmple Consensus) replaces the plain inlier count with
// a truncated-squared-error score. It costs nothing extra and stops the
// fit from being indifferent between "all inliers at 2.9 cm" and "all
// inliers at 0.2 cm", which plain RANSAC is.
inline std::optional<GroundModel> fitGroundModel(
    const std::vector<Eigen::Vector3f> &pts, const Params &P) {

    const size_t n = pts.size();
    if (n < 50) return std::nullopt;

    Eigen::Vector3f en = P.expected_normal;
    if (en.norm() < 1e-6f) en = Eigen::Vector3f(0.0f, 1.0f, 0.0f);
    en.normalize();
    const float cos_thresh =
        std::cos(P.max_normal_angle_deg * static_cast<float>(M_PI) / 180.0f);
    const float t2 = P.ransac_dist_thresh * P.ransac_dist_thresh;

    std::mt19937 rng(P.seed);
    std::uniform_int_distribution<size_t> pick(0, n - 1);

    float best_score = std::numeric_limits<float>::max();
    size_t best_inliers = 0;
    Eigen::Vector3f best_n(0.0f, 1.0f, 0.0f);
    float best_d = 0.0f;
    bool found = false;

    for (int iter = 0; iter < P.ransac_iterations; ++iter) {
        const size_t i1 = pick(rng), i2 = pick(rng), i3 = pick(rng);
        if (i1 == i2 || i2 == i3 || i1 == i3) continue;

        const Eigen::Vector3f &p1 = pts[i1];
        const Eigen::Vector3f v1 = pts[i2] - p1;
        const Eigen::Vector3f v2 = pts[i3] - p1;
        Eigen::Vector3f nrm = v1.cross(v2);
        const float len = nrm.norm();
        // Reject near-collinear triples relatively, not just absolutely:
        // three points 5 m apart in a near-line still give |cross| >> 1e-8.
        if (len < 1e-4f * v1.norm() * v2.norm() || len < 1e-8f) continue;
        nrm /= len;

        if (std::abs(nrm.dot(en)) < cos_thresh) continue;  // not ground-like

        const float d = nrm.dot(p1);
        float score = 0.0f;
        size_t cnt = 0;
        for (size_t i = 0; i < n; ++i) {
            const float r = pts[i].dot(nrm) - d;
            const float r2 = r * r;
            if (r2 < t2) {
                score += r2;
                ++cnt;
            } else {
                score += t2;
            }
        }
        if (score < best_score) {
            best_score = score;
            best_inliers = cnt;
            best_n = nrm;
            best_d = d;
            found = true;
        }
    }

    if (!found ||
        static_cast<float>(best_inliers) < P.min_inlier_frac * static_cast<float>(n))
        return std::nullopt;

    // --- Huber IRLS refinement -----------------------------------------
    // The old code refit by unweighted PCA over hard inliers. That is fine
    // but throws away the soft boundary: a point at 3.01 cm counts zero, a
    // point at 2.99 cm counts fully. IRLS with a Huber weight gives a
    // smoother, less threshold-sensitive fit. Points beyond 4x the knee
    // are dropped outright so a large rock can't tilt the ground.
    const float k = std::max(P.ransac_dist_thresh, 1e-4f);
    Eigen::Vector3f nrm = best_n;
    float d = best_d;

    for (int it = 0; it < std::max(1, P.irls_iterations); ++it) {
        Eigen::Vector3f centroid = Eigen::Vector3f::Zero();
        float wsum = 0.0f;
        for (size_t i = 0; i < n; ++i) {
            const float r = pts[i].dot(nrm) - d;
            const float ar = std::abs(r);
            if (ar > 4.0f * k) continue;
            const float w = (ar <= k) ? 1.0f : k / ar;
            centroid += w * pts[i];
            wsum += w;
        }
        if (wsum < 1e-6f) break;
        centroid /= wsum;

        Eigen::Matrix3f cov = Eigen::Matrix3f::Zero();
        for (size_t i = 0; i < n; ++i) {
            const float r = pts[i].dot(nrm) - d;
            const float ar = std::abs(r);
            if (ar > 4.0f * k) continue;
            const float w = (ar <= k) ? 1.0f : k / ar;
            const Eigen::Vector3f c = pts[i] - centroid;
            cov.noalias() += w * c * c.transpose();
        }

        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> solver(cov);
        if (solver.info() != Eigen::Success) break;
        // Eigenvalues ascending: the smallest is the least-variance
        // direction, i.e. the plane normal.
        Eigen::Vector3f nn = solver.eigenvectors().col(0).normalized();
        if (nn.dot(nrm) < 0.0f) nn = -nn;  // keep orientation stable
        nrm = nn;
        d = nrm.dot(centroid);
    }

    // The refined normal can drift; re-apply the prior so a bad refinement
    // can't hand back a wall. The original code never re-checked this.
    if (std::abs(nrm.dot(en)) < cos_thresh) return std::nullopt;

    // Orient so the camera origin (0,0,0) has a negative residual.
    if (-d > 0.0f) {
        nrm = -nrm;
        d = -d;
    }

    GroundModel g;
    g.normal = nrm;
    g.d = d;
    makePlaneBasis(g.normal, g.ex, g.ey);
    g.order = 1;

    // --- optional quadratic correction ---------------------------------
    if (P.ground_model_order >= 2) {
        Eigen::Matrix<float, 6, 6> A = Eigen::Matrix<float, 6, 6>::Zero();
        Eigen::Matrix<float, 6, 1> b = Eigen::Matrix<float, 6, 1>::Zero();
        for (size_t i = 0; i < n; ++i) {
            const float r = pts[i].dot(g.normal) - g.d;
            const float ar = std::abs(r);
            if (ar > 4.0f * k) continue;
            const float w = (ar <= k) ? 1.0f : k / ar;
            const float s = pts[i].dot(g.ex), t = pts[i].dot(g.ey);
            Eigen::Matrix<float, 6, 1> f;
            f << 1.0f, s, t, s * s, t * t, s * t;
            A.noalias() += w * f * f.transpose();
            b.noalias() += w * f * r;
        }
        A.diagonal().array() += 1e-4f;  // ridge, for conditioning
        const Eigen::Matrix<float, 6, 1> sol = A.ldlt().solve(b);
        if (sol.allFinite()) {
            g.poly = sol;
            g.order = 2;
        }
    }

    // --- fit quality ----------------------------------------------------
    size_t inl = 0;
    double sq = 0.0;
    for (size_t i = 0; i < n; ++i) {
        const float r = g.residual(pts[i]);
        if (std::abs(r) <= k) {
            ++inl;
            sq += static_cast<double>(r) * r;
        }
    }
    g.inlier_frac = static_cast<float>(inl) / static_cast<float>(n);
    g.rms_m = inl ? static_cast<float>(std::sqrt(sq / static_cast<double>(inl)))
                  : 0.0f;
    return g;
}

// ---------------------------------------------------------------------
// Full pipeline
// ---------------------------------------------------------------------

namespace detail {

// Per-blob point accumulator. Everything is summarised with order
// statistics rather than min/max/mean, so a handful of surviving mixed
// pixels cannot dominate the reported geometry.
struct BlobAcc {
    std::vector<float> xs, ys, zs;  // camera-frame coordinates
    std::vector<float> ss, ts;      // in-plane coordinates, for footprint
    std::vector<float> rr;          // range from camera
    std::vector<float> res;         // ground residual
    int gap_px = 0;                 // pixels contributed by the range shadow
    size_t size() const { return zs.size(); }
};

inline cv::Mat morph(const cv::Mat &m, int op, int k) {
    if (k < 3) return m;
    cv::Mat out;
    cv::Mat kern =
        cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(k | 1, k | 1));
    cv::morphologyEx(m, out, op, kern);
    return out;
}

}  // namespace detail

inline SegmentResult segment(const cv::Mat &depth_in, const Intrinsics &intr,
                             const Params &P = Params()) {
    SegmentResult result;

    // --- 1. units, validity gating, decimation, denoise -----------------
    const cv::Mat depth_m = toMeters(depth_in);
    CV_Assert(depth_m.type() == CV_32F);

    const int f = std::max(1, P.downsample_factor);
    cv::Mat depth =
        decimateValidMedian(depth_m, f, P.min_valid_depth, P.max_valid_depth);
    if (P.median_ksize >= 3) {
        const int ks = P.median_ksize | 1;
        const int min_valid = std::max(1, (ks * ks) / 3);
        for (int pass = 0; pass < std::max(0, P.median_passes); ++pass)
            depth = medianFilterValid(depth, ks, min_valid);
    }
    depth = removeFlyingPixels(depth, P.flying_pixel_rel, P.flying_pixel_abs);

    // Correct intrinsic rescaling. The previous version used cx' = cx * s,
    // which is off by half a pixel at every decimation level; the exact
    // relation for integer block decimation is below.
    Intrinsics eff;
    eff.fx = intr.fx / static_cast<float>(f);
    eff.fy = intr.fy / static_cast<float>(f);
    eff.cx = (intr.cx + 0.5f) / static_cast<float>(f) - 0.5f;
    eff.cy = (intr.cy + 0.5f) / static_cast<float>(f) - 0.5f;

    const int h = depth.rows, w = depth.cols;
    result.depth_used = depth;
    result.scale = 1.0f / static_cast<float>(f);
    result.intrinsics_used = eff;
    result.residuals = cv::Mat(h, w, CV_32F, cv::Scalar(detail::kNaN()));
    result.rock_mask = cv::Mat::zeros(h, w, CV_8U);
    result.crater_mask = cv::Mat::zeros(h, w, CV_8U);
    result.ground_mask = cv::Mat::zeros(h, w, CV_8U);
    result.gap_mask = cv::Mat::zeros(h, w, CV_8U);

    // --- 2. back-projection ---------------------------------------------
    const cv::Mat points = backprojectDepth(depth, eff);

    // --- 3. ground sample ------------------------------------------------
    // Uniform random subsample, not a fixed stride. Striding a row-major
    // compacted list correlates with image columns and can sample a biased
    // vertical band of the scene.
    const int v_start =
        (P.ground_sample_bottom_frac >= 1.0f)
            ? 0
            : std::max(0, static_cast<int>(
                              h * (1.0f - std::max(0.05f,
                                                   P.ground_sample_bottom_frac))));

    std::vector<Eigen::Vector3f> cand;
    cand.reserve(static_cast<size_t>(h - v_start) * w / 2 + 1);
    for (int v = v_start; v < h; ++v) {
        const cv::Vec3f *prow = points.ptr<cv::Vec3f>(v);
        for (int u = 0; u < w; ++u)
            if (std::isfinite(prow[u][2]))
                cand.emplace_back(prow[u][0], prow[u][1], prow[u][2]);
    }

    if (P.max_ransac_points > 0 &&
        static_cast<int>(cand.size()) > P.max_ransac_points) {
        std::mt19937 rng(P.seed ^ 0x9e3779b9u);
        for (int i = 0; i < P.max_ransac_points; ++i) {
            std::uniform_int_distribution<size_t> d(static_cast<size_t>(i),
                                                    cand.size() - 1);
            std::swap(cand[static_cast<size_t>(i)], cand[d(rng)]);
        }
        cand.resize(static_cast<size_t>(P.max_ransac_points));
    }

    auto ground_opt = fitGroundModel(cand, P);
    if (!ground_opt.has_value()) return result;  // masks stay empty
    result.ground = ground_opt;
    const GroundModel &G = *ground_opt;

    // --- 4. residuals + range-adaptive classification --------------------
    for (int v = 0; v < h; ++v) {
        const cv::Vec3f *prow = points.ptr<cv::Vec3f>(v);
        float *res = result.residuals.ptr<float>(v);
        uchar *rock = result.rock_mask.ptr<uchar>(v);
        uchar *crat = result.crater_mask.ptr<uchar>(v);
        uchar *grnd = result.ground_mask.ptr<uchar>(v);
        for (int u = 0; u < w; ++u) {
            const cv::Vec3f &p = prow[u];
            if (!std::isfinite(p[2])) continue;
            const float r =
                G.residual(Eigen::Vector3f(p[0], p[1], p[2]));
            res[u] = r;

            // Per-pixel propagation of range noise into the residual.
            const float z = p[2];
            const float sens =
                std::abs(Eigen::Vector3f(p[0], p[1], p[2]).dot(G.normal)) / z;
            const float noise =
                P.noise_sigmas * sens * P.depth_noise_coeff * z * z;
            const float tr = std::max(P.rock_thresh, noise);
            const float tc = std::max(P.crater_thresh, noise);

            if (r < -tr) {
                // No upper bound here on purpose. The old per-pixel
                // max_rock_height_m gate made tall obstacles invisible;
                // Rock-vs-Wall is now decided per blob instead.
                rock[u] = 255;
            } else if (r > tc && r < P.max_crater_depth_m) {
                crat[u] = 255;
            } else if (r >= -tr && r <= tc) {
                grnd[u] = 255;
            }
        }
    }

    // --- 5. morphology: close, then open ---------------------------------
    // Close first so a blob fragmented by speckle becomes one component,
    // then open to delete what is left of the speckle. Opening alone (the
    // old behaviour) shatters marginal blobs before they can be merged.
    result.rock_mask =
        detail::morph(result.rock_mask, cv::MORPH_CLOSE, P.morph_close_px);
    result.rock_mask =
        detail::morph(result.rock_mask, cv::MORPH_OPEN, P.morph_open_px);
    result.crater_mask =
        detail::morph(result.crater_mask, cv::MORPH_CLOSE, P.morph_close_px);
    result.crater_mask =
        detail::morph(result.crater_mask, cv::MORPH_OPEN, P.morph_open_px);

    // --- 6. occlusion-gap detection for negative obstacles ---------------
    if (P.detect_gaps) {
        cv::Mat rock_dil = result.rock_mask;
        if (P.rock_shadow_dilate_px >= 1) {
            cv::Mat kern = cv::getStructuringElement(
                cv::MORPH_ELLIPSE, cv::Size(P.rock_shadow_dilate_px | 1,
                                            P.rock_shadow_dilate_px | 1));
            cv::dilate(result.rock_mask, rock_dil, kern);
        }

        for (int u = 0; u < w; ++u) {
            int v = h - 1;  // bottom of the image == nearest ground
            while (v >= 0) {
                if (std::isfinite(depth.at<float>(v, u))) {
                    --v;
                    continue;
                }
                const int v_bot = v;
                int v_top = v;
                while (v_top - 1 >= 0 &&
                       !std::isfinite(depth.at<float>(v_top - 1, u)))
                    --v_top;

                const int run = v_bot - v_top + 1;
                const int v_near = v_bot + 1;  // valid pixel below the run
                const int v_far = v_top - 1;   // valid pixel above the run

                // v_far >= 0 means the run does not reach the top of the
                // image, i.e. it is not sky or beyond the far clip.
                bool is_gap = (run >= P.min_gap_px) && (v_near < h) &&
                              (v_far >= 0);
                if (is_gap) {
                    // Near rim must be real ground, not a rock. A rock's own
                    // range shadow is not a hole -- that false positive
                    // would put a phantom crater directly behind every rock.
                    is_gap = result.ground_mask.at<uchar>(v_near, u) &&
                             !rock_dil.at<uchar>(v_near, u);
                }
                if (is_gap) {
                    // How far across the ground does this shadow reach?
                    auto a = rayGroundIntersect(static_cast<float>(u),
                                                static_cast<float>(v_near), eff, G);
                    auto b = rayGroundIntersect(static_cast<float>(u),
                                                static_cast<float>(v_far), eff, G);
                    if (a && b) {
                        const float span = (*b - *a).norm();
                        is_gap = span >= P.min_gap_span_m &&
                                 span <= P.max_gap_span_m;
                    } else {
                        is_gap = false;
                    }
                }
                if (is_gap)
                    for (int vv = v_top; vv <= v_bot; ++vv)
                        result.gap_mask.at<uchar>(vv, u) = 255;

                v = v_top - 1;
            }
        }

        // A crater is one physical object: its far wall (below-ground
        // residuals) plus its shadow. Merge them so they emerge as a single
        // component rather than two half-detections.
        cv::bitwise_or(result.crater_mask, result.gap_mask, result.crater_mask);
        result.crater_mask =
            detail::morph(result.crater_mask, cv::MORPH_CLOSE, P.morph_close_px);
    }

    // --- 7. connected components + metric blob filtering -----------------
    // min_area_px stays a full-resolution API for stability; convert it
    // into decimated-pixel units to match what CC actually reports.
    const int min_area_scaled = std::max(
        1, static_cast<int>(std::lround(static_cast<double>(P.min_area_px) /
                                        (static_cast<double>(f) * f))));

    struct ClassSpec {
        Label base;
        cv::Mat *mask;
    };
    const ClassSpec specs[2] = {{Label::Rock, &result.rock_mask},
                                {Label::Crater, &result.crater_mask}};

    for (const ClassSpec &spec : specs) {
        cv::Mat labels, stats, centroids;
        const int num =
            cv::connectedComponentsWithStats(*spec.mask, labels, stats,
                                             centroids, 8, CV_32S);
        if (num <= 1) continue;

        // Single pass over the label image, accumulating every blob at
        // once. The old code re-scanned the whole image once per blob,
        // which is O(num_blobs * H * W) -- the dominant cost on any frame
        // with more than a handful of detections.
        std::vector<detail::BlobAcc> acc(static_cast<size_t>(num));
        for (int v = 0; v < h; ++v) {
            const int *lrow = labels.ptr<int>(v);
            const cv::Vec3f *prow = points.ptr<cv::Vec3f>(v);
            const float *rrow = result.residuals.ptr<float>(v);
            const uchar *grow = result.gap_mask.ptr<uchar>(v);
            for (int u = 0; u < w; ++u) {
                const int id = lrow[u];
                if (id <= 0) continue;
                if (grow[u]) ++acc[static_cast<size_t>(id)].gap_px;
                const cv::Vec3f &p = prow[u];
                if (!std::isfinite(p[2])) continue;  // gap pixel: no 3D data
                const float r = rrow[u];
                if (!std::isfinite(r)) continue;
                detail::BlobAcc &a = acc[static_cast<size_t>(id)];
                const Eigen::Vector3f pv(p[0], p[1], p[2]);
                a.xs.push_back(p[0]);
                a.ys.push_back(p[1]);
                a.zs.push_back(p[2]);
                a.ss.push_back(pv.dot(G.ex));
                a.ts.push_back(pv.dot(G.ey));
                a.rr.push_back(pv.norm());
                a.res.push_back(r);
            }
        }

        for (int i = 1; i < num; ++i) {
            const int area = stats.at<int>(i, cv::CC_STAT_AREA);
            if (area < min_area_scaled) continue;

            detail::BlobAcc &a = acc[static_cast<size_t>(i)];
            const size_t np = a.size();
            const double cu = centroids.at<double>(i, 0);
            const double cv_ = centroids.at<double>(i, 1);

            // A blob counts as MEASURED when enough real returns back it,
            // and otherwise as INFERRED when it is mostly range shadow.
            // Note the ordering: a mostly-shadow crater that happens to
            // include three below-ground returns must not be rejected for
            // having "too few points" when the same blob with zero points
            // would have been accepted. More evidence must never hurt.
            const bool measured = static_cast<int>(np) >= P.min_points;
            const bool inferred = !measured && a.gap_px >= min_area_scaled;
            if (!measured && !inferred) continue;

            Detection det;
            det.from_gap = !measured;

            if (measured) {
                det.centroid = Eigen::Vector3f(detail::medianInPlace(a.xs),
                                               detail::medianInPlace(a.ys),
                                               detail::medianInPlace(a.zs));
                det.n_points = static_cast<int>(np);

                // Nearest: 2nd percentile of range, not the single minimum.
                // The closest point of an obstacle is exactly where a stray
                // pixel does the most damage to a braking distance.
                // Approximation: the near RANGE is measured honestly, but
                // the bearing is taken from the centroid, so `nearest` is
                // the near range along the blob's centre ray rather than the
                // true nearest point. For a compact blob the two are within
                // half a footprint; for a Wall spanning the frame they are
                // not, so use `distance` for braking and the bbox for
                // steering rather than treating `nearest` as an exact point.
                const float rnear = detail::percentileInPlace(a.rr, 0.02f);
                det.nearest = det.centroid *
                              (rnear / std::max(1e-3f, det.centroid.norm()));

                // Height: 2nd/98th percentile of the residual in the
                // direction that matters for this class. Using the true
                // extremum would let one surviving flying pixel decide
                // whether a rock is classified as a Wall.
                det.height_m = (spec.base == Label::Rock)
                                   ? detail::percentileInPlace(a.res, 0.02f)
                                   : detail::percentileInPlace(a.res, 0.98f);

                // Footprint: 5th-95th percentile spread. This slightly
                // UNDER-states the true extent (~90% of it for a uniform
                // blob), which is the safe direction for a size filter but
                // the unsafe direction if you feed it to a clearance check.
                std::vector<float> s2 = a.ss, t2 = a.ts;
                const float s_lo = detail::percentileInPlace(a.ss, 0.05f);
                const float s_hi = detail::percentileInPlace(s2, 0.95f);
                const float t_lo = detail::percentileInPlace(a.ts, 0.05f);
                const float t_hi = detail::percentileInPlace(t2, 0.95f);
                det.footprint_w_m = std::max(0.0f, s_hi - s_lo);
                det.footprint_l_m = std::max(0.0f, t_hi - t_lo);
            } else {
                // Mostly range shadow: too little was measured inside it to
                // trust. Infer geometry by intersecting the blob's rays with
                // the ground surface. This is an estimate, flagged as such.
                if (spec.base != Label::Crater) continue;
                auto c = rayGroundIntersect(static_cast<float>(cu),
                                            static_cast<float>(cv_), eff, G);
                if (!c.has_value()) continue;
                const int bx = stats.at<int>(i, cv::CC_STAT_LEFT);
                const int by = stats.at<int>(i, cv::CC_STAT_TOP);
                const int bw = stats.at<int>(i, cv::CC_STAT_WIDTH);
                const int bh = stats.at<int>(i, cv::CC_STAT_HEIGHT);
                auto c0 = rayGroundIntersect(static_cast<float>(bx),
                                             static_cast<float>(by + bh), eff, G);
                auto c1 = rayGroundIntersect(static_cast<float>(bx + bw),
                                             static_cast<float>(by + bh), eff, G);
                det.centroid = *c;
                det.nearest = *c;
                det.n_points = static_cast<int>(np);
                det.height_m = P.crater_thresh;  // unknown depth, assume marginal
                det.footprint_w_m =
                    (c0 && c1) ? (*c1 - *c0).norm() : P.min_obstacle_size_m;
                det.footprint_l_m = det.footprint_w_m;
                // A shadow much wider than any plausible hole is far more
                // likely to be a field of stereo dropout on textureless
                // ground. Don't hand the planner a 6 m crater it can't act on.
                if (det.footprint_w_m > P.max_gap_span_m) continue;
            }

            const float extent =
                std::max(det.footprint_w_m, det.footprint_l_m);
            if (extent < P.min_obstacle_size_m) continue;

            // Rock vs Wall decided here, per blob, using metric geometry.
            det.label = spec.base;
            if (spec.base == Label::Rock &&
                (std::abs(det.height_m) > P.wall_height_m ||
                 extent > P.max_obstacle_size_m)) {
                det.label = Label::Wall;
            }

            det.distance = det.nearest.norm();
            det.centroid_distance = det.centroid.norm();

            // Confidence: how well the blob is supported by real returns,
            // combined with how far past the noise floor its height is.
            // A heuristic for ranking and gating, not a probability.
            const float z = std::max(0.1f, det.centroid.z());
            const float sens = std::abs(det.centroid.dot(G.normal)) / z;
            const float thr =
                std::max((spec.base == Label::Rock) ? P.rock_thresh
                                                    : P.crater_thresh,
                         P.noise_sigmas * sens * P.depth_noise_coeff * z * z);
            const float support = det.from_gap ? static_cast<float>(a.gap_px)
                                              : static_cast<float>(np);
            const float fill =
                std::min(1.0f, support / std::max(1.0f, static_cast<float>(area)));
            const float mag =
                std::min(1.0f, std::abs(det.height_m) / (2.0f * thr));
            det.confidence = std::max(0.0f, std::min(1.0f, 0.5f * fill + 0.5f * mag));
            if (det.from_gap) det.confidence *= 0.6f;

            // Back to original full-resolution pixel units.
            const float ff = static_cast<float>(f);
            det.pixel_u = (static_cast<float>(cu) + 0.5f) * ff - 0.5f;
            det.pixel_v = (static_cast<float>(cv_) + 0.5f) * ff - 0.5f;
            det.bbox_x = stats.at<int>(i, cv::CC_STAT_LEFT) * f;
            det.bbox_y = stats.at<int>(i, cv::CC_STAT_TOP) * f;
            det.bbox_w = stats.at<int>(i, cv::CC_STAT_WIDTH) * f;
            det.bbox_h = stats.at<int>(i, cv::CC_STAT_HEIGHT) * f;
            det.area_px = area * f * f;

            result.detections.push_back(det);
        }
    }

    // Nearest hazard first -- that is the one the planner cares about.
    std::sort(result.detections.begin(), result.detections.end(),
              [](const Detection &a, const Detection &b) {
                  return a.distance < b.distance;
              });

    return result;
}


// ---------------------------------------------------------------------
// 10. Temporal tracking
// ---------------------------------------------------------------------
//
// Everything above is single-frame. Single-frame detection flickers: a rock
// at the edge of min_area_px appears and disappears frame to frame, and one
// bad frame can inject a false obstacle straight into the planner. Tracking
// buys three things that per-frame detection cannot provide at all:
//
//   * stable IDs, so a consumer can say "obstacle 7" across time
//   * confirmation, so a hazard must be seen confirm_hits times before it is
//     acted on, which is what actually kills one-frame false positives
//   * coasting, so an obstacle that drops out for a few frames (occluded by
//     a rock in front, or briefly outside the depth range) does not vanish
//     from the world model the moment it stops being visible
//
// FRAME CONVENTION -- read this before using it. Detections come out of
// segment() in the CAMERA optical frame, which moves with the robot. Tracks
// must live in a frame that does not. Pass `cam_to_world` (from tf: the
// camera's pose in odom/map at the image timestamp) on every update. If you
// leave it as identity, tracking is only valid while the robot is
// stationary -- on a moving rover every track will be dragged backwards
// through the world and association will break down at speed.
//
// Terrain does not move, so the motion model is static position with process
// noise, not constant velocity. A constant-velocity filter on a stationary
// rock will happily invent a drift velocity out of measurement noise.

struct TrackerParams {
    // Association gate, widened with range because a detection 8 m away is
    // far less well localised laterally than one at 2 m.
    float gate_base_m = 0.30f;
    float gate_range_frac = 0.06f;

    int confirm_hits = 3;   // sightings before a track is acted on
    int max_misses = 5;     // consecutive misses before a confirmed track dies
    int tentative_misses = 1;  // unconfirmed tracks are dropped fast
    float max_coast_s = 3.0f;  // hard time limit on coasting

    // Static-position Kalman filter. process_noise_m is per sqrt(second) and
    // represents how much a track is allowed to wander (ground model drift,
    // odometry error), NOT physical motion of the terrain.
    float process_noise_m = 0.02f;
    float meas_noise_base_m = 0.03f;
    float meas_noise_range_frac = 0.02f;
    // Gap-inferred detections have no measured 3D points; their position is
    // a ray-plane estimate and deserves a much weaker update.
    float gap_noise_mult = 3.0f;

    float conf_smoothing = 0.3f;   // EMA factor on per-frame confidence
    float geom_smoothing = 0.3f;   // EMA factor on height/footprint
    float merge_dist_m = 0.15f;    // duplicate track collapse distance
};

struct Track {
    int id = -1;
    Label label = Label::Rock;

    Eigen::Vector3f position{0.0f, 0.0f, 0.0f};  // WORLD frame
    Eigen::Matrix3f covariance = Eigen::Matrix3f::Identity();

    float height_m = 0.0f;
    float footprint_w_m = 0.0f, footprint_l_m = 0.0f;
    float confidence = 0.0f;

    int hits = 0, misses = 0, age = 0;
    bool confirmed = false;
    bool visible = false;        // matched to a detection this frame
    double last_update_s = 0.0;

    // Most recent raw detection, still in the CAMERA frame of that update.
    Detection last_detection;

    float positionStdDev() const { return std::sqrt(covariance.trace() / 3.0f); }
};

// Rock and Wall are the same physical thing seen at different sizes, so a
// blob that grows past wall_height_m as the robot approaches must not sever
// its own track. Crater is genuinely a different class.
inline bool labelCompatible(Label a, Label b) {
    if (a == b) return true;
    const bool a_pos = (a == Label::Rock || a == Label::Wall);
    const bool b_pos = (b == Label::Rock || b == Label::Wall);
    return a_pos && b_pos;
}

class TerrainTracker {
  public:
    explicit TerrainTracker(TrackerParams p = TrackerParams()) : P_(p) {}

    void reset() {
        tracks_.clear();
        next_id_ = 1;
        last_stamp_ = -1.0;
    }

    const std::vector<Track> &tracks() const { return tracks_; }

    // The only list a planner should consume.
    std::vector<Track> confirmedTracks() const {
        std::vector<Track> out;
        for (const Track &t : tracks_)
            if (t.confirmed) out.push_back(t);
        return out;
    }

    const std::vector<Track> &update(
        const std::vector<Detection> &dets, double stamp_s,
        const Eigen::Isometry3f &cam_to_world = Eigen::Isometry3f::Identity()) {

        const float dt =
            (last_stamp_ < 0.0)
                ? 0.0f
                : std::max(0.0f, std::min(1.0f,
                                          static_cast<float>(stamp_s - last_stamp_)));
        last_stamp_ = stamp_s;

        // --- predict: static position, growing uncertainty ---------------
        const float q = P_.process_noise_m * P_.process_noise_m * dt;
        for (Track &t : tracks_) {
            t.covariance.diagonal().array() += q;
            t.visible = false;
            ++t.age;
        }

        // --- detections into the world frame -----------------------------
        const size_t nd = dets.size();
        std::vector<Eigen::Vector3f> zs(nd);
        for (size_t i = 0; i < nd; ++i)
            zs[i] = cam_to_world * dets[i].centroid;

        // --- gate, then greedy nearest-neighbour association -------------
        // Greedy on a distance-sorted pair list rather than Hungarian:
        // terrain blobs are sparse and well separated, the optimal
        // assignment almost never differs, and greedy is deterministic and
        // trivially debuggable. Revisit if you ever track dense rock fields.
        struct Pair { float dist; size_t ti, di; };
        std::vector<Pair> pairs;
        for (size_t ti = 0; ti < tracks_.size(); ++ti) {
            for (size_t di = 0; di < nd; ++di) {
                if (!labelCompatible(tracks_[ti].label, dets[di].label)) continue;
                const float dist = (zs[di] - tracks_[ti].position).norm();
                const float gate =
                    P_.gate_base_m + P_.gate_range_frac * dets[di].distance;
                if (dist <= gate) pairs.push_back({dist, ti, di});
            }
        }
        std::sort(pairs.begin(), pairs.end(),
                  [](const Pair &a, const Pair &b) { return a.dist < b.dist; });

        std::vector<int> track_of_det(nd, -1);
        std::vector<char> track_taken(tracks_.size(), 0);
        for (const Pair &pr : pairs) {
            if (track_taken[pr.ti] || track_of_det[pr.di] >= 0) continue;
            track_taken[pr.ti] = 1;
            track_of_det[pr.di] = static_cast<int>(pr.ti);
        }

        // --- update matched tracks ---------------------------------------
        for (size_t di = 0; di < nd; ++di) {
            if (track_of_det[di] < 0) continue;
            Track &t = tracks_[static_cast<size_t>(track_of_det[di])];
            const Detection &d = dets[di];

            float r = P_.meas_noise_base_m +
                      P_.meas_noise_range_frac * d.distance;
            if (d.from_gap) r *= P_.gap_noise_mult;
            const float R = r * r;

            // Scalar-diagonal Kalman update: K = P (P+R)^-1.
            const Eigen::Matrix3f S =
                t.covariance + Eigen::Matrix3f::Identity() * R;
            const Eigen::Matrix3f K = t.covariance * S.inverse();
            t.position += K * (zs[di] - t.position);
            t.covariance =
                (Eigen::Matrix3f::Identity() - K) * t.covariance;

            const float a = P_.geom_smoothing;
            t.height_m = (1 - a) * t.height_m + a * d.height_m;
            t.footprint_w_m = (1 - a) * t.footprint_w_m + a * d.footprint_w_m;
            t.footprint_l_m = (1 - a) * t.footprint_l_m + a * d.footprint_l_m;
            t.confidence = (1 - P_.conf_smoothing) * t.confidence +
                           P_.conf_smoothing * d.confidence;

            // Let the label escalate Rock -> Wall as the blob resolves, but
            // never silently downgrade a Wall to a Rock on one small frame.
            if (d.label == Label::Wall) t.label = Label::Wall;
            else if (t.label != Label::Wall) t.label = d.label;

            t.last_detection = d;
            t.last_update_s = stamp_s;
            t.visible = true;
            ++t.hits;
            t.misses = 0;
            if (t.hits >= P_.confirm_hits) t.confirmed = true;
        }

        // --- spawn tracks for unmatched detections -----------------------
        for (size_t di = 0; di < nd; ++di) {
            if (track_of_det[di] >= 0) continue;
            const Detection &d = dets[di];
            Track t;
            t.id = next_id_++;
            t.label = d.label;
            t.position = zs[di];
            float r = P_.meas_noise_base_m +
                      P_.meas_noise_range_frac * d.distance;
            if (d.from_gap) r *= P_.gap_noise_mult;
            t.covariance = Eigen::Matrix3f::Identity() * (r * r);
            t.height_m = d.height_m;
            t.footprint_w_m = d.footprint_w_m;
            t.footprint_l_m = d.footprint_l_m;
            t.confidence = d.confidence;
            t.hits = 1;
            t.misses = 0;
            t.age = 1;
            t.visible = true;
            t.confirmed = (P_.confirm_hits <= 1);
            t.last_detection = d;
            t.last_update_s = stamp_s;
            tracks_.push_back(t);
        }

        // --- age out unmatched tracks ------------------------------------
        for (size_t ti = 0; ti < tracks_.size(); ++ti)
            if (!tracks_[ti].visible) ++tracks_[ti].misses;

        // --- prune --------------------------------------------------------
        tracks_.erase(
            std::remove_if(
                tracks_.begin(), tracks_.end(),
                [&](const Track &t) {
                    const int limit =
                        t.confirmed ? P_.max_misses : P_.tentative_misses;
                    if (t.misses > limit) return true;
                    // Coasting has a wall-clock limit as well as a frame
                    // count, so a stalled camera cannot keep stale obstacles
                    // alive indefinitely.
                    return (stamp_s - t.last_update_s) > P_.max_coast_s;
                }),
            tracks_.end());

        // --- collapse duplicates -----------------------------------------
        // A blob that splits in two for one frame otherwise leaves a
        // permanent phantom twin sitting next to the real track.
        for (size_t i = 0; i < tracks_.size(); ++i) {
            for (size_t j = i + 1; j < tracks_.size();) {
                const bool close = (tracks_[i].position - tracks_[j].position)
                                       .norm() < P_.merge_dist_m;
                if (close && labelCompatible(tracks_[i].label, tracks_[j].label)) {
                    // Keep the better-supported track, inherit the best of
                    // the other so evidence is never thrown away.
                    if (tracks_[j].hits > tracks_[i].hits) std::swap(tracks_[i], tracks_[j]);
                    tracks_[i].hits += tracks_[j].hits;
                    tracks_[i].confirmed =
                        tracks_[i].confirmed || tracks_[j].confirmed;
                    if (tracks_[i].hits >= P_.confirm_hits) tracks_[i].confirmed = true;
                    tracks_.erase(tracks_.begin() + static_cast<long>(j));
                } else {
                    ++j;
                }
            }
        }

        return tracks_;
    }

  private:
    TrackerParams P_;
    std::vector<Track> tracks_;
    int next_id_ = 1;
    double last_stamp_ = -1.0;
};

// ---------------------------------------------------------------------
// Debug visualisation (tuning aid, not part of the detection path)
// ---------------------------------------------------------------------

// Blue = below ground (crater), red = above ground (rock), grey = ground,
// black = no return. `span` is the residual in metres mapped to full
// saturation.
inline cv::Mat colorizeResiduals(const SegmentResult &r, float span = 0.25f) {
    cv::Mat out(r.residuals.size(), CV_8UC3, cv::Scalar(0, 0, 0));
    for (int v = 0; v < r.residuals.rows; ++v) {
        const float *res = r.residuals.ptr<float>(v);
        cv::Vec3b *orow = out.ptr<cv::Vec3b>(v);
        const uchar *gap = r.gap_mask.empty() ? nullptr : r.gap_mask.ptr<uchar>(v);
        for (int u = 0; u < r.residuals.cols; ++u) {
            if (gap && gap[u]) {
                orow[u] = cv::Vec3b(255, 0, 255);  // magenta: inferred hole
                continue;
            }
            const float x = res[u];
            if (!std::isfinite(x)) continue;
            const float t =
                std::max(-1.0f, std::min(1.0f, x / std::max(1e-6f, span)));
            const uchar m = static_cast<uchar>(60 + 195 * std::abs(t));
            orow[u] = (t < 0) ? cv::Vec3b(60, 60, m)     // BGR: red-ish
                              : cv::Vec3b(m, 60, 60);    // BGR: blue-ish
        }
    }
    return out;
}

}  // namespace terrain_features
