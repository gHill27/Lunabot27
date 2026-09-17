#pragma once
#include "terrain_features.hpp"

// Ray-march a terrain height field so occlusion, foreshortening and range
// shadows are all physically correct, rather than warping a flat-ground
// solution (which silently places a crater floor metres past the hole).
struct Scene {
    float cam_h = 0.8f;          // camera height above the datum, metres
    bool rock = false;           // 0.15 m hemisphere at (0.3, 4.0)
    bool crater = false;         // 0.6 m wide, 0.4 m deep pit at (-0.5, 5.0)
    bool slope = false;          // 12 deg up-slope beyond z = 3 m
    bool slab = false;           // 1.2 m vertical slab at z = 3 m
    float pit_return_prob = 0.0f;  // chance a ray hitting inside the pit
                                   // returns anything at all (shadowed)
    float pitch_down_deg = 0.0f;   // genuine camera rotation about its X axis

    // Terrain height in the camera's y-down convention: larger = lower.
    float surfaceY(float x, float z) const {
        float y = cam_h;
        if (slope) y -= 0.21f * std::max(0.0f, z - 3.0f);
        if (crater) {
            const float dx = x + 0.5f, dz = z - 5.0f;
            if (dx * dx + dz * dz < 0.30f * 0.30f) y += 0.40f;
        }
        if (rock) {
            const float dx = x - 0.3f, dz = z - 4.0f;
            const float rr2 = dx * dx + dz * dz;
            if (rr2 < 0.15f * 0.15f) y -= std::sqrt(0.15f * 0.15f - rr2);
        }
        return y;
    }
    bool insidePit(float x, float z) const {
        if (!crater) return false;
        const float dx = x + 0.5f, dz = z - 5.0f;
        return dx * dx + dz * dz < 0.30f * 0.30f;
    }
};

inline cv::Mat renderScene(const Scene &S, const terrain_features::Intrinsics &in,
                           int W, int H, float noise_coeff, unsigned seed = 7) {
    const float nan = std::numeric_limits<float>::quiet_NaN();
    cv::Mat depth(H, W, CV_32F, cv::Scalar(nan));
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::uniform_real_distribution<float> ud(0.0f, 1.0f);

    for (int v = 0; v < H; ++v) {
        for (int u = 0; u < W; ++u) {
            const float dx = (u - in.cx) / in.fx, dy = (v - in.cy) / in.fy;
            // Rotate the camera ray into the level world frame. R_x(-pitch)
            // maps camera-frame down (0,cos p,sin p) onto world down (0,1,0).
            const float a = -S.pitch_down_deg * float(M_PI) / 180.0f;
            const float ca = std::cos(a), sa = std::sin(a);
            const float gx = dx;
            const float gy = ca * dy - sa * 1.0f;
            const float gz = sa * dy + ca * 1.0f;

            float hit = -1.0f;
            // Opaque vertical slab across the middle of the frame at z = 3.
            if (S.slab && u >= 200 && u < 440) {
                const float y = dy * 3.0f;
                if (y < S.cam_h && y > S.cam_h - 1.2f) hit = 3.0f;
            }
            // March the ray parameter t; reported depth is the CAMERA-frame
            // z, which is exactly t because the camera-frame dir has z = 1.
            if (hit < 0.0f && gy > 1e-4f && gz > 1e-4f) {
                float prev = gy * 0.2f - S.surfaceY(gx * 0.2f, gz * 0.2f);
                for (float t = 0.21f; t < 12.0f; t += 0.01f) {
                    const float gap =
                        gy * t - S.surfaceY(gx * t, gz * t);  // >0 = below
                    if (gap >= 0.0f) {
                        const float f =
                            (prev == gap) ? 0.0f : -prev / (gap - prev);
                        hit = t - 0.01f + 0.01f * f;
                        break;
                    }
                    prev = gap;
                }
            }
            if (hit <= 0.0f || hit > 9.0f) continue;
            if (S.insidePit(gx * hit, gz * hit) && ud(rng) > S.pit_return_prob)
                continue;
            depth.at<float>(v, u) = hit + noise_coeff * nd(rng) * hit * hit;
        }
    }
    return depth;
}
