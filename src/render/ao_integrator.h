// Ambient-occlusion ray evaluation, independent of framebuffer ownership and parallel dispatch.
#pragma once

#include "random.h"
#include "scene_intersect.h"

namespace happyray {

struct AOIntegrator {
    int samples_per_pixel_per_frame = 1;
    // Callers with a scene should use DefaultMaxDistance(scene) from scene.h,
    // which scales the occlusion reach to 0.1 times the longest scene bounding-box side.
    float max_distance = 2.0f;
    Vec3 background_color = Vec3(0.05f, 0.05f, 0.08f);

    // Averages cosine-weighted visibility and tints it by the surface's diffuse color.
    // sample_offset and stable per-pixel jitter preserve progressive Halton sampling.
    HAPPYRAY_HD Vec3 Integrate(const SceneView& scene, const AccelStructures& accel,
                          const Ray& primary_ray, uint32_t sample_offset,
                          const Vec2& pixel_jitter) const {
        HitRecord hit;
        if (!IntersectScene(scene, accel, primary_ray, &hit)) {
            return background_color;
        }

        int unoccluded = 0;
        for (int i = 0; i < samples_per_pixel_per_frame; ++i) {
            const Vec2 u = Halton2D(sample_offset + static_cast<uint32_t>(i), pixel_jitter);
            const Vec3 sample_dir = SampleCosineHemisphere(hit.normal, u.x, u.y);
            const Ray occlusion_ray(hit.position + hit.normal * 1e-3f, sample_dir, 1e-4f,
                                     max_distance);
            if (!IsOccluded(scene, accel, occlusion_ray)) {
                ++unoccluded;
            }
        }
        const float ao = static_cast<float>(unoccluded) / samples_per_pixel_per_frame;
        return scene.material_diffuse_colors[hit.material_id] * ao;
    }
};

}  // namespace happyray
