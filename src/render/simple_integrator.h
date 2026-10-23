// Cartoon shading following happyray's RenderBug fragment shader.
#pragma once

#include "scene_intersect.h"

namespace happyray {

struct SimpleIntegrator {
    Vec3 light_direction = Vec3(0.0f, 0.0f, 1.0f);
    Vec3 background_color = Vec3(0.05f, 0.05f, 0.08f);

    HAPPYRAY_HD Vec3 Integrate(const SceneView& scene, const AccelStructures& accel,
                          const Ray& primary_ray, uint32_t /*sample_offset*/,
                          const Vec2& /*pixel_jitter*/) const {
        HitRecord hit;
        if (!IntersectScene(scene, accel, primary_ray, &hit)) {
            return background_color;
        }
        const float intensity = Dot(Normalize(light_direction), Normalize(hit.normal));
        const float shade = intensity > 0.95f ? 1.2f : (intensity > 0.25f ? intensity : 0.1f);
        return scene.material_diffuse_colors[hit.material_id] * shade;
    }
};

}  // namespace happyray
