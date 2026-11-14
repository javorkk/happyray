// Cartoon shading following happyray's RenderBug fragment shader.
#pragma once

#include "../accel/intersect_common.h"

namespace happyray
{

	struct SimpleIntegrator
	{
		Vec3f light_direction  = Vec3f(0.0f, 0.0f, 1.0f);
		Vec3f background_color = Vec3f(0.05f, 0.05f, 0.08f);

		template <typename P = DynamicPrimitive, typename A = DynamicAccel>
		HAPPYRAY_HD Vec3f Integrate(const SceneView		  &scene,
									const AccelStructures &accel,
									const Ray			  &primary_ray,
									uint32_t /*sample_offset*/,
									const Vec2f & /*pixel_jitter*/) const
		{
			HitRecord hit;
			if (!IntersectScene<P, A>(scene, accel, primary_ray, &hit))
			{
				return background_color;
			}

			const float intensity = -Dot(primary_ray.direction, Normalize(hit.normal));
			const float shade	  = 1.f / (1.f + std::exp(-intensity * 2.f)); // Sigmoid for cartoon shading.
			return scene.material_diffuse_colors[hit.material_id] * shade;
		}
	};

} // namespace happyray
