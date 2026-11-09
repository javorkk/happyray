// Procedural test scenes: a Lorenz attractor trajectory as points, spheres, or Bezier curves.
#pragma once

#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>

#include "mesh.h"

namespace happyray
{

	// Denser trajectories overlap heavily at screen-space radii, inflating grid references.
	constexpr uint32_t kDefaultAttractorSampleCount = 20000;

	// Parses "attractor:<points|spheres|curves>[:count]". Returns false for other inputs.
	inline bool ParseAttractorSpec(const std::string &spec, PrimitiveType *type, uint32_t *count)
	{
		const std::string prefix = "attractor:";
		if (spec.compare(0, prefix.size(), prefix) != 0)
			return false;
		const std::string rest	= spec.substr(prefix.size());
		const size_t	  colon = rest.find(':');
		const std::string kind	= rest.substr(0, colon);
		if (kind == "points")
			*type = PrimitiveType::kPoint;
		else if (kind == "spheres")
			*type = PrimitiveType::kSphere;
		else if (kind == "curves")
			*type = PrimitiveType::kBezierCurve;
		else
			return false;
		*count = kDefaultAttractorSampleCount;
		if (colon != std::string::npos)
		{
			const long parsed = std::strtol(rest.c_str() + colon + 1, nullptr, 10);
			if (parsed <= 0)
				return false;
			*count = static_cast<uint32_t>(parsed);
		}
		return true;
	}

	// RK4-integrated Lorenz system, sampled every `dt` after a transient.
	inline std::vector<Vec3f> LorenzTrajectory(uint32_t count, float dt = 0.005f)
	{
		constexpr float sigma = 10.0f, rho = 28.0f, beta = 8.0f / 3.0f;
		auto			derivative = [&](const Vec3f &p)
		{ return Vec3f(sigma * (p.y - p.x), p.x * (rho - p.z) - p.y, p.x * p.y - beta * p.z); };
		Vec3f			   p(1.0f, 1.0f, 1.0f);
		std::vector<Vec3f> samples;
		samples.reserve(count);
		for (uint32_t step = 0; samples.size() < count; ++step)
		{
			const Vec3f k1 = derivative(p);
			const Vec3f k2 = derivative(p + k1 * (0.5f * dt));
			const Vec3f k3 = derivative(p + k2 * (0.5f * dt));
			const Vec3f k4 = derivative(p + k3 * dt);
			p += (k1 + k2 * 2.0f + k3 * 2.0f + k4) * (dt / 6.0f);
			if (step >= 1000)
				samples.push_back(p);
		}
		return samples;
	}

	// `count` primitives along the trajectory, colored by time. Point/curve radii are world-space
	// fallbacks; the renderer replaces them with screen-space radii.
	inline Scene MakeAttractorScene(PrimitiveType type, uint32_t count = kDefaultAttractorSampleCount)
	{
		constexpr float kSphereRadius = 0.1f;
		constexpr float kThinRadius	  = 0.03f;
		constexpr int	kColorCount	  = 8;

		Scene scene;
		for (int i = 0; i < kColorCount; ++i)
		{
			const float s = static_cast<float>(i) / (kColorCount - 1);
			Material	material;
			material.name		   = "attractor" + std::to_string(i);
			material.diffuse_color = Lerp(Vec3f(0.15f, 0.35f, 0.9f), Vec3f(0.95f, 0.55f, 0.15f), s);
			scene.materials.push_back(material);
		}

		const bool				 curves	 = type == PrimitiveType::kBezierCurve;
		const std::vector<Vec3f> samples = LorenzTrajectory(curves ? count + 3 : count);
		for (uint32_t i = 0; i < count; ++i)
		{
			scene.primitive_material_ids.push_back(static_cast<uint32_t>(uint64_t(i) * kColorCount / count));
			switch (type)
			{
			case PrimitiveType::kPoint:
				scene.points.emplace_back(samples[i], kThinRadius);
				break;
			case PrimitiveType::kSphere:
				scene.spheres.emplace_back(samples[i], kSphereRadius);
				break;
			default:
			{
				// Catmull-Rom span samples[i + 1] -> samples[i + 2] as a cubic Bezier.
				const Vec3f &p0 = samples[i], &p1 = samples[i + 1], &p2 = samples[i + 2], &p3 = samples[i + 3];
				scene.curves.emplace_back(p1, p1 + (p2 - p0) * (1.0f / 6.0f), p2 - (p3 - p1) * (1.0f / 6.0f), p2, kThinRadius);
				break;
			}
			}
		}
		return scene;
	}

} // namespace happyray
