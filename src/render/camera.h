// Pinhole camera with fly controls. Orientation is stored as yaw/pitch (easy to clamp, no drift)
// and converted to a basis through a quaternion.
#pragma once

#include "../core/mat4.h"
#include "../core/quaternion.h"
#include "../core/ray.h"
#include "../core/vec.h"

namespace happyray
{

	struct Camera
	{
		Vec3f position		   = Vec3f(0.0f, 0.0f, 3.0f);
		float yaw_deg		   = -90.0f; // -90 so the camera starts looking down -Z.
		float pitch_deg		   = 0.0f;
		float vertical_fov_deg = 60.0f;

		// Pitch around local Z, then yaw around world -Y; matches the spherical-coordinates basis.
		HAPPYRAY_HD Quaternion Orientation() const
		{
			constexpr float	 kDegToRad		= 3.14159265358979323846f / 180.0f;
			const Quaternion yaw_rotation	= Quaternion::FromAxisAngle(Vec3f(0.0f, 1.0f, 0.0f), -yaw_deg * kDegToRad);
			const Quaternion pitch_rotation = Quaternion::FromAxisAngle(Vec3f(0.0f, 0.0f, 1.0f), pitch_deg * kDegToRad);
			return Normalize(yaw_rotation * pitch_rotation);
		}

		HAPPYRAY_HD Vec3f Forward() const
		{
			return RotateVector(Orientation(), Vec3f(1.0f, 0.0f, 0.0f));
		}
		HAPPYRAY_HD Vec3f Right() const
		{
			return RotateVector(Orientation(), Vec3f(0.0f, 0.0f, 1.0f));
		}
		HAPPYRAY_HD Vec3f Up() const
		{
			return RotateVector(Orientation(), Vec3f(0.0f, 1.0f, 0.0f));
		}

		// `pixel_x`/`pixel_y` may be fractional (for supersampling); (0,0) is the top-left pixel.
		HAPPYRAY_HD Ray GenerateRay(float pixel_x, float pixel_y, int width, int height) const
		{
			const float aspect		 = static_cast<float>(width) / static_cast<float>(height);
			const float tan_half_fov = std::tan(0.5f * vertical_fov_deg * (3.14159265358979323846f / 180.0f));

			const float ndc_x = (2.0f * (pixel_x + 0.5f) / width - 1.0f) * aspect * tan_half_fov;
			const float ndc_y = (1.0f - 2.0f * (pixel_y + 0.5f) / height) * tan_half_fov;

			const Vec3f dir = Normalize(Forward() + Right() * ndc_x + Up() * ndc_y);
			return Ray(position, dir);
		}

		void MoveForward(float distance)
		{
			position += Forward() * distance;
		}
		void MoveRight(float distance)
		{
			position += Right() * distance;
		}
		void MoveUp(float distance)
		{
			position += Vec3f(0.0f, 1.0f, 0.0f) * distance;
		}

		void Look(float yaw_delta_deg, float pitch_delta_deg)
		{
			yaw_deg += yaw_delta_deg;
			pitch_deg += pitch_delta_deg;
			const float kMaxPitch = 89.0f;
			pitch_deg = pitch_deg > kMaxPitch ? kMaxPitch : (pitch_deg < -kMaxPitch ? -kMaxPitch : pitch_deg);
		}
	};

} // namespace happyray
