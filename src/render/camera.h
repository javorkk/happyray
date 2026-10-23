// Pinhole camera with FPS-style fly controls (position + yaw/pitch), used by the interactive
// viewer. Shared by the CPU and (later) GPU render paths since `GenerateRay` is a plain,
// host/device-compatible function of camera state.
//
// Orientation is stored as yaw/pitch degrees (simple to clamp pitch with and free of any
// accumulation drift), but the actual rotation math -- turning those angles into the camera's
// forward/right/up basis -- goes through a quaternion (see Orientation()). This keeps the
// camera itself gimbal-safe/composable and reuses the same Quaternion type that will later
// drive keyframed camera/mesh animation and GPU-side basis computation.
#pragma once

#include "mat4.h"
#include "quaternion.h"
#include "ray.h"
#include "vec.h"

namespace happyray {

struct Camera {
    Vec3 position = Vec3(0.0f, 0.0f, 3.0f);
    float yaw_deg = -90.0f;    // -90 so the camera starts looking down -Z.
    float pitch_deg = 0.0f;
    float vertical_fov_deg = 60.0f;

    // World-space orientation: local-space pitch around Z (tilting the initial +X "forward"
    // into the XY-plane), then yaw around world -Y. This specific axis/sign/order combination
    // was chosen (and verified numerically) to reproduce the exact same forward/right/up basis
    // as a direct spherical-coordinates formula, so swapping the implementation to go through
    // quaternions didn't change the camera's behavior.
    HAPPYRAY_HD Quaternion Orientation() const {
        constexpr float kDegToRad = 3.14159265358979323846f / 180.0f;
        const Quaternion yaw_rotation =
            Quaternion::FromAxisAngle(Vec3(0.0f, 1.0f, 0.0f), -yaw_deg * kDegToRad);
        const Quaternion pitch_rotation =
            Quaternion::FromAxisAngle(Vec3(0.0f, 0.0f, 1.0f), pitch_deg * kDegToRad);
        return Normalize(yaw_rotation * pitch_rotation);
    }

    HAPPYRAY_HD Vec3 Forward() const { return RotateVector(Orientation(), Vec3(1.0f, 0.0f, 0.0f)); }
    HAPPYRAY_HD Vec3 Right() const { return RotateVector(Orientation(), Vec3(0.0f, 0.0f, 1.0f)); }
    HAPPYRAY_HD Vec3 Up() const { return RotateVector(Orientation(), Vec3(0.0f, 1.0f, 0.0f)); }

    // `pixel_x`/`pixel_y` may be fractional (for supersampling); (0,0) is the top-left pixel.
    HAPPYRAY_HD Ray GenerateRay(float pixel_x, float pixel_y, int width, int height) const {
        const float aspect = static_cast<float>(width) / static_cast<float>(height);
        const float tan_half_fov =
            std::tan(0.5f * vertical_fov_deg * (3.14159265358979323846f / 180.0f));

        const float ndc_x = (2.0f * (pixel_x + 0.5f) / width - 1.0f) * aspect * tan_half_fov;
        const float ndc_y = (1.0f - 2.0f * (pixel_y + 0.5f) / height) * tan_half_fov;

        const Vec3 dir = Normalize(Forward() + Right() * ndc_x + Up() * ndc_y);
        return Ray(position, dir);
    }

    void MoveForward(float distance) { position += Forward() * distance; }
    void MoveRight(float distance) { position += Right() * distance; }
    void MoveUp(float distance) { position += Vec3(0.0f, 1.0f, 0.0f) * distance; }

    void Look(float yaw_delta_deg, float pitch_delta_deg) {
        yaw_deg += yaw_delta_deg;
        pitch_deg += pitch_delta_deg;
        const float kMaxPitch = 89.0f;
        pitch_deg = pitch_deg > kMaxPitch ? kMaxPitch : (pitch_deg < -kMaxPitch ? -kMaxPitch : pitch_deg);
    }
};

}  // namespace happyray
