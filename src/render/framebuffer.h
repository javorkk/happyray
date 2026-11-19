// Progressive framebuffer: per-pixel sample sums, the Halton jitter table, the preview image, and
// a per-pixel error estimate used to decide when a still image has converged.
#pragma once

#include <cstdint>
#include <limits>
#include <memory>
#include <vector>

#include "../core/vec.h"

namespace happyray
{

	HAPPYRAY_HD inline float PixelLuminance(const Vec3f &color)
	{
		return (color.x + color.y + color.z) / 3.0f;
	}

	// Backend pointers for one frame's render kernel (see Framebuffer::BeginFrame).
	struct FramebufferView
	{
		Vec3f		*accumulated  = nullptr;
		const Vec2f *pixel_jitter = nullptr;
		// Luminance sum of the half estimate (A or B) this frame adds to; null unless this frame's
		// samples go into the error estimate.
		float *half_estimate_sum = nullptr;
		bool   first_half_sample = false; // Overwrite whatever a reset left.

		HAPPYRAY_HD void AddSample(uint32_t pixel, const Vec3f &sample) const
		{
			accumulated[pixel] += sample;
			if (!half_estimate_sum)
				return;
			const float luminance	 = PixelLuminance(sample);
			half_estimate_sum[pixel] = first_half_sample ? luminance : half_estimate_sum[pixel] + luminance;
		}
	};

	// Each frame adds one sample per pixel. While error tracking is enabled, frames passed to
	// BeginFrame as error samples alternate between two half estimates A and B of each pixel's
	// luminance (r+g+b)/3. After an even number n of them, the average (A+B)/2 differs from A by
	// (A-B)/2, which is distributed like the error of the average itself, so a pixel's error is
	// |A-B|/2 = |sum_A - sum_B| / n with no assumption about the convergence rate. The samples of A
	// and B must each cover the sampled domain evenly (hence Halton bases 3 and 5, see random.h).
	// The image has converged once, after at least kMinErrorSamples error samples, the mean pixel
	// error is below TargetPixelError().
	class Framebuffer
	{
	  public:
		static constexpr uint32_t kMinErrorSamples		   = 16; // Even, so A and B match.
		static constexpr float	  kDefaultTargetPixelError = 0.01f;

		Framebuffer();
		~Framebuffer();
		Framebuffer(Framebuffer &&) noexcept;
		Framebuffer &operator=(Framebuffer &&) noexcept;
		Framebuffer(const Framebuffer &)			= delete;
		Framebuffer &operator=(const Framebuffer &) = delete;

		// Clears everything and regenerates the per-pixel jitter table (see random.h).
		void Resize(int width, int height);
		// Clears the accumulated samples and the error estimate; keeps the jitter table.
		void Reset();

		int		 Width() const;
		int		 Height() const;
		uint32_t PixelCount() const;
		// Frames since the last Reset or Resize.
		uint32_t SampleCount() const;

		// `error_sample`: whether this frame's samples may enter the error estimate (only used while
		// tracking is enabled). Call EndFrame after the kernel has run.
		FramebufferView BeginFrame(bool error_sample);
		void			EndFrame();

		// Resolves the accumulated average over `accumulated_frames` into ResolvedPixels().
		void Resolve(uint32_t accumulated_frames);
		// Same, displayed as Lerp(preview, average, average_weight).
		void ResolveOverPreview(uint32_t accumulated_frames, float average_weight);
		// Keeps the resolved image as the preview and zeroes the sums, leaving SampleCount().
		void StorePreview();
		const std::vector<Vec3f> &ResolvedPixels() const;

		// Changing it clears the error estimate. Disabled by default.
		void	 SetErrorTrackingEnabled(bool enabled);
		bool	 ErrorTrackingEnabled() const;
		void	 ResetErrorEstimate();
		uint32_t ErrorSampleCount() const;

		// Must be finite and nonnegative; 0 never converges. Default kDefaultTargetPixelError.
		void  SetTargetPixelError(float target_pixel_error);
		float TargetPixelError() const;
		// Mean pixel error, updated after each even error sample count; infinity before
		// kMinErrorSamples.
		float AveragePixelError() const;
		bool  Converged() const;

	  private:
		struct Storage;
		std::unique_ptr<Storage> storage_;
		int						 width_					= 0;
		int						 height_				= 0;
		uint32_t				 sample_count_			= 0;
		bool					 error_tracking_enabled_ = false;
		bool					 frame_tracks_error_	 = false;
		uint32_t				 error_samples_			 = 0;
		float					 target_pixel_error_	= kDefaultTargetPixelError;
		float					 pixel_error_			= std::numeric_limits<float>::infinity();
	};

} // namespace happyray
