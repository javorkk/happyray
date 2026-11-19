// Framebuffer storage and reductions in Thrust, so it builds for the CUDA and TBB backends.
#if HAPPYRAY_THRUST_BACKEND_IS_TBB
#define THRUST_HOST_SYSTEM THRUST_HOST_SYSTEM_CPP
#define THRUST_DEVICE_SYSTEM THRUST_DEVICE_SYSTEM_TBB
#endif

#include "framebuffer.h"

#include <cmath>
#include <stdexcept>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <thrust/for_each.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform_reduce.h>

#include "../core/random.h"

namespace happyray
{
	namespace
	{

		struct ResolveFunctor
		{
			float		 inv_sample_count;
			const Vec3f *accumulated;
			Vec3f		*resolved;
			const Vec3f *preview;
			float		 average_weight;

			HAPPYRAY_HD void operator()(uint32_t pixel) const
			{
				const Vec3f average = accumulated[pixel] * inv_sample_count;
				resolved[pixel]		= preview ? Lerp(preview[pixel], average, average_weight) : average;
			}
		};

		// |A-B|/2 for one pixel; see the Framebuffer class comment.
		struct PixelErrorFunctor
		{
			const float *sum_a;
			const float *sum_b;
			float		 inv_sample_count; // 1 / n, n counting both halves

			HAPPYRAY_HD float operator()(uint32_t pixel) const
			{
				return fabsf(sum_a[pixel] - sum_b[pixel]) * inv_sample_count;
			}
		};

	} // namespace

	struct Framebuffer::Storage
	{
		thrust::device_vector<Vec3f> accumulated;
		thrust::device_vector<Vec2f> pixel_jitter;
		thrust::device_vector<Vec3f> resolved_device;
		thrust::device_vector<Vec3f> preview;
		// Luminance sums of the half estimates A (even error samples) and B (odd ones).
		thrust::device_vector<float> half_estimate_sums[2];
		std::vector<Vec3f>			 resolved_host;
	};

	Framebuffer::Framebuffer() : storage_(std::make_unique<Storage>()) {}
	Framebuffer::~Framebuffer()								= default;
	Framebuffer::Framebuffer(Framebuffer &&) noexcept			= default;
	Framebuffer &Framebuffer::operator=(Framebuffer &&) noexcept = default;

	void Framebuffer::Resize(int width, int height)
	{
		if (width < 0 || height < 0)
			throw std::invalid_argument("Framebuffer dimensions must be nonnegative");
		width_					 = width;
		height_					 = height;
		const size_t pixel_count = PixelCount();
		storage_->accumulated.assign(pixel_count, Vec3f(0.0f, 0.0f, 0.0f));
		storage_->resolved_device.resize(pixel_count);
		storage_->resolved_host.resize(pixel_count);
		storage_->preview.clear();
		for (auto &sums : storage_->half_estimate_sums)
			sums.resize(pixel_count);
		sample_count_ = 0;
		ResetErrorEstimate();

		// Per-pixel offset that decorrelates the Halton sequence (see random.h).
		std::vector<Vec2f> jitter(pixel_count);
		for (int y = 0; y < height; ++y)
		{
			for (int x = 0; x < width; ++x)
			{
				Rng rng(HashSeed(static_cast<uint32_t>(x), static_cast<uint32_t>(y), 0u),
						static_cast<uint64_t>(y) * width + x);
				jitter[static_cast<size_t>(y) * width + x] = Vec2f(rng.NextFloat(), rng.NextFloat());
			}
		}
		storage_->pixel_jitter.assign(jitter.begin(), jitter.end());
	}

	void Framebuffer::Reset()
	{
		thrust::fill(storage_->accumulated.begin(), storage_->accumulated.end(), Vec3f(0.0f, 0.0f, 0.0f));
		sample_count_ = 0;
		ResetErrorEstimate();
	}

	int Framebuffer::Width() const
	{
		return width_;
	}

	int Framebuffer::Height() const
	{
		return height_;
	}

	uint32_t Framebuffer::PixelCount() const
	{
		return static_cast<uint32_t>(width_) * static_cast<uint32_t>(height_);
	}

	uint32_t Framebuffer::SampleCount() const
	{
		return sample_count_;
	}

	FramebufferView Framebuffer::BeginFrame(bool error_sample)
	{
		FramebufferView view;
		view.accumulated	= thrust::raw_pointer_cast(storage_->accumulated.data());
		view.pixel_jitter	= thrust::raw_pointer_cast(storage_->pixel_jitter.data());
		frame_tracks_error_ = error_tracking_enabled_ && error_sample;
		if (frame_tracks_error_)
		{
			view.half_estimate_sum = thrust::raw_pointer_cast(storage_->half_estimate_sums[error_samples_ % 2].data());
			view.first_half_sample = error_samples_ < 2;
		}
		return view;
	}

	void Framebuffer::EndFrame()
	{
		++sample_count_;
		if (!frame_tracks_error_)
			return;
		frame_tracks_error_ = false;
		++error_samples_;
		// Only equal halves give |A-B|/2 the distribution of the average's error.
		if (error_samples_ < kMinErrorSamples || error_samples_ % 2 != 0 || PixelCount() == 0)
			return;
		const PixelErrorFunctor error{thrust::raw_pointer_cast(storage_->half_estimate_sums[0].data()),
									  thrust::raw_pointer_cast(storage_->half_estimate_sums[1].data()),
									  1.0f / static_cast<float>(error_samples_)};
		const float				error_sum = thrust::transform_reduce(thrust::counting_iterator<uint32_t>(0),
																	 thrust::counting_iterator<uint32_t>(PixelCount()),
																	 error,
																	 0.0f,
																	 thrust::plus<float>());
		pixel_error_					  = error_sum / static_cast<float>(PixelCount());
	}

	void Framebuffer::Resolve(uint32_t accumulated_frames)
	{
		ResolveOverPreview(accumulated_frames, 1.0f);
	}

	void Framebuffer::ResolveOverPreview(uint32_t accumulated_frames, float average_weight)
	{
		if (PixelCount() == 0)
			return;
		const bool			 blend = average_weight < 1.0f;
		const ResolveFunctor resolve{1.0f / static_cast<float>(accumulated_frames),
									 thrust::raw_pointer_cast(storage_->accumulated.data()),
									 thrust::raw_pointer_cast(storage_->resolved_device.data()),
									 blend ? thrust::raw_pointer_cast(storage_->preview.data()) : nullptr,
									 average_weight};
		if (blend && storage_->preview.size() != PixelCount())
			throw std::logic_error("Blending over a preview requires StorePreview first");
		thrust::for_each(thrust::counting_iterator<uint32_t>(0), thrust::counting_iterator<uint32_t>(PixelCount()), resolve);
		thrust::copy(storage_->resolved_device.begin(), storage_->resolved_device.end(), storage_->resolved_host.begin());
	}

	void Framebuffer::StorePreview()
	{
		storage_->preview = storage_->resolved_device;
		thrust::fill(storage_->accumulated.begin(), storage_->accumulated.end(), Vec3f(0.0f, 0.0f, 0.0f));
	}

	const std::vector<Vec3f> &Framebuffer::ResolvedPixels() const
	{
		return storage_->resolved_host;
	}

	void Framebuffer::SetErrorTrackingEnabled(bool enabled)
	{
		if (enabled == error_tracking_enabled_)
			return;
		error_tracking_enabled_ = enabled;
		ResetErrorEstimate();
	}

	bool Framebuffer::ErrorTrackingEnabled() const
	{
		return error_tracking_enabled_;
	}

	void Framebuffer::ResetErrorEstimate()
	{
		// The first sample of each half overwrites its sums (FramebufferView::AddSample).
		error_samples_ = 0;
		pixel_error_   = std::numeric_limits<float>::infinity();
	}

	uint32_t Framebuffer::ErrorSampleCount() const
	{
		return error_samples_;
	}

	void Framebuffer::SetTargetPixelError(float target_pixel_error)
	{
		if (!std::isfinite(target_pixel_error) || target_pixel_error < 0.0f)
			throw std::invalid_argument("Target pixel error must be finite and nonnegative");
		target_pixel_error_ = target_pixel_error;
	}

	float Framebuffer::TargetPixelError() const
	{
		return target_pixel_error_;
	}

	float Framebuffer::AveragePixelError() const
	{
		return pixel_error_;
	}

	bool Framebuffer::Converged() const
	{
		return error_samples_ >= kMinErrorSamples && pixel_error_ < target_pixel_error_;
	}

} // namespace happyray
