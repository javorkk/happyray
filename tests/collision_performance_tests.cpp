// Collision detection benchmark: frames 105 to 106 of the Fairy Forest animation, interpolated
// from keyframes f100.obj and f110.obj (frame 105 = 0.5, frame 106 = 0.6 between them). Each
// triangle is replaced by the box bounding its motion over the interval (MakeSweptTriangleBoxes),
// every grid is built over these boxes as PrimitiveType::kBox primitives (with densities tuned
// per backend for this benchmark, see kDensities), and the renderer's
// collision detection (CollisionDetector::ComputeCollisions) tests boxes from different objects
// sharing a leaf cell for overlap. Reports, per grid, build, collision, and total time, the pair
// tests executed (candidate pairs), and how many of them overlapped (colliding pairs), next to
// the runtime of a sweep-and-prune oracle (baseline) running on the same Thrust backend, with the
// same persistent temporary storage as the detector. Per-box
// collision results must match the oracle; timings are informational.
// With --search-densities, also searches each grid's densities for the lowest total time.
// Pure Thrust, so it compiles for the CUDA (nvcc) or TBB (g++) backend like the other tests.
#if HAPPYRAY_THRUST_BACKEND_IS_TBB
#define THRUST_HOST_SYSTEM THRUST_HOST_SYSTEM_CPP
#define THRUST_DEVICE_SYSTEM THRUST_DEVICE_SYSTEM_TBB
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <map>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <thrust/binary_search.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/for_each.h>
#include <thrust/gather.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/scan.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/transform.h>

#include "../src/accel/irregular_grid.h"
#include "../src/accel/two_level_grid.h"
#include "../src/accel/uniform_grid.h"
#include "../src/app/obj_loader.h"
#include "../src/app/scene.h"
#include "../src/core/thrust_scratch.h"
#include "../src/render/collision_detector.h"

namespace
{

	using namespace happyray;

	constexpr float kStartFrame	  = 105.0f;
	constexpr float kEndFrame	  = 106.0f;
	constexpr float kKeyframeA	  = 100.0f;
	constexpr float kKeyframeB	  = 110.0f;
	constexpr int	kMeasuredRuns = 5;
#if HAPPYRAY_THRUST_BACKEND_IS_TBB
	constexpr const char *kBackendName = "TBB";
#else
	constexpr const char *kBackendName = "CUDA";
#endif

	// Grid densities minimizing this benchmark's total (build + collision) time per backend, found
	// with --search-densities (half-octave lattice values, 2^(k/2)). The renderer keeps its own
	// defaults, which are tuned for ray tracing.
	struct BenchmarkDensities
	{
		float uniform;
		float two_level_top, two_level_leaf;
		float irregular_top, irregular_leaf;
	};
#if HAPPYRAY_THRUST_BACKEND_IS_TBB
	constexpr BenchmarkDensities kDensities{1.4142135f, 0.044194173f, 0.044194173f, 0.022097087f, 0.03125f};
#else
	constexpr BenchmarkDensities kDensities{16.0f, 0.0625f, 0.25f, 0.125f, 0.088388347f};
#endif

	template <typename T> const T *Raw(const thrust::device_vector<T> &values)
	{
		return thrust::raw_pointer_cast(values.data());
	}

	struct DeviceBoxScene
	{
		thrust::device_vector<BBox>		boxes;
		thrust::device_vector<uint32_t> materials, object_ids;

		explicit DeviceBoxScene(const FlatScene &scene)
			: boxes(scene.boxes), materials(scene.material_ids), object_ids(scene.object_ids)
		{
		}

		SceneView View() const
		{
			SceneView view;
			view.boxes			 = Raw(boxes);
			view.material_ids	 = Raw(materials);
			view.object_ids		 = Raw(object_ids);
			view.primitive_count = static_cast<uint32_t>(boxes.size());
			view.primitive_type	 = PrimitiveType::kBox;
			return view;
		}
	};

	// Reading one element back makes the CUDA backend finish all pending work (see
	// accel_performance_tests.cpp); a no-op wait on TBB.
	template <typename T> void Synchronize(const T *device_data)
	{
		static_cast<void>(static_cast<T>(*thrust::device_pointer_cast(device_data)));
	}

	void SyncGrid(const UniformGrid &grid)
	{
		Synchronize(grid.View().cell_ranges);
	}
	void SyncGrid(const TwoLevelGrid &grid)
	{
		Synchronize(grid.View().top_level_cells);
	}
	void SyncGrid(const IrregularGrid &grid)
	{
		Synchronize(grid.View().top_level.cell_ranges);
	}

	double ElapsedMs(std::chrono::steady_clock::time_point start, std::chrono::steady_clock::time_point end)
	{
		return std::chrono::duration<double, std::milli>(end - start).count();
	}

	struct Measurement
	{
		std::string			  name;
		double				  build_ms			   = 0.0;
		double				  collision_ms		   = 0.0;
		double				  candidate_pairs	   = 0.0;
		double				  colliding_pairs	   = 0.0;
		uint32_t			  colliding_primitives = 0;
		std::vector<uint32_t> collision_ids; // From the last run.
	};

	// One untimed warmup, then the mean of `runs` (build + collision detection) runs.
	template <typename MakeGrid>
	Measurement Measure(const std::string &name,
						MakeGrid		   make_grid,
						const SceneView	  &scene,
						uint32_t		   object_count,
						int				   runs = kMeasuredRuns)
	{
		CollisionDetector				detector;
		thrust::device_vector<uint32_t> collision_ids(scene.primitive_count);
		uint32_t					   *ids = thrust::raw_pointer_cast(collision_ids.data());
		{
			const auto warmup_grid = make_grid();
			SyncGrid(*warmup_grid);
			detector.ComputeCollisions(scene, object_count, *warmup_grid, ids);
		}
		Measurement result;
		result.name = name;
		for (int run = 0; run < runs; ++run)
		{
			const auto build_start = std::chrono::steady_clock::now();
			const auto grid		   = make_grid();
			SyncGrid(*grid);
			const auto build_end = std::chrono::steady_clock::now();
			// The returned counts are read back to the host, which synchronizes the backend.
			const CollisionStats stats	  = detector.ComputeCollisions(scene, object_count, *grid, ids);
			const auto			 test_end = std::chrono::steady_clock::now();
			result.build_ms += ElapsedMs(build_start, build_end);
			result.collision_ms += ElapsedMs(build_end, test_end);
			result.candidate_pairs += static_cast<double>(stats.candidate_pair_count);
			result.colliding_pairs += static_cast<double>(stats.colliding_pair_count);
			result.colliding_primitives = stats.colliding_primitive_count;
		}
		result.build_ms /= runs;
		result.collision_ms /= runs;
		result.candidate_pairs /= runs;
		result.colliding_pairs /= runs;
		const thrust::host_vector<uint32_t> host_ids = collision_ids;
		result.collision_ids.assign(host_ids.begin(), host_ids.end());
		return result;
	}

	// Keeps the smallest object id (see collision_detector.cpp).
	HAPPYRAY_HD inline void AtomicMinId(uint32_t *address, uint32_t value)
	{
#ifdef __CUDA_ARCH__
		atomicMin(address, value);
#else
		uint32_t current = __atomic_load_n(address, __ATOMIC_RELAXED);
		while (value < current &&
			   !__atomic_compare_exchange_n(address, &current, value, true, __ATOMIC_RELAXED, __ATOMIC_RELAXED))
		{
		}
#endif
	}

	struct MinX
	{
		HAPPYRAY_HD float operator()(const BBox &box) const
		{
			return box.min_bound.x;
		}
	};

	struct MaxX
	{
		HAPPYRAY_HD float operator()(const BBox &box) const
		{
			return box.max_bound.x;
		}
	};

	// Small chunks balance GPU warps; CPU threads amortize each chunk's binary search over more tests.
#if HAPPYRAY_THRUST_BACKEND_IS_TBB
	constexpr uint32_t kSweepChunk = 1024;
#else
	constexpr uint32_t kSweepChunk = 64;
#endif

	// Number of kSweepChunk-sized pieces of sorted box i's sweep [i + 1, sweep_end).
	struct SweepChunkCount
	{
		HAPPYRAY_HD uint32_t operator()(uint32_t i, uint32_t sweep_end) const
		{
			const uint32_t length = sweep_end > i + 1 ? sweep_end - (i + 1) : 0u;
			return (length + kSweepChunk - 1) / kSweepChunk;
		}
	};

	// Tests sorted box i against one chunk of the later boxes whose x interval starts inside its
	// own. Fixed-size chunks balance the very uneven sweep lengths across threads.
	struct SweepAndPrune
	{
		const BBox	   *sorted_boxes;
		const uint32_t *sorted_object_ids;
		const uint32_t *order;		   // Sorted position -> box index.
		const uint32_t *sweep_ends;	   // Per sorted box, the first box past its x interval.
		const uint32_t *chunk_offsets; // Exclusive scan of the per-box chunk counts.
		uint32_t		count;
		uint32_t	   *expected;

		HAPPYRAY_HD void operator()(uint32_t work_item) const
		{
			// Last sorted box whose first chunk is <= work_item.
			uint32_t lo = 0, hi = count;
			while (hi - lo > 1)
			{
				const uint32_t mid = lo + (hi - lo) / 2;
				if (chunk_offsets[mid] <= work_item)
					lo = mid;
				else
					hi = mid;
			}
			const uint32_t i		= lo;
			const uint32_t begin	= i + 1 + (work_item - chunk_offsets[i]) * kSweepChunk;
			const uint32_t end		= begin + kSweepChunk < sweep_ends[i] ? begin + kSweepChunk : sweep_ends[i];
			const BBox	   a		= sorted_boxes[i];
			const uint32_t object_a = sorted_object_ids[i];
			uint32_t	   min_a	= kNoCollision;
			for (uint32_t j = begin; j < end; ++j)
			{
				const uint32_t object_b = sorted_object_ids[j];
				if (object_a != object_b && Overlaps(a, sorted_boxes[j]))
				{
					min_a = object_b < min_a ? object_b : min_a;
					AtomicMinId(expected + order[j], object_a);
				}
			}
			if (min_a != kNoCollision)
				AtomicMinId(expected + order[i], min_a);
		}
	};

	// Sweep and prune along x on the Thrust backend (CUDA or TBB): per box, the smallest id of
	// another object whose box overlaps it. Sorts the boxes by min x, finds where each box's sweep
	// ends with a binary search, then tests all sweeps in parallel, split into fixed-size chunks.
	// Like CollisionDetector, keeps its buffers and Thrust's scratch memory between runs.
	class CollisionOracle
	{
	  public:
		// Returns the per-box results (scene.primitive_count entries), valid until the next run.
		thrust::device_ptr<const uint32_t> Run(const SceneView &scene)
		{
			const uint32_t count  = scene.primitive_count;
			const auto	   boxes  = thrust::device_pointer_cast(scene.boxes);
			const auto	   first  = thrust::counting_iterator<uint32_t>(0);
			const auto	   policy = thrust::device(allocator_);
			const auto	   min_x  = thrust::device_pointer_cast(Reserve(&min_x_, count));
			const auto	   order  = thrust::device_pointer_cast(Reserve(&order_, count));
			thrust::transform(policy, boxes, boxes + count, min_x, MinX());
			thrust::sequence(policy, order, order + count);
			thrust::sort_by_key(policy, min_x, min_x + count, order);

			const auto sorted_boxes		 = thrust::device_pointer_cast(Reserve(&sorted_boxes_, count));
			const auto sorted_object_ids = thrust::device_pointer_cast(Reserve(&sorted_object_ids_, count));
			thrust::gather(policy, order, order + count, boxes, sorted_boxes);
			thrust::gather(policy,
						   order,
						   order + count,
						   thrust::device_pointer_cast(scene.object_ids),
						   sorted_object_ids);

			// Box i's sweep ends at the first box starting past its max x.
			const auto max_x = thrust::device_pointer_cast(Reserve(&max_x_, count));
			thrust::transform(policy, sorted_boxes, sorted_boxes + count, max_x, MaxX());
			const auto sweep_ends = thrust::device_pointer_cast(Reserve(&sweep_ends_, count));
			thrust::upper_bound(policy, min_x, min_x + count, max_x, max_x + count, sweep_ends);
			const auto chunk_counts	 = thrust::device_pointer_cast(Reserve(&chunk_counts_, count));
			const auto chunk_offsets = thrust::device_pointer_cast(Reserve(&chunk_offsets_, count));
			thrust::transform(policy, first, first + count, sweep_ends, chunk_counts, SweepChunkCount());
			thrust::exclusive_scan(policy, chunk_counts, chunk_counts + count, chunk_offsets);
			const uint32_t chunk_total =
				count > 0 ? static_cast<uint32_t>(chunk_offsets[count - 1]) + chunk_counts[count - 1] : 0u;

			const auto expected = thrust::device_pointer_cast(Reserve(&expected_, count));
			thrust::fill(policy, expected, expected + count, kNoCollision);
			thrust::for_each(policy,
							 first,
							 first + chunk_total,
							 SweepAndPrune{thrust::raw_pointer_cast(sorted_boxes),
										   thrust::raw_pointer_cast(sorted_object_ids),
										   thrust::raw_pointer_cast(order),
										   thrust::raw_pointer_cast(sweep_ends),
										   thrust::raw_pointer_cast(chunk_offsets),
										   count,
										   thrust::raw_pointer_cast(expected)});
			return expected;
		}

	  private:
		CachingAllocator				allocator_;
		thrust::device_vector<float>	min_x_, max_x_;
		thrust::device_vector<BBox>		sorted_boxes_;
		thrust::device_vector<uint32_t> order_, sorted_object_ids_, sweep_ends_, chunk_counts_, chunk_offsets_,
			expected_;
	};

	FlatScene LoadKeyframe(const std::string &path)
	{
		Scene scene;
		if (!LoadObj(path, &scene))
			throw std::runtime_error("Cannot load " + path);
		return FlattenScene(scene);
	}

	// Coarse-to-fine search for the densities minimizing total (build + collision) time. Densities
	// lie on a half-octave lattice, 2^(k/2) for integer k: a 2x-spaced coarse sweep, a hill climb
	// over the sqrt(2)-spaced neighbors of the best point until it stops moving, then the best few
	// and the benchmark's current densities re-measured with more runs. `make_grid(top, leaf)` builds a grid;
	// single-level grids ignore `top`. Every configuration must match the oracle.
	template <typename MakeGrid>
	void SearchDensities(const char					 *name,
						 bool						  nested,
						 float						  current_top,
						 float						  current_leaf,
						 MakeGrid					  make_grid,
						 const SceneView			 &view,
						 uint32_t					  object_count,
						 const std::vector<uint32_t> &expected)
	{
		constexpr int kSearchRuns = 5, kFinalRuns = 15, kFinalists = 5, kMaxConfigurations = 200;
		using Exponents							   = std::pair<int, int>; // (top, leaf) half-octaves.
		auto density = [](int k) { return static_cast<float>(std::exp2(0.5 * k)); };
		auto total	 = [](const Measurement &m) { return m.build_ms + m.collision_ms; };
		auto measure = [&](float top, float leaf, int runs)
		{
			Measurement m = Measure(name, [&] { return make_grid(top, leaf); }, view, object_count, runs);
			if (m.collision_ids != expected)
				throw std::runtime_error(std::string(name) + ": collision results disagree with the oracle at top=" +
										 std::to_string(top) + ", leaf=" + std::to_string(leaf));
			m.collision_ids.clear();
			return m;
		};
		std::map<Exponents, double> searched; // Mean total ms.
		auto						evaluate = [&](int top, int leaf)
		{
			if (!nested)
				top = 0;
			if (!searched.count({top, leaf}))
				searched[{top, leaf}] = total(measure(density(top), density(leaf), kSearchRuns));
		};
		auto ranked = [&]
		{
			std::vector<std::pair<Exponents, double>> order(searched.begin(), searched.end());
			std::sort(order.begin(), order.end(), [](const auto &a, const auto &b) { return a.second < b.second; });
			return order;
		};

		// Coarse: tops 1/256..1 and leaves 1/16..16 (nested), or densities 1/8..64.
		for (int top = nested ? -16 : 0; top <= 0; top += 2)
			for (int leaf = nested ? -8 : -6; leaf <= (nested ? 8 : 12); leaf += 2)
				evaluate(top, leaf);
		for (Exponents best = ranked().front().first, previous = {1 << 20, 1 << 20};
			 best != previous && searched.size() < kMaxConfigurations;
			 previous = best, best = ranked().front().first)
		{
			for (int dt = -1; dt <= 1; ++dt)
				for (int dl = -1; dl <= 1; ++dl)
					evaluate(best.first + dt, best.second + dl);
		}

		struct Finalist
		{
			float		top, leaf;
			bool		is_current;
			Measurement measurement;
		};
		std::vector<Finalist> finalists;
		for (const auto &[exponents, ms] : ranked())
		{
			if (finalists.size() == kFinalists)
				break;
			const float top = nested ? density(exponents.first) : 0.0f, leaf = density(exponents.second);
			finalists.push_back({top, leaf, false, measure(top, leaf, kFinalRuns)});
		}
		auto same = [](float a, float b) { return std::fabs(a - b) <= 1e-4f * std::fmax(a, b); };
		bool current_measured = false;
		for (Finalist &finalist : finalists)
		{
			finalist.is_current = (!nested || same(finalist.top, current_top)) && same(finalist.leaf, current_leaf);
			current_measured	= current_measured || finalist.is_current;
		}
		if (!current_measured)
			finalists.push_back({current_top, current_leaf, true, measure(current_top, current_leaf, kFinalRuns)});
		std::sort(finalists.begin(),
				  finalists.end(),
				  [&](const Finalist &a, const Finalist &b) { return total(a.measurement) < total(b.measurement); });

		std::printf("\n%s: %zu configurations searched (%d runs each); finalists re-measured with %d runs\n",
					name,
					searched.size(),
					kSearchRuns,
					kFinalRuns);
		std::printf("%10s %10s %12s %12s %14s %12s %12s\n",
					"Top",
					"Leaf",
					"Total ms",
					"Build ms",
					"Collision ms",
					"Candidates",
					"Colliding");
		for (const Finalist &finalist : finalists)
		{
			char top_text[16] = "-";
			if (nested)
				std::snprintf(top_text, sizeof(top_text), "%.4g", finalist.top);
			const Measurement &m = finalist.measurement;
			std::printf("%10s %10.4g %12.1f %12.1f %14.1f %12.0f %12.0f%s\n",
						top_text,
						finalist.leaf,
						total(m),
						m.build_ms,
						m.collision_ms,
						m.candidate_pairs,
						m.colliding_pairs,
						finalist.is_current ? "  (benchmark)" : "");
		}
	}

	void Run(const std::string &scene_dir, bool search_densities)
	{
		const float		span  = kKeyframeB - kKeyframeA;
		const FlatScene boxes = MakeSweptTriangleBoxes(LoadKeyframe(scene_dir + "/f100.obj"),
													   LoadKeyframe(scene_dir + "/f110.obj"),
													   (kStartFrame - kKeyframeA) / span,
													   (kEndFrame - kKeyframeA) / span);
		if (boxes.object_ids.size() != boxes.PrimitiveCount() || boxes.object_ids.empty())
			throw std::runtime_error("Scene needs per-triangle object ids");
		const uint32_t object_count = *std::max_element(boxes.object_ids.begin(), boxes.object_ids.end()) + 1;

		const DeviceBoxScene device(boxes);
		const SceneView		 view = device.View();
		// For boxes, exact cell overlap equals AABB overlap; kAabb skips the redundant tests.
		constexpr auto kAabb = CellOverlapMode::kAabb;
		auto		   label = [](const char *name, float top, float leaf)
		{
			char text[96];
			if (top > 0)
				std::snprintf(text, sizeof(text), "%s (top=%.3g, leaf=%.3g)", name, top, leaf);
			else
				std::snprintf(text, sizeof(text), "%s (density=%.3g)", name, leaf);
			return std::string(text);
		};

		std::vector<Measurement> results;
		results.push_back(Measure(
			label("uniform", 0, kDensities.uniform),
			[&] { return std::make_unique<UniformGrid>(view, kDensities.uniform, kAabb); },
			view,
			object_count));
		results.push_back(Measure(
			label("two-level", kDensities.two_level_top, kDensities.two_level_leaf),
			[&]
			{
				return std::make_unique<TwoLevelGrid>(view,
													  kDensities.two_level_top,
													  kDensities.two_level_leaf,
													  16,
													  kAabb);
			},
			view,
			object_count));
		results.push_back(Measure(
			label("irregular", kDensities.irregular_top, kDensities.irregular_leaf),
			[&]
			{
				return std::make_unique<IrregularGrid>(view,
													   kDensities.irregular_top,
													   kDensities.irregular_leaf,
													   16,
													   0.7f,
													   2,
													   kAabb);
			},
			view,
			object_count));

		// Baseline: the oracle, timed like the grids (one untimed warmup that also allocates its
		// storage, then the mean of runs).
		CollisionOracle oracle;
		oracle.Run(view);
		double							   baseline_ms = 0.0;
		thrust::device_ptr<const uint32_t> device_expected;
		for (int run = 0; run < kMeasuredRuns; ++run)
		{
			const auto start = std::chrono::steady_clock::now();
			device_expected	 = oracle.Run(view);
			Synchronize(thrust::raw_pointer_cast(device_expected));
			baseline_ms += ElapsedMs(start, std::chrono::steady_clock::now());
		}
		baseline_ms /= kMeasuredRuns;
		const thrust::host_vector<uint32_t> host_expected(device_expected, device_expected + view.primitive_count);
		const std::vector<uint32_t>			expected(host_expected.begin(), host_expected.end());

		std::printf("\nScene: %s/f100.obj + f110.obj, frames %g to %g (%u swept triangle boxes, %u objects)\n",
					scene_dir.c_str(),
					kStartFrame,
					kEndFrame,
					view.primitive_count,
					object_count);
		std::printf("Grids over box primitives (cell overlap: aabb); mean of %d runs after one warmup; baseline: "
					"sweep and prune on the %s backend (the oracle)\n",
					kMeasuredRuns,
					kBackendName);
		std::printf("%-38s %12s %12s %12s %14s %12s %12s\n",
					"Grid (tuned densities)",
					"Baseline ms",
					"Total ms",
					"Build ms",
					"Collision ms",
					"Candidates",
					"Colliding");
		for (const Measurement &result : results)
			std::printf("%-38s %12.1f %12.1f %12.1f %14.1f %12.0f %12.0f\n",
						result.name.c_str(),
						baseline_ms,
						result.build_ms + result.collision_ms,
						result.build_ms,
						result.collision_ms,
						result.candidate_pairs,
						result.colliding_pairs);

		const auto expected_colliding = static_cast<uint32_t>(
			std::count_if(expected.begin(), expected.end(), [](uint32_t id) { return id != kNoCollision; }));
		bool correct = expected_colliding > 0;
		for (const Measurement &result : results)
		{
			if (result.collision_ids != expected || result.colliding_primitives != expected_colliding ||
				result.colliding_pairs < 1.0 || result.candidate_pairs < result.colliding_pairs)
			{
				std::printf("%s: %u colliding boxes, expected %u (sweep-and-prune oracle)%s\n",
							result.name.c_str(),
							result.colliding_primitives,
							expected_colliding,
							result.collision_ids == expected ? "" : "; per-box collision ids differ");
				correct = false;
			}
		}
		if (!correct)
			throw std::runtime_error("Collision results disagree with the oracle");

		if (!search_densities)
			return;
		std::printf("\nDensity search (minimizing total ms)\n");
		SearchDensities(
			"uniform",
			false,
			0.0f,
			kDensities.uniform,
			[&](float, float leaf) { return std::make_unique<UniformGrid>(view, leaf, kAabb); },
			view,
			object_count,
			expected);
		SearchDensities(
			"two-level",
			true,
			kDensities.two_level_top,
			kDensities.two_level_leaf,
			[&](float top, float leaf) { return std::make_unique<TwoLevelGrid>(view, top, leaf, 16, kAabb); },
			view,
			object_count,
			expected);
		SearchDensities(
			"irregular",
			true,
			kDensities.irregular_top,
			kDensities.irregular_leaf,
			[&](float top, float leaf)
			{ return std::make_unique<IrregularGrid>(view, top, leaf, 16, 0.7f, 2, kAabb); },
			view,
			object_count,
			expected);
	}

} // namespace

int main(int argc, char **argv)
{
	try
	{
		const bool search = argc == 3 && std::string(argv[2]) == "--search-densities";
		if (argc != 2 && !search)
			throw std::runtime_error(
				"Usage: happyray_collision_performance_tests <fairy scene directory> [--search-densities]");
		Run(argv[1], search);
		return EXIT_SUCCESS;
	}
	catch (const std::exception &error)
	{
		std::fprintf(stderr, "Collision performance test failed: %s\n", error.what());
		return EXIT_FAILURE;
	}
}
