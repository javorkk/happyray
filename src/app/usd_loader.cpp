#include "usd_loader.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <filesystem>

#ifdef HAPPYRAY_WITH_USD
#include <atomic>
#include <limits>
#include <mutex>
#include <set>
#include <stdexcept>
#include <thread>

#include <pxr/base/gf/interval.h>
#include <pxr/base/gf/matrix4d.h>
#include <pxr/base/vt/array.h>
#include <pxr/usd/usd/prim.h>
#include <pxr/usd/usd/primFlags.h>
#include <pxr/usd/usd/primRange.h>
#include <pxr/usd/usd/stage.h>
#include <pxr/usd/usd/timeCode.h>
#include <pxr/usd/usdGeom/imageable.h>
#include <pxr/usd/usdGeom/mesh.h>
#include <pxr/usd/usdGeom/metrics.h>
#include <pxr/usd/usdGeom/tokens.h>
#include <pxr/usd/usdGeom/xformCache.h>
#include <pxr/usd/usdGeom/xformable.h>
#endif

namespace happyray
{

	bool IsUsdPath(const std::string &path)
	{
		std::string extension = std::filesystem::path(path).extension().string();
		std::transform(extension.begin(), extension.end(), extension.begin(), [](unsigned char c) {
			return static_cast<char>(std::tolower(c));
		});
		return extension == ".usd" || extension == ".usda" || extension == ".usdc" || extension == ".usdz";
	}

#ifndef HAPPYRAY_WITH_USD

	bool UsdSupported()
	{
		return false;
	}

	bool LoadUsdFrames(const std::string &path, const UsdLoadOptions &, std::vector<FlatScene> *frames, Scene *)
	{
		frames->clear();
		std::fprintf(stderr,
					 "Cannot load '%s': this build has no OpenUSD support (configure with "
					 "-DHAPPYRAY_ENABLE_USD=ON)\n",
					 path.c_str());
		return false;
	}

#else

	namespace
	{
		PXR_NAMESPACE_USING_DIRECTIVE

		constexpr unsigned kMaxLoadThreads = 16;

		struct MeshSource
		{
			UsdGeomMesh mesh;
			std::string name;
		};

		bool IsLoadablePurpose(const UsdPrim &prim)
		{
			const TfToken purpose = UsdGeomImageable(prim).ComputePurpose();
			return purpose == UsdGeomTokens->default_ || purpose == UsdGeomTokens->render;
		}

		Vec3f ToVec3f(const GfVec3f &v)
		{
			return Vec3f(v[0], v[1], v[2]);
		}

		// Appends a mesh's world-space triangles and smooth vertex normals at `time` to `scene`.
		// Only `scene->materials`' index `material_id` is referenced.
		void AppendMesh(const MeshSource &source,
						UsdGeomXformCache &cache,
						UsdTimeCode		   time,
						bool			   z_up,
						uint32_t		   material_id,
						Scene			  *scene)
		{
			const UsdGeomMesh &mesh = source.mesh;
			VtVec3fArray	   points;
			VtIntArray		   face_counts;
			VtIntArray		   face_indices;
			VtIntArray		   holes;
			TfToken			   orientation = UsdGeomTokens->rightHanded;
			mesh.GetPointsAttr().Get(&points, time);
			mesh.GetFaceVertexCountsAttr().Get(&face_counts, time);
			mesh.GetFaceVertexIndicesAttr().Get(&face_indices, time);
			mesh.GetHoleIndicesAttr().Get(&holes, time);
			mesh.GetOrientationAttr().Get(&orientation, time);

			const GfMatrix4d local_to_world = cache.GetLocalToWorldTransform(mesh.GetPrim());
			const bool		 flip_winding =
				(orientation == UsdGeomTokens->leftHanded) != (local_to_world.GetDeterminant3() < 0.0);

			Mesh out;
			out.name = source.name;
			out.positions.reserve(points.size());
			for (const GfVec3f &point : points)
			{
				const GfVec3f world = GfVec3f(local_to_world.Transform(point));
				out.positions.push_back(z_up ? Vec3f(world[0], world[2], -world[1]) : ToVec3f(world));
			}

			const std::set<int> hole_faces(holes.begin(), holes.end());
			size_t				cursor = 0;
			for (size_t face = 0; face < face_counts.size(); ++face)
			{
				const size_t count = face_counts[face] > 0 ? static_cast<size_t>(face_counts[face]) : 0;
				if (cursor + count > face_indices.size())
				{
					throw std::runtime_error("face vertex counts exceed face vertex indices in " + source.name);
				}
				const int *face_vertices = face_indices.data() + cursor;
				cursor += count;
				if (count < 3 || hole_faces.count(static_cast<int>(face)))
				{
					continue;
				}
				for (size_t corner = 1; corner + 1 < count; ++corner)
				{
					const int triangle[3] = {face_vertices[0], face_vertices[corner], face_vertices[corner + 1]};
					for (const int index : triangle)
					{
						if (index < 0 || static_cast<size_t>(index) >= points.size())
						{
							throw std::runtime_error("point index out of range in " + source.name);
						}
					}
					out.indices.push_back(static_cast<uint32_t>(triangle[0]));
					out.indices.push_back(static_cast<uint32_t>(triangle[flip_winding ? 2 : 1]));
					out.indices.push_back(static_cast<uint32_t>(triangle[flip_winding ? 1 : 2]));
					out.material_ids.push_back(material_id);
				}
			}

			// Unnormalized face normals are proportional to triangle area, which area-weights
			// each triangle's contribution to its vertices.
			std::vector<Vec3f> accumulated(out.positions.size(), Vec3f(0.0f, 0.0f, 0.0f));
			for (size_t i = 0; i < out.indices.size(); i += 3)
			{
				const uint32_t i0			= out.indices[i];
				const uint32_t i1			= out.indices[i + 1];
				const uint32_t i2			= out.indices[i + 2];
				const Vec3f	   face_normal	= Cross(out.positions[i1] - out.positions[i0], out.positions[i2] - out.positions[i0]);
				accumulated[i0] += face_normal;
				accumulated[i1] += face_normal;
				accumulated[i2] += face_normal;
			}
			out.attributes.resize(out.positions.size());
			for (size_t i = 0; i < accumulated.size(); ++i)
			{
				const Vec3f &n = accumulated[i];
				out.attributes[i].normal =
					Dot(n, n) > 0.0f ? Normalize(n) : Vec3f(0.0f, 1.0f, 0.0f);
				out.attributes[i].uv = Vec2f(0.0f, 0.0f);
			}
			scene->meshes.push_back(std::move(out));
		}

		// Builds the whole scene at `time`. When `expected_point_counts` is non-null, each mesh's
		// point count must match it; otherwise the counts found are stored in `found_point_counts`.
		void BuildScene(const std::vector<MeshSource> &sources,
						const std::vector<Material>	  &materials,
						UsdGeomXformCache			  &cache,
						UsdTimeCode					   time,
						bool						   z_up,
						const std::vector<size_t>	  *expected_point_counts,
						Scene						  *scene)
		{
			cache.SetTime(time);
			scene->materials = materials;
			for (size_t i = 0; i < sources.size(); ++i)
			{
				AppendMesh(sources[i], cache, time, z_up, static_cast<uint32_t>(i), scene);
				if (expected_point_counts &&
					scene->meshes.back().positions.size() != (*expected_point_counts)[i])
				{
					throw std::runtime_error("point count of " + sources[i].name +
											 " differs from the first frame at time " +
											 std::to_string(time.GetValue()));
				}
			}
		}

		Vec3f AverageDisplayColor(const UsdGeomMesh &mesh, UsdTimeCode time)
		{
			VtVec3fArray colors;
			if (!mesh.GetDisplayColorPrimvar().ComputeFlattened(&colors, time) || colors.empty())
			{
				return Vec3f(0.8f, 0.8f, 0.8f);
			}
			double sum[3] = {0.0, 0.0, 0.0};
			for (const GfVec3f &color : colors)
			{
				for (int c = 0; c < 3; ++c)
				{
					sum[c] += color[c];
				}
			}
			const double n = static_cast<double>(colors.size());
			return Vec3f(static_cast<float>(sum[0] / n), static_cast<float>(sum[1] / n), static_cast<float>(sum[2] / n));
		}
	} // namespace

	bool UsdSupported()
	{
		return true;
	}

	bool LoadUsdFrames(const std::string		&path,
					   const UsdLoadOptions		&options,
					   std::vector<FlatScene>	*frames,
					   Scene					*first_scene)
	{
		PXR_NAMESPACE_USING_DIRECTIVE
		frames->clear();
		try
		{
			const UsdStageRefPtr stage = UsdStage::Open(path);
			if (!stage)
			{
				std::fprintf(stderr, "Failed to open USD file '%s'\n", path.c_str());
				return false;
			}
			const bool z_up = UsdGeomGetStageUpAxis(stage) == UsdGeomTokens->z;

			const double lowest = std::numeric_limits<double>::lowest();
			const double highest = std::numeric_limits<double>::max();
			const GfInterval interval(options.start_time.value_or(lowest), options.end_time.value_or(highest));

			std::vector<MeshSource> sources;
			std::set<double>		sample_set;
			std::vector<double>		samples;
			for (const UsdPrim &prim : stage->Traverse(UsdTraverseInstanceProxies()))
			{
				const UsdGeomXformable xformable(prim);
				if (xformable && xformable.GetTimeSamplesInInterval(interval, &samples))
				{
					sample_set.insert(samples.begin(), samples.end());
				}
				const UsdGeomMesh mesh(prim);
				if (mesh && IsLoadablePurpose(prim))
				{
					if (mesh.GetPointsAttr().GetTimeSamplesInInterval(interval, &samples))
					{
						sample_set.insert(samples.begin(), samples.end());
					}
					sources.push_back({mesh, prim.GetPath().GetString()});
				}
			}

			std::vector<UsdTimeCode> times;
			const size_t			 stride = std::max<uint32_t>(options.stride, 1u);
			size_t					 sample_index = 0;
			for (const double sample : sample_set)
			{
				if (sample_index++ % stride == 0)
				{
					times.push_back(UsdTimeCode(sample));
				}
			}
			if (times.empty())
			{
				times.push_back(UsdTimeCode::Default());
			}

			// Visibility is evaluated once, at the first frame, to fix the mesh set.
			{
				std::vector<MeshSource> visible;
				for (MeshSource &source : sources)
				{
					if (UsdGeomImageable(source.mesh.GetPrim()).ComputeVisibility(times.front()) !=
						UsdGeomTokens->invisible)
					{
						visible.push_back(std::move(source));
					}
				}
				sources = std::move(visible);
			}
			if (sources.empty())
			{
				std::fprintf(stderr, "USD file '%s' contains no visible meshes\n", path.c_str());
				return false;
			}

			std::vector<Material> materials;
			for (const MeshSource &source : sources)
			{
				Material material;
				material.name			= source.name;
				material.diffuse_color = AverageDisplayColor(source.mesh, times.front());
				materials.push_back(std::move(material));
			}

			UsdGeomXformCache first_cache;
			Scene			  first;
			BuildScene(sources, materials, first_cache, times.front(), z_up, nullptr, &first);
			std::vector<size_t> point_counts;
			for (const Mesh &mesh : first.meshes)
			{
				point_counts.push_back(mesh.positions.size());
			}
			if (first.TriangleCount() == 0)
			{
				std::fprintf(stderr, "USD file '%s' has no triangles at its first frame\n", path.c_str());
				return false;
			}

			std::vector<FlatScene> loaded(times.size());
			loaded[0] = FlattenScene(first);

			std::atomic<size_t> next_frame{1};
			std::mutex			error_mutex;
			std::string			error;
			auto				worker = [&]() {
				   UsdGeomXformCache cache;
				   for (size_t frame = next_frame++; frame < times.size(); frame = next_frame++)
				   {
					   try
					   {
						   Scene scene;
						   BuildScene(sources, materials, cache, times[frame], z_up, &point_counts, &scene);
						   loaded[frame] = FlattenScene(scene);
					   }
					   catch (const std::exception &e)
					   {
						   std::lock_guard<std::mutex> lock(error_mutex);
						   if (error.empty())
						   {
							   error = e.what();
						   }
						   next_frame = times.size();
						   return;
					   }
				   }
			};
			const unsigned thread_count = std::max(
				1u, std::min({kMaxLoadThreads, std::thread::hardware_concurrency(), static_cast<unsigned>(times.size())}));
			std::vector<std::thread> threads;
			for (unsigned i = 1; i < thread_count; ++i)
			{
				threads.emplace_back(worker);
			}
			worker();
			for (std::thread &thread : threads)
			{
				thread.join();
			}
			if (!error.empty())
			{
				std::fprintf(stderr, "Failed to load USD file '%s': %s\n", path.c_str(), error.c_str());
				return false;
			}

			std::printf("Loaded USD '%s': %zu mesh(es), %u triangle(s), %zu frame(s)%s\n",
						path.c_str(),
						sources.size(),
						first.TriangleCount(),
						loaded.size(),
						z_up ? " (Z-up converted to Y-up)" : "");
			*first_scene = std::move(first);
			*frames		 = std::move(loaded);
			return true;
		}
		catch (const std::exception &e)
		{
			frames->clear();
			std::fprintf(stderr, "Failed to load USD file '%s': %s\n", path.c_str(), e.what());
			return false;
		}
	}

#endif // HAPPYRAY_WITH_USD

} // namespace happyray
