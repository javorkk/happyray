#include "usd_loader.h"

#include <algorithm>
#include <cctype>
#include <cmath>
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
#include <pxr/usd/usdGeom/basisCurves.h>
#include <pxr/usd/usdGeom/imageable.h>
#include <pxr/usd/usdGeom/mesh.h>
#include <pxr/usd/usdGeom/metrics.h>
#include <pxr/usd/usdGeom/points.h>
#include <pxr/usd/usdGeom/tokens.h>
#include <pxr/usd/usdGeom/xformCache.h>
#include <pxr/usd/usdGeom/xformable.h>
#endif

namespace happyray
{

	bool IsUsdPath(const std::string &path)
	{
		std::string extension = std::filesystem::path(path).extension().string();
		std::transform(extension.begin(),
					   extension.end(),
					   extension.begin(),
					   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
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

		struct PointSource
		{
			UsdGeomPoints points;
			std::string	  name;
		};

		struct CurveSource
		{
			UsdGeomBasisCurves curves;
			std::string		   name;
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

		Vec3f ToWorld(const GfVec3f &point, const GfMatrix4d &transform, bool z_up)
		{
			const GfVec3f world(transform.Transform(point));
			return z_up ? Vec3f(world[0], world[2], -world[1]) : ToVec3f(world);
		}

		float RadiusScale(const GfMatrix4d &transform)
		{
			float scale = 0.0f;
			for (const GfVec3d axis : {GfVec3d(1, 0, 0), GfVec3d(0, 1, 0), GfVec3d(0, 0, 1)})
			{
				const GfVec3d transformed = transform.TransformDir(axis);
				const float	  length =
					static_cast<float>(std::sqrt(transformed[0] * transformed[0] + transformed[1] * transformed[1] +
												 transformed[2] * transformed[2]));
				scale = std::max(scale, length);
			}
			return scale;
		}

		// Appends a mesh's world-space triangles and smooth vertex normals at `time` to `scene`.
		// Only `scene->materials`' index `material_id` is referenced.
		void AppendMesh(const MeshSource  &source,
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
				const uint32_t i0 = out.indices[i];
				const uint32_t i1 = out.indices[i + 1];
				const uint32_t i2 = out.indices[i + 2];
				const Vec3f	   face_normal =
					Cross(out.positions[i1] - out.positions[i0], out.positions[i2] - out.positions[i0]);
				accumulated[i0] += face_normal;
				accumulated[i1] += face_normal;
				accumulated[i2] += face_normal;
			}
			out.attributes.resize(out.positions.size());
			for (size_t i = 0; i < accumulated.size(); ++i)
			{
				const Vec3f &n			 = accumulated[i];
				out.attributes[i].normal = Dot(n, n) > 0.0f ? Normalize(n) : Vec3f(0.0f, 1.0f, 0.0f);
				out.attributes[i].uv	 = Vec2f(0.0f, 0.0f);
			}
			scene->meshes.push_back(std::move(out));
		}

		// Builds the whole scene at `time`. When `expected_point_counts` is non-null, each mesh's
		// point count must match it.
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
				if (expected_point_counts && scene->meshes.back().positions.size() != (*expected_point_counts)[i])
				{
					throw std::runtime_error("point count of " + sources[i].name +
											 " differs from the first frame at time " +
											 std::to_string(time.GetValue()));
				}
			}
		}

		template <typename UsdGeomType> Vec3f AverageDisplayColor(const UsdGeomType &geometry, UsdTimeCode time)
		{
			VtVec3fArray colors;
			if (!geometry.GetDisplayColorPrimvar().ComputeFlattened(&colors, time) || colors.empty())
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
			return Vec3f(static_cast<float>(sum[0] / n),
						 static_cast<float>(sum[1] / n),
						 static_cast<float>(sum[2] / n));
		}

		float WidthAt(const VtFloatArray &widths, size_t index)
		{
			if (widths.empty())
			{
				return 0.0f;
			}
			if (index >= widths.size())
			{
				throw std::runtime_error("width count does not match geometry topology");
			}
			if (widths[index] < 0.0f)
			{
				throw std::runtime_error("negative primitive width");
			}
			return widths[index];
		}

		void BuildPoints(const std::vector<PointSource> &sources,
						 const std::vector<Material>	&materials,
						 UsdGeomXformCache				&cache,
						 UsdTimeCode					 time,
						 bool							 z_up,
						 const std::vector<size_t>		*expected_counts,
						 std::vector<size_t>			*found_counts,
						 Scene							*scene)
		{
			cache.SetTime(time);
			scene->materials = materials;
			for (size_t source_index = 0; source_index < sources.size(); ++source_index)
			{
				const PointSource &source = sources[source_index];
				VtVec3fArray	   positions;
				VtFloatArray	   widths;
				source.points.GetPointsAttr().Get(&positions, time);
				source.points.GetWidthsAttr().Get(&widths, time);
				const TfToken interpolation = source.points.GetWidthsInterpolation();
				if (!widths.empty() && interpolation == UsdGeomTokens->vertex && widths.size() != positions.size())
				{
					throw std::runtime_error("point width count differs from point count in " + source.name);
				}
				if (!widths.empty() && interpolation != UsdGeomTokens->vertex &&
					interpolation != UsdGeomTokens->constant)
				{
					throw std::runtime_error("unsupported point width interpolation in " + source.name);
				}
				const GfMatrix4d transform	  = cache.GetLocalToWorldTransform(source.points.GetPrim());
				const float		 radius_scale = RadiusScale(transform);
				for (size_t i = 0; i < positions.size(); ++i)
				{
					const size_t width_index = interpolation == UsdGeomTokens->vertex ? i : 0;
					scene->points.emplace_back(ToWorld(positions[i], transform, z_up),
											   WidthAt(widths, width_index) * 0.5f * radius_scale);
					scene->primitive_material_ids.push_back(static_cast<uint32_t>(source_index));
				}
				if (expected_counts && positions.size() != (*expected_counts)[source_index])
				{
					throw std::runtime_error("point count of " + source.name +
											 " differs from the first frame at time " +
											 std::to_string(time.GetValue()));
				}
				if (found_counts)
				{
					found_counts->push_back(positions.size());
				}
			}
		}

		GfVec3f CurveControl(const VtVec3fArray &points, size_t start, size_t count, int index, bool periodic)
		{
			if (periodic)
			{
				const int wrapped =
					(index % static_cast<int>(count) + static_cast<int>(count)) % static_cast<int>(count);
				return points[start + static_cast<size_t>(wrapped)];
			}
			if (index < 0)
			{
				return 2.0f * points[start] - points[start + 1];
			}
			if (static_cast<size_t>(index) >= count)
			{
				return 2.0f * points[start + count - 1] - points[start + count - 2];
			}
			return points[start + static_cast<size_t>(index)];
		}

		void BuildCurves(const std::vector<CurveSource> &sources,
						 const std::vector<Material>	&materials,
						 UsdGeomXformCache				&cache,
						 UsdTimeCode					 time,
						 bool							 z_up,
						 const std::vector<size_t>		*expected_counts,
						 std::vector<size_t>			*found_counts,
						 Scene							*scene)
		{
			cache.SetTime(time);
			scene->materials = materials;
			for (size_t source_index = 0; source_index < sources.size(); ++source_index)
			{
				const CurveSource &source = sources[source_index];
				VtVec3fArray	   points;
				VtIntArray		   counts;
				VtFloatArray	   widths;
				TfToken			   type	 = UsdGeomTokens->cubic;
				TfToken			   basis = UsdGeomTokens->bezier;
				TfToken			   wrap	 = UsdGeomTokens->nonperiodic;
				source.curves.GetPointsAttr().Get(&points, time);
				source.curves.GetCurveVertexCountsAttr().Get(&counts, time);
				source.curves.GetWidthsAttr().Get(&widths, time);
				source.curves.GetTypeAttr().Get(&type, time);
				source.curves.GetBasisAttr().Get(&basis, time);
				source.curves.GetWrapAttr().Get(&wrap, time);
				const TfToken width_interpolation = source.curves.GetWidthsInterpolation();
				const bool	  periodic			  = wrap == UsdGeomTokens->periodic;
				const bool	  pinned			  = wrap == UsdGeomTokens->pinned;
				if (type != UsdGeomTokens->linear && type != UsdGeomTokens->cubic)
				{
					throw std::runtime_error("unsupported curve type in " + source.name);
				}
				if (basis != UsdGeomTokens->bezier && basis != UsdGeomTokens->bspline &&
					basis != UsdGeomTokens->catmullRom)
				{
					throw std::runtime_error("unsupported curve basis in " + source.name);
				}
				if (pinned && (type != UsdGeomTokens->cubic || basis == UsdGeomTokens->bezier))
				{
					throw std::runtime_error("pinned wrap requires cubic bspline or catmullRom curves in " +
											 source.name);
				}
				const GfMatrix4d transform		 = cache.GetLocalToWorldTransform(source.curves.GetPrim());
				const float		 radius_scale	 = RadiusScale(transform);
				size_t			 point_start	 = 0;
				size_t			 varying_start	 = 0;
				size_t			 generated_count = 0;
				const size_t	 width_control_step =
					type == UsdGeomTokens->linear || basis != UsdGeomTokens->bezier ? 1u : 3u;
				for (size_t curve_index = 0; curve_index < counts.size(); ++curve_index)
				{
					if (counts[curve_index] < 0)
					{
						throw std::runtime_error("negative curve vertex count in " + source.name);
					}
					const size_t count = static_cast<size_t>(counts[curve_index]);
					if (point_start + count > points.size())
					{
						throw std::runtime_error("curve vertex counts exceed points in " + source.name);
					}
					size_t segment_count = 0;
					if (type == UsdGeomTokens->linear)
					{
						segment_count = count > 1 ? count - 1 : 0;
					}
					else if (basis == UsdGeomTokens->bezier)
					{
						if (count < 4 || (periodic ? count % 3 != 0 : (count - 1) % 3 != 0))
						{
							throw std::runtime_error("invalid Bezier curve vertex count in " + source.name);
						}
						segment_count = periodic ? count / 3 : (count - 1) / 3;
					}
					else if (pinned)
					{
						if (count < 2)
						{
							throw std::runtime_error("invalid pinned curve vertex count in " + source.name);
						}
						segment_count = count - 1;
					}
					else if (periodic)
					{
						if (count < 4)
						{
							throw std::runtime_error("invalid periodic cubic curve vertex count in " + source.name);
						}
						segment_count = count;
					}
					else
					{
						if (count < 4)
						{
							throw std::runtime_error("invalid cubic curve vertex count in " + source.name);
						}
						segment_count = count - 3;
					}
					for (size_t segment = 0; segment < segment_count; ++segment)
					{
						GfVec3f control[4];
						if (type == UsdGeomTokens->linear)
						{
							const GfVec3f p0 = points[point_start + segment];
							const GfVec3f p1 = points[point_start + segment + 1];
							control[0]		 = p0;
							control[1]		 = p0 + (p1 - p0) / 3.0f;
							control[2]		 = p0 + (p1 - p0) * (2.0f / 3.0f);
							control[3]		 = p1;
						}
						else if (basis == UsdGeomTokens->bezier)
						{
							for (int i = 0; i < 4; ++i)
							{
								control[i] = CurveControl(points,
														  point_start,
														  count,
														  static_cast<int>(segment * 3) + i,
														  periodic);
							}
						}
						else
						{
							const int	  first_control = static_cast<int>(segment) - (pinned ? 1 : 0);
							const GfVec3f p0 = CurveControl(points, point_start, count, first_control, periodic);
							const GfVec3f p1 = CurveControl(points, point_start, count, first_control + 1, periodic);
							const GfVec3f p2 = CurveControl(points, point_start, count, first_control + 2, periodic);
							const GfVec3f p3 = CurveControl(points, point_start, count, first_control + 3, periodic);
							if (basis == UsdGeomTokens->catmullRom)
							{
								control[0] = p1;
								control[1] = p1 + (p2 - p0) / 6.0f;
								control[2] = p2 - (p3 - p1) / 6.0f;
								control[3] = p2;
							}
							else
							{
								control[0] = (p0 + 4.0f * p1 + p2) / 6.0f;
								control[1] = (2.0f * p1 + p2) / 3.0f;
								control[2] = (p1 + 2.0f * p2) / 3.0f;
								control[3] = (p1 + 4.0f * p2 + p3) / 6.0f;
							}
						}
						float diameter = 0.0f;
						if (!widths.empty())
						{
							if (width_interpolation == UsdGeomTokens->constant)
							{
								diameter = WidthAt(widths, 0);
							}
							else if (width_interpolation == UsdGeomTokens->uniform)
							{
								diameter = WidthAt(widths, curve_index);
							}
							else if (width_interpolation == UsdGeomTokens->vertex)
							{
								const size_t width_control_count = type == UsdGeomTokens->linear ? 2u : 4u;
								for (size_t i = 0; i < width_control_count; ++i)
								{
									size_t control_index = segment * width_control_step + i;
									if (periodic && type == UsdGeomTokens->cubic)
									{
										control_index %= count;
									}
									diameter =
										std::max(diameter,
												 WidthAt(widths, point_start + std::min(control_index, count - 1)));
								}
							}
							else if (width_interpolation == UsdGeomTokens->varying)
							{
								const size_t varying_count = segment_count + (periodic ? 0u : 1u);
								diameter				   = std::max(
									WidthAt(widths, varying_start + segment),
									WidthAt(widths, varying_start + (segment + 1 < varying_count ? segment + 1 : 0)));
							}
							else
							{
								throw std::runtime_error("unsupported curve width interpolation in " + source.name);
							}
						}
						const Vec3f p[4] = {ToWorld(control[0], transform, z_up),
											ToWorld(control[1], transform, z_up),
											ToWorld(control[2], transform, z_up),
											ToWorld(control[3], transform, z_up)};
						scene->curves.emplace_back(p[0], p[1], p[2], p[3], diameter * 0.5f * radius_scale);
						scene->primitive_material_ids.push_back(static_cast<uint32_t>(source_index));
						++generated_count;
					}
					point_start += count;
					varying_start += segment_count + (periodic ? 0u : 1u);
				}
				if (point_start != points.size())
				{
					throw std::runtime_error("curve vertex counts do not cover all points in " + source.name);
				}
				if (!widths.empty())
				{
					const size_t expected_widths = width_interpolation == UsdGeomTokens->constant  ? 1
												   : width_interpolation == UsdGeomTokens->uniform ? counts.size()
												   : width_interpolation == UsdGeomTokens->vertex  ? points.size()
																								   : varying_start;
					if (widths.size() != expected_widths)
					{
						throw std::runtime_error("curve width count does not match interpolation in " + source.name);
					}
				}
				if (expected_counts && generated_count != (*expected_counts)[source_index])
				{
					throw std::runtime_error("curve primitive count of " + source.name +
											 " differs from the first frame at time " +
											 std::to_string(time.GetValue()));
				}
				if (found_counts)
				{
					found_counts->push_back(generated_count);
				}
			}
		}
	} // namespace

	bool UsdSupported()
	{
		return true;
	}

	bool LoadUsdFrames(const std::string	  &path,
					   const UsdLoadOptions	  &options,
					   std::vector<FlatScene> *frames,
					   Scene				  *first_scene)
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

			const double	 lowest	 = std::numeric_limits<double>::lowest();
			const double	 highest = std::numeric_limits<double>::max();
			const GfInterval interval(options.start_time.value_or(lowest), options.end_time.value_or(highest));

			std::vector<MeshSource>	 mesh_sources;
			std::vector<PointSource> point_sources;
			std::vector<CurveSource> curve_sources;
			std::set<double>		 sample_set;
			std::vector<double>		 samples;
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
					mesh_sources.push_back({mesh, prim.GetPath().GetString()});
					continue;
				}
				const UsdGeomPoints points(prim);
				if (points && IsLoadablePurpose(prim))
				{
					if (points.GetPointsAttr().GetTimeSamplesInInterval(interval, &samples))
					{
						sample_set.insert(samples.begin(), samples.end());
					}
					if (points.GetWidthsAttr().GetTimeSamplesInInterval(interval, &samples))
					{
						sample_set.insert(samples.begin(), samples.end());
					}
					point_sources.push_back({points, prim.GetPath().GetString()});
					continue;
				}
				const UsdGeomBasisCurves curves(prim);
				if (curves && IsLoadablePurpose(prim))
				{
					if (curves.GetPointsAttr().GetTimeSamplesInInterval(interval, &samples))
					{
						sample_set.insert(samples.begin(), samples.end());
					}
					if (curves.GetCurveVertexCountsAttr().GetTimeSamplesInInterval(interval, &samples))
					{
						sample_set.insert(samples.begin(), samples.end());
					}
					if (curves.GetWidthsAttr().GetTimeSamplesInInterval(interval, &samples))
					{
						sample_set.insert(samples.begin(), samples.end());
					}
					curve_sources.push_back({curves, prim.GetPath().GetString()});
				}
			}

			std::vector<UsdTimeCode> times;
			const size_t			 stride		  = std::max<uint32_t>(options.stride, 1u);
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

			// Visibility is evaluated once, at the first frame, to fix the primitive set.
			mesh_sources.erase(
				std::remove_if(mesh_sources.begin(),
							   mesh_sources.end(),
							   [&](const MeshSource &source)
							   {
								   return UsdGeomImageable(source.mesh.GetPrim()).ComputeVisibility(times.front()) ==
										  UsdGeomTokens->invisible;
							   }),
				mesh_sources.end());
			point_sources.erase(
				std::remove_if(point_sources.begin(),
							   point_sources.end(),
							   [&](const PointSource &source)
							   {
								   return UsdGeomImageable(source.points.GetPrim()).ComputeVisibility(times.front()) ==
										  UsdGeomTokens->invisible;
							   }),
				point_sources.end());
			curve_sources.erase(
				std::remove_if(curve_sources.begin(),
							   curve_sources.end(),
							   [&](const CurveSource &source)
							   {
								   return UsdGeomImageable(source.curves.GetPrim()).ComputeVisibility(times.front()) ==
										  UsdGeomTokens->invisible;
							   }),
				curve_sources.end());
			const unsigned primitive_types = static_cast<unsigned>(!mesh_sources.empty()) +
											 static_cast<unsigned>(!point_sources.empty()) +
											 static_cast<unsigned>(!curve_sources.empty());
			if (primitive_types > 1)
			{
				std::fprintf(
					stderr,
					"USD file '%s' mixes meshes, points, or curves; only one primitive type is supported per stage\n",
					path.c_str());
				return false;
			}
			if (primitive_types == 0)
			{
				std::fprintf(stderr, "USD file '%s' contains no visible supported geometry\n", path.c_str());
				return false;
			}

			std::vector<Material> materials;
			auto				  add_material = [&](const auto &source, const auto &geometry)
			{
				Material material;
				material.name		   = source.name;
				material.diffuse_color = AverageDisplayColor(geometry, times.front());
				materials.push_back(std::move(material));
			};
			for (const MeshSource &source : mesh_sources)
				add_material(source, source.mesh);
			for (const PointSource &source : point_sources)
				add_material(source, source.points);
			for (const CurveSource &source : curve_sources)
				add_material(source, source.curves);

			auto build_frame = [&](UsdGeomXformCache		 &cache,
								   UsdTimeCode				  time,
								   const std::vector<size_t> *expected_counts,
								   std::vector<size_t>		 *found_counts,
								   Scene					 *scene)
			{
				if (!mesh_sources.empty())
				{
					BuildScene(mesh_sources, materials, cache, time, z_up, expected_counts, scene);
					if (found_counts)
					{
						for (const Mesh &mesh : scene->meshes)
							found_counts->push_back(mesh.positions.size());
					}
				}
				else if (!point_sources.empty())
				{
					BuildPoints(point_sources, materials, cache, time, z_up, expected_counts, found_counts, scene);
				}
				else
				{
					BuildCurves(curve_sources, materials, cache, time, z_up, expected_counts, found_counts, scene);
				}
			};

			UsdGeomXformCache	first_cache;
			Scene				first;
			std::vector<size_t> primitive_counts;
			build_frame(first_cache, times.front(), nullptr, &primitive_counts, &first);
			if (first.PrimitiveCount() == 0)
			{
				std::fprintf(stderr, "USD file '%s' has no renderable primitives at its first frame\n", path.c_str());
				return false;
			}

			std::vector<FlatScene> loaded(times.size());
			loaded[0] = FlattenScene(first);

			std::atomic<size_t> next_frame{1};
			std::mutex			error_mutex;
			std::string			error;
			auto				worker = [&]()
			{
				UsdGeomXformCache cache;
				for (size_t frame = next_frame++; frame < times.size(); frame = next_frame++)
				{
					try
					{
						Scene scene;
						build_frame(cache, times[frame], &primitive_counts, nullptr, &scene);
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
				1u,
				std::min({kMaxLoadThreads, std::thread::hardware_concurrency(), static_cast<unsigned>(times.size())}));
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

			std::printf("Loaded USD '%s': %u primitive(s), %u triangle(s), %zu frame(s)%s\n",
						path.c_str(),
						first.PrimitiveCount(),
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
