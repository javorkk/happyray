#include "obj_loader.h"
#include "scene.h"

#include <cctype>
#include <cstdio>
#include <sstream>
#include <unordered_map>

namespace happyray
{
	namespace
	{

		// 1-based or negative (relative) OBJ index -> 0-based; -1 if `token` is empty.
		int ResolveIndex(const std::string &token, int count)
		{
			if (token.empty())
			{
				return -1;
			}
			const int value = std::stoi(token);
			return value < 0 ? count + value : value - 1;
		}

		struct FaceVertexKey
		{
			int position_index;
			int uv_index;
			int normal_index;

			bool operator==(const FaceVertexKey &other) const
			{
				return position_index == other.position_index && uv_index == other.uv_index &&
					   normal_index == other.normal_index;
			}
		};

		struct FaceVertexKeyHash
		{
			size_t operator()(const FaceVertexKey &key) const
			{
				size_t h = static_cast<size_t>(key.position_index);
				h		 = h * 73856093u ^ static_cast<size_t>(key.uv_index) * 19349663u;
				h		 = h * 83492791u ^ static_cast<size_t>(key.normal_index);
				return h;
			}
		};

		FaceVertexKey ParseFaceVertexToken(const std::string &token, int position_count, int uv_count, int normal_count)
		{
			std::string parts[3];
			int			part_index = 0;
			for (char c : token)
			{
				if (c == '/')
				{
					++part_index;
				}
				else
				{
					parts[part_index] += c;
				}
			}
			FaceVertexKey key;
			key.position_index = ResolveIndex(parts[0], position_count);
			key.uv_index	   = ResolveIndex(parts[1], uv_count);
			key.normal_index   = ResolveIndex(parts[2], normal_count);
			return key;
		}

		// OBJ/MTL references are relative to the referencing file, not the working directory.
		std::string ResolveSiblingPath(const std::string &base_path, const std::string &relative_path)
		{
			const size_t slash = base_path.find_last_of("/\\");
			if (slash == std::string::npos)
			{
				return relative_path;
			}
			return base_path.substr(0, slash + 1) + relative_path;
		}

		// Reads only "newmtl" and "Kd".
		std::unordered_map<std::string, Vec3f> ParseMtlFile(const std::string &path)
		{
			std::unordered_map<std::string, Vec3f> diffuse_colors;
			std::FILE							  *file = std::fopen(path.c_str(), "r");
			if (!file)
			{
				std::fprintf(stderr, "LoadObj: failed to open mtllib '%s'\n", path.c_str());
				return diffuse_colors;
			}

			std::string current_material_name;
			char		line_buffer[1024];
			while (std::fgets(line_buffer, sizeof(line_buffer), file))
			{
				std::istringstream line(line_buffer);
				std::string		   keyword;
				line >> keyword;

				if (keyword == "newmtl")
				{
					line >> current_material_name;
				}
				else if (keyword == "Kd" && !current_material_name.empty())
				{
					Vec3f kd;
					line >> kd.x >> kd.y >> kd.z;
					diffuse_colors[current_material_name] = kd;
				}
			}
			std::fclose(file);
			return diffuse_colors;
		}

	} // namespace

	bool LoadObj(const std::string &path, Scene *scene)
	{
		std::FILE *file = std::fopen(path.c_str(), "r");
		if (!file)
		{
			std::fprintf(stderr, "LoadObj: failed to open '%s'\n", path.c_str());
			return false;
		}

		std::vector<Vec3f> positions;
		std::vector<Vec3f> normals;
		std::vector<Vec2f> uvs;

		Mesh mesh;
		mesh.name = path;

		std::unordered_map<FaceVertexKey, uint32_t, FaceVertexKeyHash> vertex_cache;
		std::unordered_map<std::string, Vec3f>						   mtllib_diffuse_colors;
		uint32_t													   current_material_id = 0;
		bool														   have_material	   = false;
		// Ids are assigned on an object's first face, so they stay consecutive (like string2shape).
		std::string								  current_object_name;
		std::string								  current_group_name;
		std::unordered_map<std::string, uint32_t> object_ids_by_name;
		bool									  have_object_id	= false;
		uint32_t								  current_object_id = 0;

		auto get_or_add_material = [&](const std::string &name) -> uint32_t
		{
			for (size_t i = 0; i < scene->materials.size(); ++i)
			{
				if (scene->materials[i].name == name)
				{
					return static_cast<uint32_t>(i);
				}
			}
			Material material;
			material.name = name;
			const auto it = mtllib_diffuse_colors.find(name);
			if (it != mtllib_diffuse_colors.end())
			{
				material.diffuse_color = it->second;
			}
			scene->materials.push_back(material);
			return static_cast<uint32_t>(scene->materials.size() - 1);
		};

		char line_buffer[1024];
		while (std::fgets(line_buffer, sizeof(line_buffer), file))
		{
			std::istringstream line(line_buffer);
			std::string		   keyword;
			line >> keyword;

			if (keyword == "v")
			{
				Vec3f p;
				line >> p.x >> p.y >> p.z;
				positions.push_back(p);
			}
			else if (keyword == "vn")
			{
				Vec3f n;
				line >> n.x >> n.y >> n.z;
				normals.push_back(n);
			}
			else if (keyword == "vt")
			{
				Vec2f uv;
				line >> uv.x >> uv.y;
				uvs.push_back(uv);
			}
			else if (keyword == "mtllib")
			{
				std::string mtl_filename;
				line >> mtl_filename;
				const std::unordered_map<std::string, Vec3f> parsed =
					ParseMtlFile(ResolveSiblingPath(path, mtl_filename));
				mtllib_diffuse_colors.insert(parsed.begin(), parsed.end());
			}
			else if (keyword == "usemtl")
			{
				std::string name;
				line >> name;
				current_material_id = get_or_add_material(name);
				have_material		= true;
			}
			else if (keyword == "o" || keyword == "g")
			{
				std::string name;
				std::getline(line >> std::ws, name);
				while (!name.empty() && std::isspace(static_cast<unsigned char>(name.back())))
				{
					name.pop_back();
				}
				if (keyword == "o")
				{
					current_object_name = name;
					current_group_name.clear();
				}
				else
				{
					current_group_name = name;
				}
				have_object_id = false;
			}
			else if (keyword == "f")
			{
				if (!have_object_id)
				{
					const std::string key = current_object_name + '\n' + current_group_name;
					current_object_id =
						object_ids_by_name.emplace(key, static_cast<uint32_t>(object_ids_by_name.size())).first->second;
					have_object_id = true;
				}
				std::vector<uint32_t> face_vertex_indices;
				std::string			  token;
				while (line >> token)
				{
					const FaceVertexKey key = ParseFaceVertexToken(token,
																   static_cast<int>(positions.size()),
																   static_cast<int>(uvs.size()),
																   static_cast<int>(normals.size()));

					auto it = vertex_cache.find(key);
					if (it != vertex_cache.end())
					{
						face_vertex_indices.push_back(it->second);
						continue;
					}

					Vec3f			 position;
					VertexAttributes attributes;
					if (key.position_index >= 0 && key.position_index < static_cast<int>(positions.size()))
					{
						position = positions[key.position_index];
					}
					if (key.uv_index >= 0 && key.uv_index < static_cast<int>(uvs.size()))
					{
						attributes.uv = uvs[key.uv_index];
					}
					if (key.normal_index >= 0 && key.normal_index < static_cast<int>(normals.size()))
					{
						attributes.normal = normals[key.normal_index];
					}

					const uint32_t new_index = static_cast<uint32_t>(mesh.positions.size());
					mesh.positions.push_back(position);
					mesh.attributes.push_back(attributes);
					vertex_cache.emplace(key, new_index);
					face_vertex_indices.push_back(new_index);
				}

				for (size_t i = 1; i + 1 < face_vertex_indices.size(); ++i)
				{
					mesh.indices.push_back(face_vertex_indices[0]);
					mesh.indices.push_back(face_vertex_indices[i]);
					mesh.indices.push_back(face_vertex_indices[i + 1]);
					mesh.material_ids.push_back(have_material ? current_material_id : 0);
					mesh.object_ids.push_back(current_object_id);
				}
			}
		}
		std::fclose(file);

		if (normals.empty() && !mesh.positions.empty())
		{
			std::vector<Vec3f> accumulated(mesh.positions.size(), Vec3f(0.0f, 0.0f, 0.0f));
			for (size_t tri = 0; tri < mesh.indices.size(); tri += 3)
			{
				const uint32_t i0 = mesh.indices[tri];
				const uint32_t i1 = mesh.indices[tri + 1];
				const uint32_t i2 = mesh.indices[tri + 2];
				const Vec3f	  &p0 = mesh.positions[i0];
				const Vec3f	  &p1 = mesh.positions[i1];
				const Vec3f	  &p2 = mesh.positions[i2];
				// Unnormalized, so contributions are area-weighted.
				const Vec3f face_normal = Cross(p1 - p0, p2 - p0);
				accumulated[i0] += face_normal;
				accumulated[i1] += face_normal;
				accumulated[i2] += face_normal;
			}
			for (size_t i = 0; i < mesh.positions.size(); ++i)
			{
				mesh.attributes[i].normal = Normalize(accumulated[i]);
			}
		}

		if (!mesh.indices.empty())
		{
			scene->meshes.push_back(std::move(mesh));
		}
		SubdivideLargeTriangles(scene);
		return true;
	}

} // namespace happyray
