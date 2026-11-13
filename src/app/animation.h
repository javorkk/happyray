// In-memory OBJ/USD keyframe loading and frame-step interpolation for the viewer.
#pragma once

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "obj_loader.h"
#include "scene.h"
#include "usd_loader.h"

namespace happyray
{

	class KeyframeAnimation
	{
	  public:
		KeyframeAnimation() = default;
		KeyframeAnimation(std::vector<FlatScene> frames, float frame_step)
			: frames_(std::move(frames)), frame_step_(frame_step)
		{
		}

		bool LoadConfig(const std::string &config_path, Scene *first_scene)
		{
			frames_.clear();
			frame_step_	  = 0.4f;
			current_time_ = 0.0f;
			paused_		  = false;
			camera_config_path_.clear();
			if (!first_scene)
			{
				return false;
			}

			Config config;
			if (!ReadConfig(config_path, &config) || !HasConfiguredSequence(config))
			{
				return false;
			}
			const std::filesystem::path config_directory = std::filesystem::path(config_path).parent_path();
			if (!config.camera_path.empty())
			{
				config.camera_path = ResolvePath(config_directory, config.camera_path);
			}
			if (!config.usd_path.empty())
			{
				return LoadUsd(ResolvePath(config_directory, config.usd_path),
							   config.usd_options,
							   config.frame_step,
							   config.camera_path,
							   first_scene);
			}
			config.prefix = ResolvePath(config_directory, config.prefix);

			std::vector<std::string> paths;
			if (!MakeFramePaths(config.prefix, config.suffix, config.frame_count, &paths))
			{
				return false;
			}
			return LoadFrames(paths, config.frame_step, config.camera_path, first_scene);
		}

		// Loads the frames of a USD file (see usd_loader.h) as an animation. A file with a single
		// frame loads successfully but is not an animation (IsLoaded() stays false); its geometry
		// is returned in `first_scene`. `camera_path` defaults to `<usd stem>.camera.cfg` next to
		// the USD file.
		bool LoadUsd(const std::string	  &usd_path,
					 const UsdLoadOptions &options,
					 float				   frame_step,
					 const std::string	  &camera_path,
					 Scene				  *first_scene)
		{
			frames_.clear();
			frame_step_	  = frame_step;
			current_time_ = 0.0f;
			paused_		  = false;
			camera_config_path_.clear();
			if (!first_scene)
			{
				return false;
			}
			Scene first_frame_scene;
			if (!LoadUsdFrames(usd_path, options, &frames_, &first_frame_scene))
			{
				return false;
			}
			*first_scene		= std::move(first_frame_scene);
			camera_config_path_ = camera_path;
			if (camera_config_path_.empty() || !std::filesystem::exists(camera_config_path_))
			{
				std::filesystem::path path(usd_path);
				path.replace_filename(path.stem().string() + ".camera.cfg");
				camera_config_path_ = path.string();
			}
			return true;
		}

		bool LoadUsd(const std::string &usd_path, Scene *first_scene)
		{
			return LoadUsd(usd_path, UsdLoadOptions{}, 0.4f, std::string(), first_scene);
		}

		bool IsLoaded() const
		{
			return frames_.size() > 1;
		}
		bool IsPaused() const
		{
			return paused_;
		}
		void TogglePause()
		{
			paused_ = !paused_;
		}
		float FrameStep() const
		{
			return frame_step_;
		}
		const std::string &CameraConfigPath() const
		{
			return camera_config_path_;
		}

		void Advance()
		{
			if (!paused_ && IsLoaded())
			{
				current_time_ += frame_step_;
				if (current_time_ >= static_cast<float>(frames_.size()))
				{
					current_time_ = std::fmod(current_time_, static_cast<float>(frames_.size()));
				}
			}
		}

		FlatScene InterpolatedScene() const
		{
			if (!IsLoaded())
			{
				return {};
			}
			const float		 frame_position = std::fmod(current_time_, static_cast<float>(frames_.size()));
			const size_t	 first_index	= static_cast<size_t>(frame_position);
			const size_t	 second_index	= (first_index + 1) % frames_.size();
			const float		 amount			= frame_position - static_cast<float>(first_index);
			const FlatScene &first			= frames_[first_index];
			FlatScene		 result			= frames_[second_index];
			const size_t	 position_count = std::min(first.positions.size(), result.positions.size());
			for (size_t i = 0; i < position_count; ++i)
			{
				result.positions[i] = Lerp(first.positions[i], result.positions[i], amount);
			}
			const size_t normal_count = std::min(first.normals.size(), result.normals.size());
			for (size_t i = 0; i < normal_count; ++i)
			{
				result.normals[i] = Normalize(Lerp(first.normals[i], result.normals[i], amount));
			}
			const size_t point_count = std::min(first.points.size(), result.points.size());
			for (size_t i = 0; i < point_count; ++i)
			{
				result.points[i].vtx[0] = Lerp(first.points[i].vtx[0], result.points[i].vtx[0], amount);
				result.points[i].radius =
					first.points[i].radius + (result.points[i].radius - first.points[i].radius) * amount;
			}
			const size_t curve_count = std::min(first.curves.size(), result.curves.size());
			for (size_t i = 0; i < curve_count; ++i)
			{
				for (int control = 0; control < BezierCurve::kNumVertices; ++control)
				{
					result.curves[i].vtx[control] =
						Lerp(first.curves[i].vtx[control], result.curves[i].vtx[control], amount);
				}
				result.curves[i].radius =
					first.curves[i].radius + (result.curves[i].radius - first.curves[i].radius) * amount;
			}
			return result;
		}

	  private:
		struct Config
		{
			std::string	   prefix;
			std::string	   suffix;
			std::string	   camera_path;
			std::string	   usd_path;
			UsdLoadOptions usd_options;
			size_t		   frame_count = 0;
			float		   frame_step  = 0.4f;
		};

		static std::string Trim(std::string value)
		{
			const size_t start = value.find_first_not_of(" \t\r\n");
			if (start == std::string::npos)
			{
				return {};
			}
			const size_t end = value.find_last_not_of(" \t\r\n");
			return value.substr(start, end - start + 1);
		}

		static bool ReadConfig(const std::string &path, Config *config)
		{
			std::ifstream file(path);
			if (!file)
			{
				return false;
			}
			std::string line;
			while (std::getline(file, line))
			{
				const size_t comment = line.find('#');
				if (comment != std::string::npos)
				{
					line.erase(comment);
				}
				const size_t equals = line.find('=');
				if (equals == std::string::npos)
				{
					continue;
				}
				const std::string key	= Trim(line.substr(0, equals));
				const std::string value = Trim(line.substr(equals + 1));
				if (key == "frame prefix" || key == "frameprefix" || key == "framename")
				{
					config->prefix = value;
				}
				else if (key == "frame suffix" || key == "framesuffix")
				{
					config->suffix = value;
				}
				else if (key == "number of frames" || key == "frame number" || key == "number")
				{
					try
					{
						config->frame_count = static_cast<size_t>(std::stoul(value));
					}
					catch (...)
					{
						config->frame_count = 0;
					}
				}
				else if (key == "frame step" || key == "framestep" || key == "step size")
				{
					try
					{
						const float step = std::stof(value);
						if (std::isfinite(step) && step > 0.0f)
						{
							config->frame_step = step;
						}
					}
					catch (...)
					{
						config->frame_step = 0.4f;
					}
				}
				else if (key == "usd" || key == "usd file")
				{
					config->usd_path = value;
				}
				else if (key == "start time" || key == "end time")
				{
					try
					{
						(key == "start time" ? config->usd_options.start_time : config->usd_options.end_time) =
							std::stod(value);
					}
					catch (...)
					{
					}
				}
				else if (key == "frame stride")
				{
					try
					{
						config->usd_options.stride = static_cast<uint32_t>(std::stoul(value));
					}
					catch (...)
					{
						config->usd_options.stride = 1;
					}
				}
				else if (key == "camera")
				{
					config->camera_path = value;
				}
			}
			return true;
		}

		static bool HasConfiguredSequence(const Config &config)
		{
			return !config.usd_path.empty() ||
				   (!config.prefix.empty() && !config.suffix.empty() && config.frame_count > 1);
		}

		static bool MakeFramePaths(const std::string		&prefix,
								   const std::string		&suffix,
								   size_t					 frame_count,
								   std::vector<std::string> *paths)
		{
			if (frame_count < 2)
			{
				return false;
			}
			const size_t digits = std::to_string(frame_count - 1).size();
			for (size_t i = 0; i < frame_count; ++i)
			{
				std::ostringstream frame_number;
				frame_number << std::setw(static_cast<int>(digits)) << std::setfill('0') << i;
				paths->push_back(prefix + frame_number.str() + suffix);
			}
			return true;
		}

		static std::string ResolvePath(const std::filesystem::path &base, const std::string &value)
		{
			std::filesystem::path path(value);
			if (path.is_relative())
			{
				path = base / path;
			}
			return path.lexically_normal().string();
		}

		bool LoadFrames(const std::vector<std::string> &paths,
						float							frame_step,
						const std::string			   &camera_path,
						Scene						   *first_scene)
		{
			Scene first_frame_scene;
			for (size_t i = 0; i < paths.size(); ++i)
			{
				Scene frame_scene;
				if (!LoadObj(paths[i], &frame_scene))
				{
					frames_.clear();
					return false;
				}
				frames_.push_back(FlattenScene(frame_scene));
				if (i == 0)
				{
					first_frame_scene = std::move(frame_scene);
				}
			}
			if (frames_.size() < 2)
			{
				frames_.clear();
				return false;
			}
			*first_scene		= std::move(first_frame_scene);
			frame_step_			= frame_step;
			camera_config_path_ = camera_path;
			if (camera_config_path_.empty() || !std::filesystem::exists(camera_config_path_))
			{
				camera_config_path_ = CameraConfigPathForFrame(paths.front());
			}
			return true;
		}

		static std::string CameraConfigPathForFrame(const std::string &frame_path)
		{
			std::filesystem::path path(frame_path);
			std::string			  stem = path.stem().string();
			while (!stem.empty() && stem.back() >= '0' && stem.back() <= '9')
			{
				stem.pop_back();
			}
			path.replace_filename(stem + ".camera.cfg");
			return path.string();
		}

		std::vector<FlatScene> frames_;
		float				   frame_step_	 = 0.4f;
		float				   current_time_ = 0.0f;
		bool				   paused_		 = false;
		std::string			   camera_config_path_;
	};

} // namespace happyray