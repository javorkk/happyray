// OpenUSD loader tests on tiny generated .usda files. Built only with -DHAPPYRAY_ENABLE_USD=ON.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "../src/app/animation.h"
#include "../src/app/usd_loader.h"

namespace
{

	int g_failures = 0;

#define HAPPYRAY_CHECK(cond)                                                                                           \
	do                                                                                                                 \
	{                                                                                                                  \
		if (!(cond))                                                                                                   \
		{                                                                                                              \
			std::fprintf(stderr, "FAILED: %s (%s:%d)\n", #cond, __FILE__, __LINE__);                                   \
			++g_failures;                                                                                              \
		}                                                                                                              \
	} while (0)

	bool NearlyEqual(float a, float b, float eps = 1e-4f)
	{
		return std::fabs(a - b) < eps;
	}

	bool NearlyEqual(const happyray::Vec3f &a, const happyray::Vec3f &b)
	{
		return NearlyEqual(a.x, b.x) && NearlyEqual(a.y, b.y) && NearlyEqual(a.z, b.z);
	}

	std::string WriteFile(const std::filesystem::path &directory, const std::string &name, const std::string &text)
	{
		const std::filesystem::path path = directory / name;
		std::ofstream(path) << text;
		return path.string();
	}

	// A translated quad whose first point moves over time codes 1-3; `extra_point_at_time_3` adds
	// an invalid extra point at the last sample.
	std::string QuadStage(const std::string &up_axis, bool extra_point_at_time_3)
	{
		std::string points_at_3 = "(1, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)";
		if (extra_point_at_time_3)
		{
			points_at_3 += ", (5, 5, 5)";
		}
		return "#usda 1.0\n(\n    upAxis = \"" + up_axis +
			   "\"\n    startTimeCode = 1\n    endTimeCode = 3\n)\n"
			   "def Xform \"root\"\n{\n"
			   "    double3 xformOp:translate = (10, 0, 0)\n"
			   "    uniform token[] xformOpOrder = [\"xformOp:translate\"]\n"
			   "    def Mesh \"quad\"\n    {\n"
			   "        int[] faceVertexCounts = [4]\n"
			   "        int[] faceVertexIndices = [0, 1, 2, 3]\n"
			   "        point3f[] points.timeSamples = {\n"
			   "            1: [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)],\n"
			   "            2: [(0.5, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)],\n"
			   "            3: [" +
			   points_at_3 +
			   "],\n"
			   "        }\n"
			   "        color3f[] primvars:displayColor = [(1, 0, 0), (0, 0, 1)] (interpolation = \"uniform\")\n"
			   "    }\n"
			   "    def Mesh \"hidden\"\n    {\n"
			   "        token visibility = \"invisible\"\n"
			   "        int[] faceVertexCounts = [3]\n"
			   "        int[] faceVertexIndices = [0, 1, 2]\n"
			   "        point3f[] points = [(0, 0, 0), (1, 0, 0), (0, 1, 0)]\n"
			   "    }\n"
			   "    def Mesh \"guide\"\n    {\n"
			   "        uniform token purpose = \"guide\"\n"
			   "        int[] faceVertexCounts = [3]\n"
			   "        int[] faceVertexIndices = [0, 1, 2]\n"
			   "        point3f[] points = [(0, 0, 0), (1, 0, 0), (0, 1, 0)]\n"
			   "    }\n"
			   "}\n";
	}

	void TestLoadsAnimatedQuad(const std::filesystem::path &directory)
	{
		const std::string path = WriteFile(directory, "quad.usda", QuadStage("Y", false));
		HAPPYRAY_CHECK(happyray::IsUsdPath(path));
		HAPPYRAY_CHECK(!happyray::IsUsdPath("scene.obj"));
		HAPPYRAY_CHECK(happyray::UsdSupported());

		std::vector<happyray::FlatScene> frames;
		happyray::Scene					 first;
		HAPPYRAY_CHECK(happyray::LoadUsdFrames(path, {}, &frames, &first));
		HAPPYRAY_CHECK(frames.size() == 3);
		HAPPYRAY_CHECK(first.meshes.size() == 1); // The invisible and guide meshes are skipped.
		HAPPYRAY_CHECK(first.TriangleCount() == 2);
		HAPPYRAY_CHECK(first.materials.size() == 1);
		if (!first.materials.empty())
		{
			HAPPYRAY_CHECK(NearlyEqual(first.materials[0].diffuse_color, happyray::Vec3f(0.5f, 0.0f, 0.5f)));
		}
		if (frames.size() == 3)
		{
			HAPPYRAY_CHECK(frames[0].positions.size() == 4 && frames[0].indices.size() == 6);
			HAPPYRAY_CHECK(NearlyEqual(frames[0].positions[0], happyray::Vec3f(10.0f, 0.0f, 0.0f)));
			HAPPYRAY_CHECK(NearlyEqual(frames[1].positions[0], happyray::Vec3f(10.5f, 0.0f, 0.0f)));
			HAPPYRAY_CHECK(NearlyEqual(frames[2].positions[0], happyray::Vec3f(11.0f, 0.0f, 0.0f)));
			// Counter-clockwise quad in XY: right-handed winding gives +Z normals.
			HAPPYRAY_CHECK(NearlyEqual(frames[0].normals[1], happyray::Vec3f(0.0f, 0.0f, 1.0f)));
		}

		happyray::KeyframeAnimation animation;
		happyray::Scene				scene;
		HAPPYRAY_CHECK(animation.LoadUsd(path, &scene));
		HAPPYRAY_CHECK(animation.IsLoaded());
		HAPPYRAY_CHECK(animation.CameraConfigPath() == (directory / "quad.camera.cfg").string());
	}

	void TestOptionsAndZUp(const std::filesystem::path &directory)
	{
		const std::string		 path = WriteFile(directory, "quad_z.usda", QuadStage("Z", false));
		happyray::UsdLoadOptions options;
		options.stride = 2;
		std::vector<happyray::FlatScene> frames;
		happyray::Scene					 first;
		HAPPYRAY_CHECK(happyray::LoadUsdFrames(path, options, &frames, &first));
		HAPPYRAY_CHECK(frames.size() == 2); // Time codes 1 and 3.
		if (frames.size() == 2)
		{
			// Z-up (x, y, z) becomes Y-up (x, z, -y); the quad's +Z normal becomes +Y.
			HAPPYRAY_CHECK(NearlyEqual(frames[1].positions[2], happyray::Vec3f(11.0f, 0.0f, -1.0f)));
			HAPPYRAY_CHECK(NearlyEqual(frames[0].normals[1], happyray::Vec3f(0.0f, 1.0f, 0.0f)));
		}

		happyray::UsdLoadOptions range;
		range.start_time = 2.0;
		range.end_time	 = 2.0;
		HAPPYRAY_CHECK(happyray::LoadUsdFrames(path, range, &frames, &first));
		HAPPYRAY_CHECK(frames.size() == 1);
		if (!frames.empty())
		{
			HAPPYRAY_CHECK(NearlyEqual(frames[0].positions[0], happyray::Vec3f(10.5f, 0.0f, 0.0f)));
		}

		// A single frame loads, but is not an animation.
		happyray::KeyframeAnimation animation;
		happyray::Scene				scene;
		HAPPYRAY_CHECK(animation.LoadUsd(path, range, 0.4f, std::string(), &scene));
		HAPPYRAY_CHECK(!animation.IsLoaded());
		HAPPYRAY_CHECK(scene.TriangleCount() == 2);
	}

	void TestAnimationConfig(const std::filesystem::path &directory)
	{
		WriteFile(directory, "cfg_quad.usda", QuadStage("Y", false));
		const std::string			config = WriteFile(directory,
													   "scene.cfg",
													   "usd = cfg_quad.usda\n"
													   "start time = 2\n"
													   "frame step = 0.5\n"
													   "camera = quad.camera.cfg\n");
		happyray::KeyframeAnimation animation;
		happyray::Scene				scene;
		HAPPYRAY_CHECK(animation.LoadConfig(config, &scene));
		HAPPYRAY_CHECK(animation.IsLoaded()); // Time codes 2 and 3.
		HAPPYRAY_CHECK(NearlyEqual(animation.FrameStep(), 0.5f));
		HAPPYRAY_CHECK(animation.InterpolatedScene().positions.size() == 4);
	}

	void TestAnimatedPoints(const std::filesystem::path &directory)
	{
		const std::string path =
			WriteFile(directory,
					  "points.usda",
					  "#usda 1.0\n(\n    upAxis = \"Z\"\n)\n"
					  "def Xform \"root\"\n{\n"
					  "    double3 xformOp:translate = (10, 0, 0)\n"
					  "    uniform token[] xformOpOrder = [\"xformOp:translate\"]\n"
					  "    def Points \"cloud\"\n    {\n"
					  "        point3f[] points.timeSamples = {\n"
					  "            1: [(0, 0, 0), (1, 0, 0)],\n"
					  "            2: [(0, 0, 1), (1, 0, 1)],\n"
					  "        }\n"
					  "        float[] widths = [0.2, 0.4]\n"
					  "        color3f[] primvars:displayColor = [(0, 1, 0)] (interpolation = \"constant\")\n"
					  "    }\n}\n");
		std::vector<happyray::FlatScene> frames;
		happyray::Scene					 first;
		HAPPYRAY_CHECK(happyray::LoadUsdFrames(path, {}, &frames, &first));
		HAPPYRAY_CHECK(first.Type() == happyray::PrimitiveType::kPoint);
		HAPPYRAY_CHECK(first.points.size() == 2 && first.PrimitiveCount() == 2);
		HAPPYRAY_CHECK(frames.size() == 2);
		if (first.points.size() == 2 && frames.size() == 2)
		{
			HAPPYRAY_CHECK(NearlyEqual(first.points[0].vtx[0], happyray::Vec3f(10.0f, 0.0f, 0.0f)));
			HAPPYRAY_CHECK(NearlyEqual(first.points[0].radius, 0.1f));
			HAPPYRAY_CHECK(NearlyEqual(first.points[1].radius, 0.2f));
			HAPPYRAY_CHECK(NearlyEqual(frames[1].points[0].vtx[0], happyray::Vec3f(10.0f, 1.0f, 0.0f)));
		}
		happyray::KeyframeAnimation animation(frames, 0.5f);
		animation.Advance();
		const happyray::FlatScene interpolated = animation.InterpolatedScene();
		if (!interpolated.points.empty())
		{
			HAPPYRAY_CHECK(NearlyEqual(interpolated.points[0].vtx[0], happyray::Vec3f(10.0f, 0.5f, 0.0f)));
		}
	}

	void TestAnimatedBasisCurves(const std::filesystem::path &directory)
	{
		const std::string path = WriteFile(directory,
										   "curves.usda",
										   "#usda 1.0\n(\n    startTimeCode = 1\n    endTimeCode = 2\n)\n"
										   "def BasisCurves \"hair\"\n{\n"
										   "    uniform token type = \"cubic\"\n"
										   "    uniform token basis = \"bezier\"\n"
										   "    uniform token wrap = \"nonperiodic\"\n"
										   "    int[] curveVertexCounts = [4]\n"
										   "    float[] widths = [0.2, 0.2, 0.2, 0.2]\n"
										   "    point3f[] points.timeSamples = {\n"
										   "        1: [(0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0)],\n"
										   "        2: [(0, 1, 0), (1, 1, 0), (2, 1, 0), (3, 1, 0)],\n"
										   "    }\n}\n");
		std::vector<happyray::FlatScene> frames;
		happyray::Scene					 first;
		HAPPYRAY_CHECK(happyray::LoadUsdFrames(path, {}, &frames, &first));
		HAPPYRAY_CHECK(first.Type() == happyray::PrimitiveType::kBezierCurve);
		HAPPYRAY_CHECK(first.curves.size() == 1 && first.PrimitiveCount() == 1);
		HAPPYRAY_CHECK(frames.size() == 2);
		if (first.curves.size() == 1 && frames.size() == 2)
		{
			HAPPYRAY_CHECK(NearlyEqual(first.curves[0].vtx[0], happyray::Vec3f(0.0f, 0.0f, 0.0f)));
			HAPPYRAY_CHECK(NearlyEqual(first.curves[0].radius, 0.1f));
			HAPPYRAY_CHECK(NearlyEqual(frames[1].curves[0].vtx[3], happyray::Vec3f(3.0f, 1.0f, 0.0f)));
		}
		happyray::KeyframeAnimation animation(frames, 0.5f);
		animation.Advance();
		const happyray::FlatScene interpolated = animation.InterpolatedScene();
		if (!interpolated.curves.empty())
		{
			HAPPYRAY_CHECK(NearlyEqual(interpolated.curves[0].vtx[0], happyray::Vec3f(0.0f, 0.5f, 0.0f)));
		}
	}

	void TestFailures(const std::filesystem::path &directory)
	{
		std::vector<happyray::FlatScene> frames;
		happyray::Scene					 first;
		HAPPYRAY_CHECK(!happyray::LoadUsdFrames((directory / "missing.usd").string(), {}, &frames, &first));
		HAPPYRAY_CHECK(frames.empty());

		const std::string changing_points = WriteFile(directory, "changing.usda", QuadStage("Y", true));
		HAPPYRAY_CHECK(!happyray::LoadUsdFrames(changing_points, {}, &frames, &first));
		HAPPYRAY_CHECK(frames.empty());

		const std::string no_meshes = WriteFile(directory, "empty.usda", "#usda 1.0\ndef Xform \"root\"\n{\n}\n");
		HAPPYRAY_CHECK(!happyray::LoadUsdFrames(no_meshes, {}, &frames, &first));

		const std::string mixed = WriteFile(directory,
											"mixed.usda",
											"#usda 1.0\ndef Xform \"root\"\n{\n"
											"    def Points \"cloud\" { point3f[] points = [(0, 0, 0)] }\n"
											"    def Mesh \"tri\" { point3f[] points = [(0, 0, 0), (1, 0, 0), (0, 1, "
											"0)] int[] faceVertexCounts = [3] int[] faceVertexIndices = [0, 1, 2] }\n"
											"}\n");
		HAPPYRAY_CHECK(!happyray::LoadUsdFrames(mixed, {}, &frames, &first));
		HAPPYRAY_CHECK(frames.empty());
	}

} // namespace

int main()
{
	const std::filesystem::path directory =
		std::filesystem::temp_directory_path() / ("happyray_usd_tests_" + std::to_string(std::rand()));
	std::filesystem::create_directories(directory);

	TestLoadsAnimatedQuad(directory);
	TestOptionsAndZUp(directory);
	TestAnimationConfig(directory);
	TestAnimatedPoints(directory);
	TestAnimatedBasisCurves(directory);
	TestFailures(directory);

	std::filesystem::remove_all(directory);
	if (g_failures == 0)
	{
		std::printf("All USD loader tests passed.\n");
		return EXIT_SUCCESS;
	}
	std::fprintf(stderr, "%d USD loader test check(s) failed.\n", g_failures);
	return EXIT_FAILURE;
}
