// SAH cost model for the grids, computed from cell layout with measured (not guessed) per-step
// traversal and per-test intersection costs.
#pragma once

#include "../core/vec.h"

namespace happyray
{

	// Costs in nanoseconds.
	struct SahCostConstants
	{
		float traversal	   = 0;
		float intersection = 0;
	};

	// Traversal cost is measured per structure; intersection cost is the same for all.
	struct SahCalibration
	{
		SahCostConstants uniform;
		SahCostConstants two_level;
		SahCostConstants irregular;
	};

	// Times micro-benchmarks (~100 ms); call once at startup.
	SahCalibration MeasureSahCalibration();

	void PrintSahCalibration(const SahCalibration &calibration);

	HAPPYRAY_HD inline float HalfSurfaceArea(const Vec3f &extent)
	{
		return extent.x * extent.y + extent.y * extent.z + extent.z * extent.x;
	}

} // namespace happyray
