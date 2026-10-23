// Structural SAH (surface-area-heuristic) cost model shared by every acceleration structure
// (UniformGrid::ComputeSahCost, TwoLevelGrid::ComputeSahCost, IrregularGrid::ComputeSahCost --
// see those headers): an "expected cost" estimate (surface-area-weighted traversal-step plus
// intersection-test cost, summed over every cell) computed directly from a built grid's own
// cell layout, without casting any rays. `traversal`/`intersection` (SahCostConstants below) are
// measured once, not guessed -- MeasureSahCalibration() times the actual routines each
// structure's own traversal code calls, against small synthetic test cases, so the two costs
// (and, for traversal, each structure's own cost) stay honest relative to each other and to
// whichever hardware this binary actually runs on.
#pragma once

#include "vec.h"

namespace happyray {

// `traversal` is one traversal-step's measured cost, `intersection` one ray/triangle
// intersection-test's, both in nanoseconds.
struct SahCostConstants {
    float traversal = 0;
    float intersection = 0;
};

// One structure's own measured traversal-step cost is deliberately not shared with the others
// (see MeasureSahCalibration below), since the whole point of measuring rather than guessing is
// to let a structure's own extra indirection (e.g. IrregularGridView's local_entries lookup), or
// lack thereof, show up as a genuinely different cost; `intersection` *is* shared across all
// three, since IntersectRayTriangle/IntersectTrianglesClosest is the exact same routine
// regardless of which structure calls it.
struct SahCalibration {
    SahCostConstants uniform;
    SahCostConstants two_level;
    SahCostConstants irregular;
};

// Runs the heuristic calibration test (see sah_cost.cpp): times each structure's own real
// traversal-step code against a small mock, empty-space grid, and times one shared ray/triangle
// intersection test against a fixed triangle. This runs a handful of timed micro-benchmarks
// (order ~100ms total); call it once (e.g. at application startup), not per frame or per grid
// rebuild.
SahCalibration MeasureSahCalibration();

// Prints each structure's own measured traversal-step cost and the one shared intersection-test
// cost (ns/step, ns/test) to stdout -- a diagnostic a caller can opt into right after measuring
// `calibration`, not a side effect of MeasureSahCalibration() itself.
void PrintSahCalibration(const SahCalibration& calibration);

// Half of a box's own surface area (x*y + y*z + z*x) -- the common per-cell weight every
// ComputeSahCost() implementation sums, normalized by the scene bounding box's own half surface
// area.
HAPPYRAY_HD inline float HalfSurfaceArea(const Vec3& extent) {
    return extent.x * extent.y + extent.y * extent.z + extent.z * extent.x;
}

}  // namespace happyray
