// Stationary-view search scheduling: tries densities in [0.25x, 4x] the active grid type's
// original defaults, preferring whichever rebuilt candidate has the lower structural SAH cost
// (see src/accel/sah_cost.h) -- computed directly from its own just-built cell layout, with no
// rays cast and no camera/accumulation state involved at all. Unlike measuring real interactive
// frame-render time (this class's own previous design), a SAH cost is available the instant a
// candidate finishes building, so no multi-frame waiting/averaging is needed to evaluate one:
// each eligible frame rebuilds and evaluates exactly one new candidate.
#pragma once

#include <array>
#include <cmath>
#include <stdexcept>

#include "random.h"
#include "renderer.h"

namespace happyray {

class GridDensitySearch {
public:
    // `current_sah_cost` is the SAH cost of whichever grid is already active for `type` at this
    // moment (the compiled-in defaults the first time, or the previous search's own best
    // afterward) -- seeded as the baseline every candidate must beat, exactly like this class's
    // own previous design seeded its initial timing baseline from a real measured frame before
    // evaluating any candidate.
    void Reset(AccelType type, float current_sah_cost) {
        switch (type) {
            case AccelType::kUniformGrid:
                original_ = {UniformGrid::kDefaultDensity, 0};
                break;
            case AccelType::kTwoLevelGrid:
                original_ = {TwoLevelGrid::kDefaultTopLevelDensity,
                             TwoLevelGrid::kDefaultSecondLevelDensity};
                break;
            case AccelType::kIrregularGrid:
                original_ = {IrregularGrid::kDefaultTopLevelDensity,
                             IrregularGrid::kDefaultSecondLevelDensity};
                break;
            default: throw std::logic_error("Density search requires an active grid");
        }
        if (!std::isfinite(current_sah_cost) || current_sah_cost <= 0)
            throw std::invalid_argument("SAH cost must be finite and positive");
        type_ = type;
        attempts_ = 0;
        misses_ = stopped_[static_cast<size_t>(type_)] ? 10 : 0;
        has_best_cost_ = true;
        best_sah_cost_ = current_sah_cost;
    }

    // True once the view has settled (`still_frames` still-camera frames) and the search for
    // this accel type hasn't already stopped -- a candidate can then be rebuilt and evaluated
    // immediately, in the very same frame (see Evaluate below).
    bool CanRebuild(uint32_t still_frames) const {
        return still_frames >= 32 && !Stopped();
    }

    GridDensities Candidate() const {
        // Start at the range corners, then cover log-density space with Halton points.
        const bool uniform = original_.second_level == 0;
        const uint32_t corner_count = uniform ? 2 : 4;
        const float top = attempts_ < corner_count ? (attempts_ % 2 ? 4.0f : 0.25f) :
            std::exp2(4.0f * HaltonRadicalInverse(attempts_ - corner_count + 1, 2) - 2.0f);
        const float leaf = uniform ? 0.0f :
            attempts_ < corner_count ? (attempts_ < 2 ? 0.25f : 4.0f) :
            std::exp2(4.0f * HaltonRadicalInverse(attempts_ - corner_count + 1, 3) - 2.0f);
        return {original_.top_level * top, original_.second_level * leaf};
    }

    // Accepts or rejects the just-rebuilt candidate whose structural SAH cost is
    // `candidate_sah_cost`, by comparing it to the lowest cost seen so far in this search.
    // Returns true (and retains `candidate_sah_cost` as the new best) if it improved.
    bool Evaluate(float candidate_sah_cost) {
        if (!std::isfinite(candidate_sah_cost) || candidate_sah_cost <= 0)
            throw std::invalid_argument("SAH cost must be finite and positive");
        const bool improved = !has_best_cost_ || candidate_sah_cost < best_sah_cost_;
        if (improved) {
            best_sah_cost_ = candidate_sah_cost;
            has_best_cost_ = true;
        }
        ++attempts_;
        misses_ = improved ? 0 : misses_ + 1;
        stopped_[static_cast<size_t>(type_)] = Stopped();
        return improved;
    }

    uint32_t Attempts() const { return attempts_; }
    uint32_t Misses() const { return misses_; }
    bool Stopped() const { return misses_ >= 10; }

private:
    GridDensities original_;
    AccelType type_ = AccelType::kBruteForce;
    std::array<bool, 4> stopped_{};
    uint32_t attempts_ = 0;
    uint32_t misses_ = 0;
    bool has_best_cost_ = false;
    float best_sah_cost_ = 0;
};

}  // namespace happyray
