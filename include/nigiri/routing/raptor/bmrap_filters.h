#pragma once

// The lower-bound-dijkstra skip trait bmrap_profile.cc uses to decide
// whether an engine it always bounds still needs a real lb array. Split out
// of bmrap_profile.cc's anonymous namespace so bmrap_test.cc can unit-test
// it directly, without a timetable/query.
namespace nigiri::routing {

// Whether `Algo`, run with bounds_ always set, still needs a real lb array.
// False if the engine never uses lb at all, or if it declares
// kNeedsLbWhenBounded = false (CPU raptor/basic_mcraptor: bound_prunes() is
// a tighter cutoff once bounded); true otherwise (e.g. gpu_mcraptor, the
// conservative default for an engine with no bounds_-aware fallback). Only
// valid for an Algo this file bounds unconditionally - ping never gets
// set_bounds(), so its need is decided by kUseLowerBounds alone.
template <typename Algo>
constexpr bool bounded_needs_lb() {
  if constexpr (!Algo::kUseLowerBounds) {
    return false;
  } else if constexpr (requires { Algo::kNeedsLbWhenBounded; }) {
    return Algo::kNeedsLbWhenBounded;
  } else {
    return true;
  }
}

}  // namespace nigiri::routing
