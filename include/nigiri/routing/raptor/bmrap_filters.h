#pragma once

namespace nigiri::routing {

// Whether `Algo`, run with bounds always set (bmrap_profile.cc), still needs a
// real lb array: not if it never uses lb or declares kNeedsLbWhenBounded =
// false (CPU engines, where bound pruning is tighter). The default is
// conservative. The ping is never bounded, so kUseLowerBounds alone decides.
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
