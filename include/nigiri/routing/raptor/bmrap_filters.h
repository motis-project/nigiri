#pragma once

namespace nigiri::routing {

// Whether `Algo`, always run with bounds, still needs lower bounds.
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
