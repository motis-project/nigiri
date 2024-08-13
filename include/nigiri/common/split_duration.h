#pragma once

#include <cmath>

#include "nigiri/types.h"

namespace nigiri {

template <typename A, typename B>
std::common_type_t<A, B> time_mod(A const a, B const b) {
  return a < A{0} ? ((a % b) + b) % b : a % b;
}

inline std::pair<date::days, duration_t> split_time_mod(duration_t const i) {
  auto const b = time_mod(i.count(), 1440);
  auto const a = static_cast<int>((i.count() - b) / 1440);
  return {date::days{a}, duration_t{b}};
}

}  // namespace nigiri