#pragma once

#include <cinttypes>
#include <limits>
#include <string>

#include "nigiri/types.h"

namespace nigiri::routing {

using clasz_mask_t = std::uint16_t;

constexpr inline clasz_mask_t all_clasz_allowed() {
  return std::numeric_limits<clasz_mask_t>::max();
}

constexpr inline clasz_mask_t to_mask(clasz const c) {
  auto const c_as_int = static_cast<std::underlying_type_t<clasz>>(c);
  return static_cast<clasz_mask_t>(1U << c_as_int);
}

constexpr inline bool is_allowed(clasz_mask_t const mask, clasz const c) {
  auto const c_as_mask = to_mask(c);
  return (mask & c_as_mask) == c_as_mask;
}

std::string to_str(clasz_mask_t const x);

}  // namespace nigiri::routing