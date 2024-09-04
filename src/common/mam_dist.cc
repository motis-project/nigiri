#include "nigiri/common/mam_dist.h"

#include <cassert>
#include <cstdlib>

namespace nigiri {

std::uint16_t mam_dist(std::uint16_t const a, std::uint16_t const b) {
  assert(a < 1440U);
  assert(b < 1440U);

  auto abs = std::abs(a - b);
  return static_cast<std::uint16_t>((abs > 1440 / 2) ? (1440 - abs) : abs);
}

}  // namespace nigiri