#include "nigiri/common/mam_dist.h"

#include <cassert>
#include <cstdlib>

namespace nigiri {

std::pair<std::uint16_t, std::int16_t> mam_dist(std::uint16_t const expected,
                                                std::uint16_t const actual) {
  assert(expected < 1440U);
  assert(actual < 1440U);

  auto const diff = expected - actual;
  auto const abs = std::abs(diff);

  if (abs > 1440 / 2) {
    if (diff < 0) {
      return {1440 - abs, -1};
    } else {
      return {1440 - abs, +1};
    }
  } else {
    return {abs, 0};
  }
}

}  // namespace nigiri