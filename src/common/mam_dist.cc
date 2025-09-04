#include "nigiri/common/mam_dist.h"

#include <cassert>
#include <cstdlib>

namespace nigiri {

std::pair<i32_minutes, date::days> mam_dist(i32_minutes const a,
                                            i32_minutes const b) {
  assert(i32_minutes{0} <= a && a < i32_minutes{1440});
  assert(i32_minutes{0} <= b && b < i32_minutes{1440});

  auto const diff = (a - b).count();
  auto const abs = std::abs(diff);

  if (abs > 1440 / 2) {
    return {i32_minutes{1440 - abs}, date::days{diff < 0 ? -1 : +1}};
  } else {
    return {i32_minutes{abs}, date::days{0}};
  }
}

}  // namespace nigiri