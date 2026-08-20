#include <array>
#include <cstdint>
#include <limits>
#include <string>

#include "nigiri/timetable.h"
#include "nigiri/types.h"

namespace nigiri {

struct timetable_metrics {
  struct feed_metrics {
    std::uint64_t transport_days_;
    std::uint32_t locations_;
    std::uint32_t trips_;
    std::uint16_t first_{std::numeric_limits<std::uint16_t>::max()};
    std::uint16_t last_{std::numeric_limits<std::uint16_t>::min()};
  };
  // Per profile, because a computed footpath layer replaces the one the
  // loader wrote and brings its own hubs with it.
  struct profile_metrics {
    std::uint64_t footpaths_{0U};
    std::uint64_t hub_pairs_{0U};
    std::uint32_t hubs_{0U};
    std::uint32_t rule_hubs_{0U};
  };

  vector_map<source_idx_t, feed_metrics> feeds_;
  std::array<profile_metrics, kNProfiles> profiles_{};
  std::uint32_t routes_{0U};
};

timetable_metrics get_metrics(timetable const&);
std::string to_str(timetable_metrics const&, timetable const&);

}  // namespace nigiri
