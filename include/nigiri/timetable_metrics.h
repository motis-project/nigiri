#include <cstdint>
#include <limits>

#include "nigiri/timetable.h"
#include "nigiri/types.h"

namespace nigiri {

struct timetable_metrics {
  struct feed_metrics {
    std::uint64_t transport_days_;
    std::uint32_t locations_;
    std::uint32_t trips_;
    std::uint16_t first_ = std::numeric_limits<std::uint16_t>::max();
    std::uint16_t last_ = std::numeric_limits<std::uint16_t>::min();
  };
  vector_map<source_idx_t, feed_metrics> feeds_;
};

timetable_metrics get_metrics(timetable const&);
std::string to_str(timetable_metrics const&, timetable const&);

}  // namespace nigiri
