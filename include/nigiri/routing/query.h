#pragma once

#include <cinttypes>
#include <vector>

#include "nigiri/common/interval.h"
#include "nigiri/routing/limits.h"
#include "nigiri/routing/location_match_mode.h"
#include "nigiri/types.h"

namespace nigiri::routing {

struct offset {
  location_idx_t location_;
  duration_t offset_;
  std::uint8_t type_;
};

using start_time_t = variant<unixtime_t, interval<unixtime_t>>;

struct query {
  start_time_t start_time_;
  location_match_mode start_match_mode_;
  location_match_mode dest_match_mode_;
  bool use_start_footpaths_{true};
  std::vector<offset> start_;
  std::vector<std::vector<offset>> destinations_;
  std::vector<std::vector<offset>> via_destinations_;
  cista::bitset<kNumClasses> allowed_classes_{
      cista::bitset<kNumClasses>::max()};
  std::uint8_t max_transfers_{kMaxTransfers};
  std::uint8_t min_connection_count_{0U};
  bool extend_interval_earlier_{false};
  bool extend_interval_later_{false};
};

}  // namespace nigiri::routing
