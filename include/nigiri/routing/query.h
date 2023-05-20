#pragma once

#include <cinttypes>
#include <variant>
#include <vector>

#include "nigiri/common/interval.h"
#include "nigiri/footpath.h"
#include "nigiri/routing/limits.h"
#include "nigiri/routing/location_match_mode.h"
#include "nigiri/types.h"

namespace nigiri::routing {

struct offset : public footpath {
  offset(location_idx_t const l, duration_t const d, std::uint8_t const t)
      : footpath{l, d}, type_{t} {}
  std::uint8_t type_;
};

using start_time_t = std::variant<unixtime_t, interval<unixtime_t>>;

struct query {
  start_time_t start_time_;
  location_match_mode start_match_mode_{
      nigiri::routing::location_match_mode::kExact};
  location_match_mode dest_match_mode_{
      nigiri::routing::location_match_mode::kExact};
  bool use_start_footpaths_{true};
  std::vector<offset> start_;
  std::vector<offset> destination_;
  std::uint8_t max_transfers_{kMaxTransfers};
  unsigned min_connection_count_{0U};
  bool extend_interval_earlier_{false};
  bool extend_interval_later_{false};
};

}  // namespace nigiri::routing
