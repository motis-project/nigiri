#pragma once

#include "nigiri/types.h"

namespace nigiri::routing {

struct offset {
  location_idx_t location_;
  duration_t offset_;
  std::uint8_t type_;
};

struct query {
  interval<unixtime_t> interval_;
  vector<offset> start_;
  vector<vector<offset>> destinations_;
  vector<vector<offset>> via_destinations_;
  cista::bitset<kNumClasses> allowed_classes_;
  std::uint8_t max_transfers_;
  std::uint8_t min_connection_count_;
  bool extend_interval_earlier_;
  bool extend_interval_later_;
};

}  // namespace nigiri::routing
