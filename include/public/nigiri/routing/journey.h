#pragma once

#include <cinttypes>
#include <vector>

#include "nigiri/types.h"

namespace nigiri::routing {

struct journey {
  struct leg {
    location_idx_t from_, to_;
    unixtime_t dep_time_, arr_time_;
    variant<transport, footpath_idx_t, std::uint8_t> uses_;
  };

  bool dominates(journey const& o) {
    return transfers_ <= o.transfers_ && start_time_ >= o.start_time_ &&
           dest_time_ <= o.dest_time_;
  }

  void add(leg&& l) { legs_.emplace_back(l); }

  std::vector<leg> legs_;
  unixtime_t start_time_;
  unixtime_t dest_time_;
  location_idx_t dest_;
  std::uint8_t transfers_{0U};
};

}  // namespace nigiri::routing