#pragma once

#include <cstdint>
#include <span>

#include "nigiri/loader/transfer_rules.h"
#include "nigiri/loader/virt/signatures.h"
#include "nigiri/loader/virt/transfer_durations.h"
#include "nigiri/types.h"

namespace nigiri {
struct timetable;
}  // namespace nigiri

namespace nigiri::loader {

using route_stop_idx_t = cista::strong<std::uint32_t, struct route_stop_>;

struct location_route_stop {
  route_idx_t route_;
  stop_idx_t idx_;
  std::span<signature_idx_t const> sigs_;
  bool has_arr_;
  bool has_dep_;
  route_stop_idx_t route_stop_{route_stop_idx_t::invalid()};
};

using location_route_stops_t = vecvec<location_idx_t, location_route_stop>;

location_route_stops_t get_location_route_stops(
    timetable const&,
    signatures const&,
    hash_map<transfer_pair, transfer_durations> const& durations_of);

}  // namespace nigiri::loader
