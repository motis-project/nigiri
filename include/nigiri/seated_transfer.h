#pragma once

#include "nigiri/types.h"

namespace nigiri {

struct seated_transfer {
  using value_t = route_idx_t;

  seated_transfer(value_t const val) {
    std::memcpy(this, &val, sizeof(value_t));
  }
  seated_transfer(route_idx_t const target, std::int8_t const day_offset)
      : target_{to_idx(target)},
        day_offset_{static_cast<route_idx_t::value_t>(day_offset)} {}

  route_idx_t::value_t value() const noexcept {
    return *reinterpret_cast<location_idx_t::value_t const*>(this);
  }

  cista::hash_t hash() const {
    return cista::hash_combine(cista::BASE_HASH, value());
  }

  std::uint8_t day_offset() const noexcept { return day_offset_; }
  route_idx_t target() const noexcept { return route_idx_t{target_}; }

  route_idx_t::value_t target_ : 24;
  route_idx_t::value_t day_offset_ : 8;
};

static_assert(sizeof(seated_transfer) == sizeof(route_idx_t));

}  // namespace nigiri