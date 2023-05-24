#pragma once

#include "nigiri/types.h"

namespace nigiri {

struct seated_transfer {
  using value_t = transport_idx_t;

  seated_transfer(value_t const val) {
    std::memcpy(this, &val, sizeof(value_t));
  }
  seated_transfer(transport_idx_t const target, std::uint8_t const day_offset)
      : target_{to_idx(target)}, day_offset_{day_offset} {}

  transport_idx_t::value_t value() const noexcept {
    return *reinterpret_cast<location_idx_t::value_t const*>(this);
  }

  std::uint8_t day_offset() const noexcept { return day_offset_; }
  transport_idx_t target() const noexcept { return transport_idx_t{target_}; }

  transport_idx_t::value_t target_ : 24;
  transport_idx_t::value_t day_offset_ : 8;
};

}  // namespace nigiri