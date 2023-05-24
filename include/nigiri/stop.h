#pragma once

#include "nigiri/types.h"

namespace nigiri {

struct stop {
  using value_type = location_idx_t::value_t;

  stop(location_idx_t::value_t const val) {
    std::memcpy(this, &val, sizeof(value_type));
  }

  stop(location_idx_t const location,
       bool const in_allowed,
       bool const out_allowed,
       bool const in_seated_transfer,
       bool const out_seated_transfer)
      : location_{location},
        has_in_seated_transfer_{in_seated_transfer},
        has_out_seated_transfer_{out_seated_transfer},
        in_allowed_{in_allowed ? 1U : 0U},
        out_allowed_{out_allowed ? 1U : 0U} {}

  location_idx_t location_idx() const { return location_idx_t{location_}; }
  bool in_allowed() const { return in_allowed_ != 0U; }
  bool out_allowed() const { return out_allowed_ != 0U; }
  bool out_seated_transfer() const { return has_out_seated_transfer_ != 0U; }
  bool in_seated_transfer() const { return has_in_seated_transfer_ != 0U; }

  cista::hash_t hash() const {
    return cista::hash_combine(cista::BASE_HASH, value());
  }

  location_idx_t::value_t value() const {
    return *reinterpret_cast<location_idx_t::value_t const*>(this);
  }

  friend auto operator<=>(stop const&, stop const&) = default;

  location_idx_t::value_t location_ : 28;
  location_idx_t::value_t has_in_seated_transfer_ : 1;
  location_idx_t::value_t has_out_seated_transfer_ : 1;
  location_idx_t::value_t in_allowed_ : 1;
  location_idx_t::value_t out_allowed_ : 1;
};

static_assert(sizeof(stop) == sizeof(location_idx_t));

}  // namespace nigiri