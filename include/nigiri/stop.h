#pragma once

#include <compare>

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
       bool const in_allowed_wheelchair,
       bool const out_allowed_wheelchair)
      : location_{location},
        in_allowed_{in_allowed ? 1U : 0U},
        out_allowed_{out_allowed ? 1U : 0U},
        in_allowed_wheelchair_{in_allowed_wheelchair ? 1U : 0U},
        out_allowed_wheelchair_{out_allowed_wheelchair ? 1U : 0U} {}

  location_idx_t location_idx() const { return location_idx_t{location_}; }
  bool in_allowed_wheelchair() const { return in_allowed_wheelchair_ != 0U; }
  bool out_allowed_wheelchair() const { return out_allowed_wheelchair_ != 0U; }
  bool in_allowed() const { return in_allowed_ != 0U; }
  bool out_allowed() const { return out_allowed_ != 0U; }
  bool is_cancelled() const { return !in_allowed() && !out_allowed(); }

  bool in_allowed(profile_idx_t const p) const {
    return p == 2U ? in_allowed_wheelchair() : in_allowed();
  }
  bool out_allowed(profile_idx_t const p) const {
    return p == 2U ? out_allowed_wheelchair() : out_allowed();
  }

  template <direction SearchDir>
  bool can_start(bool const is_wheelchair) const {
    if constexpr (SearchDir == direction::kForward) {
      return is_wheelchair ? in_allowed_wheelchair() : in_allowed();
    } else {
      return is_wheelchair ? out_allowed_wheelchair() : out_allowed();
    }
  }

  template <direction SearchDir>
  bool can_finish(bool const is_wheelchair) const {
    if constexpr (SearchDir == direction::kForward) {
      return is_wheelchair ? out_allowed_wheelchair() : out_allowed();
    } else {
      return is_wheelchair ? in_allowed_wheelchair() : in_allowed();
    }
  }

  cista::hash_t hash() const {
    return cista::hash_combine(cista::BASE_HASH, value());
  }

  location_idx_t::value_t value() const {
    return *reinterpret_cast<location_idx_t::value_t const*>(this);
  }

  friend auto operator<=>(stop const&, stop const&) = default;

  location_idx_t::value_t location_ : 28;
  location_idx_t::value_t in_allowed_ : 1;
  location_idx_t::value_t out_allowed_ : 1;
  location_idx_t::value_t in_allowed_wheelchair_ : 1;
  location_idx_t::value_t out_allowed_wheelchair_ : 1;
};

static_assert(sizeof(stop) == sizeof(location_idx_t));

}  // namespace nigiri