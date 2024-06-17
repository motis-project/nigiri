#pragma once

#include "nigiri/types.h"

namespace nigiri {

constexpr auto const kNull = unixtime_t{0_minutes};
constexpr auto const kInfeasible =
    duration_t{std::numeric_limits<duration_t::rep>::max()};

struct td_footpath {
  location_idx_t target_;
  unixtime_t valid_from_;
  duration_t duration_;
};

template <typename Collection, typename Fn>
void for_each_footpath(Collection const& c, unixtime_t const t, Fn&& f) {
  auto to = location_idx_t::invalid();
  auto pred = static_cast<td_footpath const*>(nullptr);
  auto const call = [&]() {
    if (pred != nullptr && pred->duration_ != kInfeasible) {
      f(std::max(pred->valid_from_, t) + pred->duration_, to);
      return true;
    }
    return false;
  };

  auto called = false;
  for (auto const& fp : c) {
    if (!called && (fp.target_ != to || fp.valid_from_ > t)) {
      called = call();
    }

    if (fp.target_ != to) {
      called = false;
    }
    to = fp.target_;
    pred = &fp;
  }

  if (!called) {
    call();
  }
}

}  // namespace nigiri