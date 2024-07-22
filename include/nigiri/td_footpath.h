#pragma once

#include "cista/reflection/comparable.h"

#include "utl/cflow.h"

#include "nigiri/footpath.h"
#include "nigiri/types.h"

namespace nigiri {

constexpr auto const kNull = unixtime_t{0_minutes};
constexpr auto const kInfeasible =
    duration_t{std::numeric_limits<duration_t::rep>::max()};

struct td_footpath {
  CISTA_FRIEND_COMPARABLE(td_footpath)

  location_idx_t target_;
  unixtime_t valid_from_;
  duration_t duration_;
};

template <direction SearchDir, typename Collection, typename Fn>
void for_each_footpath(Collection const& c, unixtime_t const t, Fn&& f) {
  auto to = location_idx_t::invalid();
  auto pred = static_cast<td_footpath const*>(nullptr);
  auto const call = [&]() -> std::pair<bool, bool> {
    if (pred != nullptr && pred->duration_ != kInfeasible) {
      auto const start = SearchDir == direction::kForward
                             ? std::max(pred->valid_from_, t)
                             : std::min(pred->valid_from_, t);
      auto const target_time =
          start + (SearchDir == direction::kForward ? 1 : -1) * pred->duration_;
      auto const duration = SearchDir == direction::kForward
                                ? (target_time - t)
                                : (t - target_time);
      auto const fp = footpath{pred->target_, duration};
      auto const stop = f(fp) == utl::cflow::kBreak;
      return {true, stop};
    }
    return {false, false};
  };

  auto called = false;
  auto stop = false;
  for (auto const& fp : c) {
    if (!called && (fp.target_ != to || fp.valid_from_ > t)) {
      std::tie(called, stop) = call();
      if (stop) {
        return;
      }
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

template <typename Collection, typename Fn>
void for_each_footpath(direction const search_dir,
                       Collection const& c,
                       unixtime_t const t,
                       Fn&& f) {
  search_dir == direction::kForward
      ? for_each_footpath<direction::kForward>(c, t, std::forward<Fn>(f))
      : for_each_footpath<direction::kBackward>(c, t, std::forward<Fn>(f));
}

}  // namespace nigiri