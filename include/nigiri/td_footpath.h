#pragma once

#include "cista/reflection/comparable.h"

#include "utl/cflow.h"
#include "utl/equal_ranges_linear.h"
#include "utl/pairwise.h"

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
  auto const r = to_range<SearchDir>(c);
  utl::equal_ranges_linear(
      begin(r), end(r),
      [](td_footpath const& a, td_footpath const& b) {
        return a.target_ == b.target_;
      },
      [&](auto&& from, auto&& to) {
        if constexpr (SearchDir == direction::kForward) {
          td_footpath const* pred = nullptr;

          auto const call = [&]() {
            auto const start = std::max(pred->valid_from_, t);
            auto const target_time = start + pred->duration_;
            auto const duration_with_waiting = target_time - t;
            if (duration_with_waiting < footpath::kMaxDuration) {
              f(footpath{from->target_, duration_with_waiting});
            }
          };

          for (auto it = from; it != to; ++it) {
            if (pred == nullptr || pred->duration_ == footpath::kMaxDuration ||
                it->valid_from_ < t) {
              pred = &*it;
            } else {
              call();
              return;
            }
          }

          if (pred != nullptr && pred->duration_ != footpath::kMaxDuration) {
            call();
          }
        } else /* (SearchDir == direction::kBackward) */ {
          auto const call = [&](unixtime_t const valid_from,
                                duration_t const duration) {
            auto const start = std::min(valid_from, t);
            auto const target_time = start - duration;
            auto const duration_with_waiting = t - target_time;
            if (duration_with_waiting < footpath::kMaxDuration) {
              f(footpath{from->target_, duration_with_waiting});
            }
          };

          if (from != to && from->valid_from_ <= t) {
            if (from->duration_ != footpath::kMaxDuration) {
              call(t, from->duration_);
              return;
            } else if (auto const next = std::next(from); next != to) {
              call(from->valid_from_, next->duration_);
              return;
            }
          }

          for (auto const [a, b] : utl::pairwise(to_range<SearchDir>(c))) {
            if (b.duration_ != footpath::kMaxDuration &&
                interval{b.valid_from_, a.valid_from_}.contains(t)) {
              call(a.valid_from_, b.duration_);
              return;
            }
          }
        }
      });
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