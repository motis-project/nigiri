#pragma once

#include <cinttypes>
#include <iosfwd>
#include <variant>
#include <vector>

#include "nigiri/common/interval.h"
#include "nigiri/footpath.h"
#include "nigiri/routing/query.h"
#include "nigiri/types.h"

namespace nigiri {
struct timetable;
}

namespace nigiri::routing {

struct journey {
  struct transport_enter_exit {
    transport_enter_exit(transport const t, unsigned const a, unsigned const b)
        : t_{t}, stop_range_{std::min(a, b), std::max(a, b) + 1} {}
    transport t_;
    interval<unsigned> stop_range_;
  };

  struct leg {
    template <typename T>
    leg(direction const d,
        location_idx_t const a,
        location_idx_t const b,
        unixtime_t const tima_at_a,
        unixtime_t const time_at_b,
        T&& uses)
        : from_{d == direction::kForward ? a : b},
          to_{d == direction::kForward ? b : a},
          dep_time_{d == direction::kForward ? tima_at_a : time_at_b},
          arr_time_{d == direction::kForward ? time_at_b : tima_at_a},
          uses_{std::forward<T>(uses)} {}

    void print(std::ostream&,
               timetable const&,
               unsigned n_indent = 0U,
               bool debug = false) const;

    location_idx_t from_, to_;
    unixtime_t dep_time_, arr_time_;
    std::variant<transport_enter_exit, footpath, offset> uses_;
  };

  bool dominates(journey const& o) const {
    if (start_time_ <= dest_time_) {
      return transfers_ <= o.transfers_ && start_time_ >= o.start_time_ &&
             dest_time_ <= o.dest_time_;
    } else {
      return transfers_ <= o.transfers_ && start_time_ <= o.start_time_ &&
             dest_time_ >= o.dest_time_;
    }
  }

  void add(leg&& l) { legs_.emplace_back(l); }

  duration_t travel_time() const { return dest_time_ - start_time_; }

  void print(std::ostream&, timetable const&, bool debug = false) const;

  std::vector<leg> legs_;
  unixtime_t start_time_;
  unixtime_t dest_time_;
  location_idx_t dest_;
  std::uint8_t transfers_{0U};
};

}  // namespace nigiri::routing
