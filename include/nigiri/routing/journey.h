#pragma once

#include <cinttypes>
#include <iosfwd>
#include <vector>

#include "nigiri/common/interval.h"
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
        unixtime_t const dep_time,
        unixtime_t const arr_time,
        T&& uses)
        : from_{d == direction::kForward ? a : b},
          to_{d == direction::kForward ? b : a},
          dep_time_{d == direction::kForward ? dep_time : arr_time},
          arr_time_{d == direction::kForward ? arr_time : dep_time},
          uses_{std::forward<T>(uses)} {}

    void print(std::ostream&, timetable const&, unsigned indent) const;

    location_idx_t from_, to_;
    unixtime_t dep_time_, arr_time_;
    variant<transport_enter_exit, footpath_idx_t, std::uint8_t> uses_;
  };

  bool dominates(journey const& o) const {
    return transfers_ <= o.transfers_ && start_time_ >= o.start_time_ &&
           dest_time_ <= o.dest_time_;
  }

  void add(leg&& l) { legs_.emplace_back(l); }

  void print(std::ostream&, timetable const&) const;

  std::vector<leg> legs_;
  unixtime_t start_time_;
  unixtime_t dest_time_;
  location_idx_t dest_;
  std::uint8_t transfers_{0U};
};

}  // namespace nigiri::routing
