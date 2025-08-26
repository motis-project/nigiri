#pragma once

#include "nigiri/timetable.h"

#include "nigiri/routing/tb/settings.h"

namespace nigiri::routing::tb {

using query_day_idx_t = std::int32_t;

inline query_day_idx_t compute_day_offset(day_idx_t const base,
                                          day_idx_t const transport_day) {
  return to_idx(transport_day) - to_idx(base);
}

struct reached {
  explicit reached(timetable const& tt)
      : tt_{tt}, earliest_stop_{to_idx(tt.next_transport_idx())} {}

  void reset() {
    for (auto r = route_idx_t{0}; r != tt_.n_routes(); ++r) {
      auto const n_stops = tt_.route_location_seq_[r].size();
      for (auto const t : tt_.route_transport_ranges_[r]) {
        earliest_stop_[t].fill(static_cast<stop_idx_t>(n_stops));
      }
    }
  }

  void update(transport_idx_t const t,
              query_day_idx_t const day_offset,
              stop_idx_t const stop_idx) {
    assert(day_offset >= 0 && day_offset < kTBMaxDayOffset);
    for (auto const x : tt_.route_transport_ranges_[tt_.transport_route_[t]]) {
      // Earlier trips are reachable on the following day.
      auto const off = x < t ? 1U : 0U;
      for (auto i = static_cast<std::uint32_t>(day_offset) + off;
           i != kTBMaxDayOffset; ++i) {
        if (earliest_stop_[x][i] > stop_idx) {
          earliest_stop_[x][i] = stop_idx;
        }
      }
    }
  }

  stop_idx_t query(transport_idx_t const t, query_day_idx_t const day_offset) {
    assert(day_offset >= 0 && day_offset < kTBMaxDayOffset);
    auto best = std::numeric_limits<stop_idx_t>::max();
    for (auto i = 0U; i <= static_cast<std::uint32_t>(day_offset); ++i) {
      if (earliest_stop_[t][i] < best) {
        best = earliest_stop_[t][i];
      }
    }
    return best;
  }

  timetable const& tt_;
  vector_map<transport_idx_t, std::array<stop_idx_t, kTBMaxDayOffset>>
      earliest_stop_;
};

}  // namespace nigiri::routing::tb
