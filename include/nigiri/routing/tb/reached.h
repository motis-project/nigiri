#pragma once

#include "nigiri/routing/tb/settings.h"
#include "nigiri/routing/tb/tb_data.h"
#include "nigiri/timetable.h"

namespace nigiri::routing::tb {

using query_day_idx_t = std::int32_t;

struct reached {
  explicit reached(timetable const& tt)
      : tt_{tt}, earliest_segment_{to_idx(tt.next_transport_idx())} {}

  void reset() {
    for (auto r = route_idx_t{0}; r != tt_.n_routes(); ++r) {
      auto const n_segments = tt_.route_location_seq_[r].size() - 1U;
      for (auto const t : tt_.route_transport_ranges_[r]) {
        earliest_segment_[t].fill(static_cast<segment_idx_t>(n_segments));
      }
    }
  }

  void update(transport_idx_t const t,
              query_day_idx_t const day_offset,
              segment_idx_t const s) {
    assert(day_offset >= 0 && day_offset < kTBMaxDayOffset);
    for (auto const x : tt_.route_transport_ranges_[tt_.transport_route_[t]]) {
      // Earlier trips are reachable on the following day.
      auto const off = x < t ? 1U : 0U;
      for (auto i = static_cast<std::uint32_t>(day_offset) + off;
           i != kTBMaxDayOffset; ++i) {
        if (earliest_segment_[x][i] > s) {
          earliest_segment_[x][i] = s;
        }
      }
    }
  }

  segment_idx_t query(transport_idx_t const t,
                      query_day_idx_t const day_offset) {
    assert(day_offset >= 0 && day_offset < kTBMaxDayOffset);
    auto best = std::numeric_limits<segment_idx_t>::max();
    for (auto i = 0U; i <= static_cast<std::uint32_t>(day_offset); ++i) {
      if (earliest_segment_[t][i] < best) {
        best = earliest_segment_[t][i];
      }
    }
    return best;
  }

  timetable const& tt_;
  vector_map<transport_idx_t, std::array<segment_idx_t, kTBMaxDayOffset>>
      earliest_segment_;
};

}  // namespace nigiri::routing::tb
