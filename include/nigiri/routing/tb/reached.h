#pragma once

#include "nigiri/routing/raptor/debug.h"
#include "nigiri/routing/tb/settings.h"
#include "nigiri/routing/tb/tb_data.h"
#include "nigiri/timetable.h"

namespace nigiri::routing::tb {

using query_day_idx_t = std::int32_t;

struct reached {
  explicit reached(timetable const& tt, tb_data const& tbd)
      : tt_{tt},
        tbd_{tbd},
        earliest_segment_offset_{to_idx(tt.next_transport_idx())} {}

  void reset() {
    for (auto r = route_idx_t{0}; r != tt_.n_routes(); ++r) {
      for (auto const t : tt_.route_transport_ranges_[r]) {
        earliest_segment_offset_[t].fill(
            static_cast<std::uint16_t>(tbd_.get_segment_range(t).size()));
      }
    }
  }

  void update(transport_idx_t const t,
              query_day_idx_t const day_offset,
              std::uint16_t const to_segment_offset) {
    assert(day_offset >= 0 && day_offset < kTBMaxDayOffset);
    trace("  reached update: transport={}, day_offset={}, to_segment_offset={}",
          t, day_offset, to_segment_offset);
    for (auto const x : tt_.route_transport_ranges_[tt_.transport_route_[t]]) {
      // Earlier trips are reachable on the following day.
      auto const off = x < t ? 1U : 0U;
      for (auto i = static_cast<std::uint32_t>(day_offset) + off;
           i != kTBMaxDayOffset; ++i) {
        if (earliest_segment_offset_[x][i] > to_segment_offset) {
          trace("    -> update: transport={}, day={}, previous={} -> {}", x, i,
                earliest_segment_offset_[x][i], to_segment_offset);
          earliest_segment_offset_[x][i] = to_segment_offset;
        }
      }
    }
  }

  std::uint16_t query(transport_idx_t const t,
                      query_day_idx_t const day_offset) {
    assert(day_offset >= 0 && day_offset < kTBMaxDayOffset);
    auto best = std::numeric_limits<std::uint16_t>::max();
    for (auto i = 0U; i <= static_cast<std::uint32_t>(day_offset); ++i) {
      if (earliest_segment_offset_[t][i] < best) {
        best = earliest_segment_offset_[t][i];
      }
    }
    return best;
  }

  timetable const& tt_;
  tb_data const& tbd_;
  vector_map<transport_idx_t, std::array<std::uint16_t, kTBMaxDayOffset>>
      earliest_segment_offset_;
};

}  // namespace nigiri::routing::tb
