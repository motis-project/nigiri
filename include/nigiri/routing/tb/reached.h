#pragma once

#include "nigiri/routing/raptor/debug.h"
#include "nigiri/routing/tb/settings.h"
#include "nigiri/routing/tb/tb_data.h"
#include "nigiri/timetable.h"

#define reached_dbg(...)
// #define reached_dbg fmt::println

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
        auto x = std::array<std::uint16_t, kMaxTransfers>{};
        x.fill(static_cast<std::uint16_t>(tbd_.get_segment_range(t).size()));
        earliest_segment_offset_[t].fill(x);
      }
    }
  }

  void update(transport_idx_t const t,
              query_day_idx_t const day_offset,
              std::uint16_t const to_segment_offset,
              std::uint8_t const k) {
    assert(day_offset >= 0 && day_offset < kTBMaxDayOffset);
    reached_dbg(
        "  reached update: k={}, dbg={}, trip={}, day={}, to_segment_offset={}",
        k, tt_.dbg(t), tt_.transport_name(t), day_offset, to_segment_offset);
    for (auto const x : tt_.route_transport_ranges_[tt_.transport_route_[t]]) {
      // Earlier trips are reachable on the following day.
      auto const off = x < t ? 1U : 0U;
      for (auto i = static_cast<std::uint32_t>(day_offset) + off;
           i != kTBMaxDayOffset; ++i) {
        for (auto j = k; j != kMaxTransfers; ++j) {
          if (earliest_segment_offset_[x][i][k] > to_segment_offset) {
            reached_dbg(
                "    -> update: k={}, dbg={}, trip={}, day={}, previous={} -> "
                "{}",
                k, tt_.dbg(x), tt_.transport_name(x), i,
                earliest_segment_offset_[x][i][k], to_segment_offset);
            earliest_segment_offset_[x][i][k] = to_segment_offset;
          }
        }
      }
    }
  }

  std::uint16_t query(transport_idx_t const t,
                      query_day_idx_t const day_offset,
                      std::uint8_t const k) {
    assert(day_offset >= 0 && day_offset < kTBMaxDayOffset);
    return earliest_segment_offset_[t][static_cast<std::uint32_t>(day_offset)]
                                   [k];
  }

  timetable const& tt_;
  tb_data const& tbd_;
  vector_map<
      transport_idx_t,
      std::array<std::array<std::uint16_t, kMaxTransfers>, kTBMaxDayOffset>>
      earliest_segment_offset_;
};

}  // namespace nigiri::routing::tb
