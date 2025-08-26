#pragma once

#include "nigiri/routing/pareto_set.h"
#include "nigiri/routing/tb/settings.h"
#include "nigiri/routing/tb/transport_segment.h"
#include "nigiri/timetable.h"
#include "nigiri/types.h"

namespace nigiri::routing::tb {

struct reached_entry {
  bool dominates(reached_entry const& o) const {
    return transport_segment_idx_ <= o.transport_segment_idx_ &&
           stop_idx_ <= o.stop_idx_ && n_transfers_ <= o.n_transfers_;
  }
  transport_segment_idx_t transport_segment_idx_;
  std::uint16_t stop_idx_;
  std::uint16_t n_transfers_;
};

struct reached {
  reached() = delete;
  explicit reached(timetable const& tt) : tt_(tt) {
    data_.resize(tt.n_routes());
  }

  void reset();

  void update(transport_segment_idx_t const transport_segment_idx,
              std::uint16_t const stop_idx,
              std::uint16_t const n_transfers) {
    data_[tt_.transport_route_[transport_idx(transport_segment_idx)].v_].add(
        reached_entry{transport_segment_idx, stop_idx, n_transfers});
  }

  std::uint16_t query(transport_segment_idx_t const transport_segment_idx,
                      std::uint16_t const n_transfers) {
    auto const route_idx =
        tt_.transport_route_[transport_idx(transport_segment_idx)];

    auto stop_idx_min =
        static_cast<uint16_t>(tt_.route_location_seq_[route_idx].size() - 1);
    // find minimal stop index among relevant entries
    for (auto const& re : data_[route_idx.v_]) {
      // only entries with less or equal n_transfers and less or equal
      // transport_segment_idx are relevant
      if (re.n_transfers_ <= n_transfers &&
          re.transport_segment_idx_ <= transport_segment_idx &&
          re.stop_idx_ < stop_idx_min) {
        stop_idx_min = re.stop_idx_;
      }
    }

    return stop_idx_min;
  }

  timetable const& tt_;

  // reached stops per route
  std::vector<pareto_set<reached_entry>> data_;
};

}  // namespace nigiri::routing::tb
