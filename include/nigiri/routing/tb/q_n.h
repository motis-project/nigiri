#pragma once

#include "nigiri/routing/tb/reached.h"
#include "nigiri/routing/tb/settings.h"
#include "nigiri/types.h"

namespace nigiri::routing::tb {

using queue_idx_t = std::uint32_t;

struct queue_entry {
  static constexpr auto const kQueryDayOffsetBits = 5;
  static_assert(1 << kQueryDayOffsetBits >= kTBMaxDayOffset);

  interval<segment_idx_t> stop_range_;
  queue_idx_t parent_ : 27;
  queue_idx_t query_day_offset_ : kQueryDayOffsetBits;
};

struct q_n {
  explicit q_n(reached& r) : r_(r) {}

  void reset(day_idx_t const new_base) {
    base_ = new_base;
    segments_.clear();
  }

  bool enqueue(segment_idx_t const from,
               stop_idx_t const stop_idx,
               queue_idx_t const parent) {
    auto const day_offset = compute_day_offset(base_, tr.day_);
    if (0 <= day_offset && day_offset < kTBMaxDayOffset) {
      // look-up the earliest stop index reached
      auto const min_stop_idx = r_.query(tr.t_idx_, day_offset);
      if (stop_idx < min_stop_idx) {
        segments_.push_back({

        });
        r_.update(transport_segment_idx, stop_idx, n_transfers);
        return true;
      }
    }
    return false;
  }

  auto& operator[](queue_idx_t const pos) { return segments_[pos]; }

  auto size() const { return segments_.size(); }

  reached& r_;
  day_idx_t base_;
  std::vector<queue_entry> segments_;
};

}  // namespace nigiri::routing::tb
