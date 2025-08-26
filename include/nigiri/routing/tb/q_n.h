#pragma once

#include "nigiri/routing/tb/reached.h"
#include "nigiri/routing/tb/settings.h"
#include "nigiri/types.h"

namespace nigiri::routing::tb {

using queue_idx_t = std::uint32_t;

struct queue_entry {
  segment_idx_t from_;
  segment_idx_t to_;
  queue_idx_t parent_ : 27;
  queue_idx_t query_day_offset_ : 5;
};

struct q_n {
  explicit q_n(reached& r) : r_(r) {}

  void reset(day_idx_t const new_base) {
    base_ = new_base;
    start_.clear();
    start_.emplace_back(0U);
    end_.clear();
    end_.emplace_back(0U);
    segments_.clear();
  }

  bool enqueue(std::uint16_t const transport_day,
               transport_idx_t const transport_idx,
               std::uint16_t const stop_idx,
               std::uint16_t const n_transfers,
               std::uint32_t const transferred_from) {
    assert(segments_.size() < std::numeric_limits<queue_idx_t>::max());
    assert(base_.has_value());

    auto const day_offset = compute_day_offset(base_->v_, transport_day);
    // query day has offset = 5, we disregard segments with offset > 6, since we
    // are only interested in journeys with max. 24h travel time
    if (0 <= day_offset && day_offset < 7) {

      // compute transport segment index
      auto const transport_segment_idx =
          embed_day_offset(day_offset, transport_idx);

      // look-up the earliest stop index reached
      auto const r_query_res = r_.query(transport_segment_idx, n_transfers);
      if (stop_idx < r_query_res) {

        // new n?
        if (n_transfers == start_.size()) {
          start_.emplace_back(segments_.size());
          end_.emplace_back(segments_.size());
        }

        // add transport segment
        segments_.emplace_back(transport_segment_idx, stop_idx, r_query_res,
                               transferred_from);

        // increment index
        ++end_[n_transfers];

        // update reached
        r_.update(transport_segment_idx, stop_idx, n_transfers);
        return true;
      }
    }
    return false;
  }

  auto& operator[](queue_idx_t pos) { return segments_[pos]; }

  auto size() const { return segments_.size(); }

  void print(std::ostream& out, queue_idx_t const q_idx) {
    out << "q_idx: " << std::to_string(q_idx) << ", segment of ";
    segments_[q_idx].print(out, r_.tt_);
  }

  reached& r_;
  std::optional<day_idx_t> base_ = std::nullopt;
  std::vector<queue_idx_t> start_;
  std::vector<queue_idx_t> end_;
  std::vector<transport_segment> segments_;
};

}  // namespace nigiri::routing::tb
