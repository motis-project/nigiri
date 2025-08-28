#pragma once

#include "nigiri/routing/tb/reached.h"
#include "nigiri/routing/tb/settings.h"
#include "nigiri/routing/tb/tb_data.h"
#include "nigiri/types.h"

namespace nigiri::routing::tb {

using queue_idx_t = std::uint32_t;

struct queue_entry {
  static constexpr auto const kQueryDayOffsetBits = 5;
  static constexpr auto const kParentBits =
      sizeof(queue_idx_t) * 8U - kQueryDayOffsetBits;
  static constexpr auto const kNoParent = static_cast<queue_idx_t>(
      std::numeric_limits<queue_idx_t>::max() >> kQueryDayOffsetBits);

  static_assert(1 << kQueryDayOffsetBits >= kTBMaxDayOffset);

  interval<segment_idx_t> segment_range_;
  queue_idx_t parent_ : 27;
  queue_idx_t transport_query_day_offset_ : kQueryDayOffsetBits;
};

struct queue {
  explicit queue(reached& r) : r_(r) {}

  void reset() { q_.clear(); }

  bool enqueue(transfer const& transfer, queue_idx_t const parent) {
    auto const day_offset =
        q_[parent].transport_query_day_offset_ + transfer.day_offset_;

    if (day_offset <= 0 || day_offset >= kTBMaxDayOffset) {
      return false;
    }

    auto const min_stop_idx = r_.query(transfer.to_transport_, day_offset);
    if (transfer.to_stop_idx_ >= min_stop_idx) {
      return false;
    }

    q_.push_back(queue_entry{
        .segment_range_ = {transfer.to_segment_,
                           transfer.to_segment_ +
                               (min_stop_idx - transfer.to_stop_idx_)},
        .parent_ = parent,
        .transport_query_day_offset_ = static_cast<queue_idx_t>(day_offset)});
    r_.update(transfer.to_transport_, day_offset, transfer.to_stop_idx_);

    return true;
  }

  void initial_enqueue(segment_idx_t const segment,
                       transport_idx_t const t,
                       stop_idx_t const stop_idx,
                       std::int8_t const day_offset) {
    auto const min_stop_idx = r_.query(t, day_offset);
    if (stop_idx >= min_stop_idx) {
      return;
    }

    q_.push_back(queue_entry{
        .segment_range_ = {segment, segment + (min_stop_idx - stop_idx)},
        .parent_ = queue_entry::kNoParent,
        .transport_query_day_offset_ = static_cast<queue_idx_t>(day_offset)});
    r_.update(t, day_offset, stop_idx);
  }

  queue_entry& operator[](queue_idx_t const pos) { return q_[pos]; }

  std::size_t size() const { return q_.size(); }

  reached& r_;
  std::vector<queue_entry> q_;
};

}  // namespace nigiri::routing::tb
