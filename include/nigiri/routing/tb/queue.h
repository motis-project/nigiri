#pragma once

#include "nigiri/routing/tb/reached.h"
#include "nigiri/routing/tb/settings.h"
#include "nigiri/routing/tb/tb_data.h"
#include "nigiri/types.h"

#define tb_queue_dbg(...)

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

  bool enqueue(transfer const& transfer,
               queue_idx_t const parent,
               std::uint8_t const k) {
    auto const day_offset =
        q_[parent].transport_query_day_offset_ + transfer.day_offset_;

    if (day_offset <= 0 || day_offset >= kTBMaxDayOffset) {
      tb_queue_dbg("  day_offset out of range: {}", day_offset);
      return false;
    }

    auto const min_segment_offset =
        r_.query(transfer.to_transport_, day_offset, k);
    if (transfer.to_segment_offset_ >= min_segment_offset) {
      tb_queue_dbg(
          "  already reached: transfer.to_segment={} >= {}=min_segment",
          transfer.to_segment_, min_segment_offset);
      return false;
    }

    q_.push_back(queue_entry{
        .segment_range_ = {transfer.to_segment_,
                           transfer.to_segment_ - transfer.to_segment_offset_ +
                               min_segment_offset},
        .parent_ = parent,
        .transport_query_day_offset_ = static_cast<queue_idx_t>(day_offset)});
    r_.update(transfer.to_transport_, day_offset, transfer.to_segment_offset_,
              k);

    return true;
  }

  void initial_enqueue([[maybe_unused]] tb_data const& tbd,
                       segment_idx_t const transport_first_segment,
                       segment_idx_t const segment,
                       transport_idx_t const t,
                       std::int8_t const day_offset) {
    if (day_offset < 0 || day_offset >= kTBMaxDayOffset) {
      tb_queue_dbg("  initial_enqueue day_offset out of range: {}", day_offset);
      return;
    }

    auto const min_segment_offset = r_.query(t, day_offset, 0);
    auto const segment_offset =
        static_cast<std::uint16_t>(to_idx(segment - transport_first_segment));
    if (segment_offset >= min_segment_offset) {
      return;
    }

    assert(tbd.get_segment_range(t).contains(
        interval{segment, transport_first_segment + min_segment_offset}));
    q_.push_back(queue_entry{
        .segment_range_ = {segment,
                           transport_first_segment + min_segment_offset},
        .parent_ = queue_entry::kNoParent,
        .transport_query_day_offset_ = static_cast<queue_idx_t>(day_offset)});
    r_.update(t, day_offset, segment_offset, 0);
  }

  queue_entry& operator[](queue_idx_t const pos) { return q_[pos]; }

  std::size_t size() const { return q_.size(); }

  reached& r_;
  std::vector<queue_entry> q_;
};

}  // namespace nigiri::routing::tb
