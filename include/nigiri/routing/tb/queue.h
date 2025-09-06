#pragma once

#include <ranges>

#include "fmt/ranges.h"

#include "nigiri/routing/tb/reached.h"
#include "nigiri/routing/tb/segment_info.h"
#include "nigiri/routing/tb/settings.h"
#include "nigiri/routing/tb/tb_data.h"
#include "nigiri/types.h"

#define tb_queue_dbg(...)
// #define tb_queue_dbg fmt::println

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
        q_[parent].transport_query_day_offset_ + transfer.get_day_offset();

    if (day_offset <= 0 || day_offset >= kTBMaxDayOffset) {
      tb_queue_dbg("  day_offset out of range: {}", day_offset);
      return false;
    }

    auto const min_segment_offset =
        r_.query(transfer.route_, transfer.transport_offset_,
                 static_cast<query_day_offset_t>(day_offset), k);
    if (transfer.to_segment_offset_ >= min_segment_offset) {
      tb_queue_dbg(
          "  already reached: transfer.route={}, transfer.to_segment={} >= "
          "{}=min_segment, reached_segments={}",
          transfer.route_, transfer.to_segment_, min_segment_offset,
          r_.data_[transfer.route_] |
              std::views::transform([&](entry const& e) {
                auto const t =
                    r_.tt_.route_transport_ranges_[transfer.route_].from_ +
                    get_transport_offset(e.transport_);
                auto const segment =
                    r_.tbd_.get_segment_range(t).from_ + e.segment_offset_;
                auto const day = base_ + get_query_day(e.transport_);
                return std::pair{e.k_,
                                 segment_info{r_.tt_, r_.tbd_, segment, day}};
              }));
      return false;
    }

    q_.push_back(queue_entry{
        .segment_range_ = {transfer.to_segment_,
                           transfer.to_segment_ - transfer.to_segment_offset_ +
                               min_segment_offset},
        .parent_ = parent,
        .transport_query_day_offset_ = static_cast<queue_idx_t>(day_offset)});
    r_.update(transfer.route_, transfer.transport_offset_,
              transfer.to_segment_offset_,
              static_cast<query_day_offset_t>(day_offset), k);

    return true;
  }

  void initial_enqueue([[maybe_unused]] tb_data const& tbd,
                       segment_idx_t const transport_first_segment,
                       segment_idx_t const segment,
                       route_idx_t const r,
                       transport_idx_t const t,
                       query_day_offset_t const query_day_offset,
                       [[maybe_unused]] day_idx_t const day) {
    auto const transport_offset = static_cast<std::uint16_t>(
        to_idx(t - r_.tt_.route_transport_ranges_[r].from_));
    auto const min_segment_offset =
        r_.query(r, transport_offset, query_day_offset, 0U);
    auto const segment_offset =
        static_cast<std::uint16_t>(to_idx(segment - transport_first_segment));
    if (segment_offset >= min_segment_offset) {
      tb_queue_dbg(
          "  initial_enqueue already "
          "reached:\n\tsegment_offset={}\n\tmin_segment_offset={}",
          seg(transport_first_segment + segment_offset, day),
          seg(transport_first_segment + min_segment_offset, day));
      return;
    }

    assert(tbd.get_segment_range(t).contains(
        interval{segment, transport_first_segment + min_segment_offset}));
    q_.push_back(queue_entry{
        .segment_range_ = {segment,
                           transport_first_segment + min_segment_offset},
        .parent_ = queue_entry::kNoParent,
        .transport_query_day_offset_ = query_day_offset});
    r_.update(r, transport_offset, segment_offset, query_day_offset, 0);
    tb_queue_dbg("  initial enqueue:\n\t\t{}\n\t\t{}", seg(segment, day),
                 seg(transport_first_segment + min_segment_offset - 1, day));
  }

  queue_entry& operator[](queue_idx_t const pos) { return q_[pos]; }

  std::size_t size() const { return q_.size(); }

  segment_info seg(segment_idx_t const s, day_idx_t const day) const {
    return {r_.tt_, r_.tbd_, s, day};
  }

  day_idx_t base_;
  reached& r_;
  std::vector<queue_entry> q_;
};

}  // namespace nigiri::routing::tb
