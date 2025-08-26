#pragma once

#include "nigiri/routing/tb/settings.h"
#include "nigiri/timetable.h"
#include "nigiri/types.h"

#define TRANSFERRED_FROM_NULL std::numeric_limits<std::uint32_t>::max()

namespace nigiri {
struct timetable;
}  // namespace nigiri

namespace nigiri::routing::tb {

using transport_segment_idx_t = std::uint32_t;

static constexpr transport_segment_idx_t day_offset_mask{
    0b1110'0000'0000'0000'0000'0000'0000'0000};

static constexpr transport_segment_idx_t transport_idx_mask{
    0b0001'1111'1111'1111'1111'1111'1111'1111};

static inline std::int32_t compute_day_offset(
    std::uint16_t const base, std::uint16_t const transport_day) {
  return static_cast<std::int32_t>(transport_day) -
         static_cast<std::int32_t>(base) + QUERY_DAY_SHIFT;
}

static inline transport_segment_idx_t embed_day_offset(
    std::uint16_t const base,
    std::uint16_t const transport_day,
    transport_idx_t const transport_idx) {
  auto const day_offset = compute_day_offset(base, transport_day);
  assert(0 <= day_offset && day_offset <= 7);
  return (static_cast<std::uint32_t>(day_offset) << (32U - DAY_OFFSET_BITS)) |
         transport_idx.v_;
}

static inline transport_segment_idx_t embed_day_offset(
    std::int32_t const day_offset, transport_idx_t const transport_idx) {
  assert(0 <= day_offset && day_offset <= 7);
  return (static_cast<std::uint32_t>(day_offset) << (32U - DAY_OFFSET_BITS)) |
         transport_idx.v_;
}

static inline std::uint32_t day_offset(
    transport_segment_idx_t const transport_segment_idx) {
  auto const day_offset =
      (transport_segment_idx & day_offset_mask) >> (32U - DAY_OFFSET_BITS);
  assert(0 <= day_offset && day_offset <= 7);
  return day_offset;
}

static inline day_idx_t transport_day(
    day_idx_t const base, transport_segment_idx_t const transport_segment_idx) {
  auto const transport_day =
      static_cast<std::int32_t>(base.v_) +
      static_cast<std::int32_t>(day_offset(transport_segment_idx)) -
      QUERY_DAY_SHIFT;
  assert(0 <= transport_day);
  return day_idx_t{transport_day};
}

static inline transport_idx_t transport_idx(
    transport_segment_idx_t const transport_segment_idx) {
  return transport_idx_t{transport_segment_idx & transport_idx_mask};
}

struct transport_segment {
  transport_segment(transport_segment_idx_t transport_segment_idx,
                    stop_idx_t stop_idx_start,
                    stop_idx_t stop_idx_end,
                    std::uint32_t transferred_from)
      : transport_segment_idx_(transport_segment_idx),
        stop_idx_start_(stop_idx_start),
        stop_idx_end_(stop_idx_end),
        transferred_from_(transferred_from) {}

  day_idx_t get_transport_day(day_idx_t const base) const {
    return transport_day(base, transport_segment_idx_);
  }

  transport_idx_t get_transport_idx() const {
    return transport_idx(transport_segment_idx_);
  }

  stop_idx_t get_stop_idx_start() const {
    return static_cast<stop_idx_t>(stop_idx_start_);
  }

  stop_idx_t get_stop_idx_end() const {
    return static_cast<stop_idx_t>(stop_idx_end_);
  }

  void print(std::ostream& out, timetable const& tt) const {
    auto const transferred_from_str = transferred_from_ == TRANSFERRED_FROM_NULL
                                          ? std::string{"INIT"}
                                          : std::to_string(transferred_from_);
    out << "transport " << get_transport_idx() << ": "
        << tt.transport_name(get_transport_idx()) << " from stop "
        << stop_idx_start_ << ": "
        << location{tt, stop{tt.route_location_seq_
                                 [tt.transport_route_[get_transport_idx()]]
                                 [stop_idx_start_]}
                            .location_idx()}
        << " to stop " << stop_idx_end_ << ": "
        << location{tt, stop{tt.route_location_seq_
                                 [tt.transport_route_[get_transport_idx()]]
                                 [stop_idx_end_]}
                            .location_idx()}
        << ", transferred_from: " << transferred_from_str << "\n";
  }

  segment_idx_t begin_, end_;

  // store day offset of the instance in upper bits of transport idx
  transport_segment_idx_t transport_segment_idx_;

  std::uint32_t stop_idx_start_ : STOP_IDX_BITS;
  std::uint32_t stop_idx_end_ : STOP_IDX_BITS;

  // queue index of the segment from which we transferred to this segment
  std::uint32_t transferred_from_;

  union {
    unixtime_t time_prune_;
    bool no_prune_;
  };
};

}  // namespace nigiri::routing::tb