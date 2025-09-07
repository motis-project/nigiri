#pragma once

#include "nigiri/routing/raptor/debug.h"
#include "nigiri/routing/tb/segment_info.h"
#include "nigiri/routing/tb/settings.h"
#include "nigiri/routing/tb/tb_data.h"
#include "nigiri/timetable.h"

#define reached_dbg(...)
// #define reached_dbg fmt::println

namespace nigiri::routing::tb {

using query_day_offset_t = std::uint16_t;

using transport_t = std::uint16_t;

inline transport_t to_transport(std::uint16_t const transport_offset,
                                query_day_offset_t const query_day_offset) {
  assert((0xF000 & transport_offset) == 0);
  assert((0xFFF0 & query_day_offset) == 0);
  return static_cast<std::uint16_t>(query_day_offset << 12U) | transport_offset;
}

inline std::uint16_t get_transport_offset(transport_t const x) {
  return x & 0x0FFF;
}

inline std::uint16_t get_query_day(transport_t const x) { return x >> 12U; }

struct entry {
  inline bool dominates(entry const& o) const {
    return k_ <= o.k_ && transport_ <= o.transport_ &&
           segment_offset_ <= o.segment_offset_;
  }

  transport_t transport_;
  std::uint16_t segment_offset_ : 12;
  std::uint16_t k_ : 4;
};

struct reached {
  explicit reached(timetable const& tt, tb_data const& tbd)
      : tt_{tt}, tbd_{tbd}, data_{tt.n_routes()} {
    for (auto& x : data_) {
      x.els_.reserve(8);
    }
  }

  void reset() {
    for (auto r = route_idx_t{0U}; r != tt_.n_routes(); ++r) {
      data_[r].clear();
    }
  }

  void update(route_idx_t const r,
              std::uint16_t const transport_offset,
              std::uint16_t const segment_offset,
              query_day_offset_t const query_day_offset,
              std::uint8_t const k,
              std::uint64_t& max_size) {
    assert(query_day_offset >= 0 && query_day_offset < kTBMaxDayOffset);
    reached_dbg(
        "  reached update: k={}, r={}, dbg={}, trip={}, day={}, "
        "to_segment_offset={}",
        k, r, tt_.dbg(tt_.route_transport_ranges_[r][transport_offset]),
        tt_.transport_name(tt_.route_transport_ranges_[r][transport_offset]),
        query_day_offset, segment_offset);
    auto const transport = to_transport(transport_offset, query_day_offset);
    data_[r].add(
        {.transport_ = transport, .segment_offset_ = segment_offset, .k_ = k});

    if (data_[r].size() > max_size) {
      max_size = data_[r].size();
    }
  }

  std::uint16_t query(route_idx_t const r,
                      std::uint16_t const transport_offset,
                      query_day_offset_t const query_day_offset,
                      std::uint8_t const k) {
    auto const transport = to_transport(transport_offset, query_day_offset);

    auto min_segment =
        static_cast<std::uint16_t>(tt_.route_location_seq_[r].size() - 1);
    for (auto const& re : data_[r]) {
      if (re.k_ <= k && re.transport_ <= transport &&
          re.segment_offset_ < min_segment) {
        min_segment = re.segment_offset_;
      }
    }

    return min_segment;
  }

  std::string to_str(day_idx_t const base, route_idx_t const r) const {
    return fmt::format(
        "route[{}]={}", r,
        data_[r] | std::views::transform([&](entry const& e) {
          auto const t = tt_.route_transport_ranges_[r].from_ +
                         get_transport_offset(e.transport_);
          auto const segment =
              tbd_.get_segment_range(t).from_ + e.segment_offset_;
          auto const day = base + get_query_day(e.transport_);
          return std::pair{e.k_, segment_info{tt_, tbd_, segment, day}};
        }));
  }

  timetable const& tt_;
  tb_data const& tbd_;
  vector_map<route_idx_t, pareto_set<entry>> data_;
};

}  // namespace nigiri::routing::tb
