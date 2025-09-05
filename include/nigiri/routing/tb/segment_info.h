#pragma once

#include <ostream>

#include "nigiri/routing/tb/tb_data.h"
#include "nigiri/timetable.h"

namespace nigiri::routing::tb {

struct segment_info {
  friend std::ostream& operator<<(std::ostream& out, segment_info const& i) {
    auto const [tt, tbd, s, day] = i;
    auto const t = tbd.segment_transports_[s];
    auto const first = tbd.transport_first_segment_[t];
    auto const rel = (s - first);
    auto const from = static_cast<stop_idx_t>(to_idx(rel));
    auto const to = static_cast<stop_idx_t>(from + 1U);
    auto const loc_seq = tt.route_location_seq_[tt.transport_route_[t]];

    out << "[dbg=" << tt.dbg(t) << ", trip=" << tt.transport_name(t)
        << ", segment=" << s << ", rel_segment=" << rel << "/"
        << tbd.get_segment_range(t).size()
        << ", from=" << location{tt, stop{loc_seq[from]}.location_idx()};

    if (day.has_value()) {
      out << " @ " << tt.event_time({t, *day}, from, event_type::kDep);
    } else {
      out << " @ " << tt.event_mam(t, from, event_type::kDep);
    }

    out << " => to=" << location{tt, stop{loc_seq[to]}.location_idx()};

    if (day.has_value()) {
      out << " @ " << tt.event_time({t, *day}, to, event_type::kArr);
    } else {
      out << " @ " << tt.event_mam(t, to, event_type::kArr);
    }

    return out << "]";
  }

  timetable const& tt_;
  tb_data const& tbd_;
  segment_idx_t s_;
  std::optional<day_idx_t> day_{};
};

}  // namespace nigiri::routing::tb

template <>
struct fmt::formatter<nigiri::routing::tb::segment_info> : ostream_formatter {};