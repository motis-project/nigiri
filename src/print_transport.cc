#include "nigiri/print_transport.h"

#include <cassert>

#include "utl/enumerate.h"
#include "utl/zip.h"

#include "date/date.h"

#include "nigiri/common/indent.h"
#include "nigiri/lookup/get_transport_stop_tz.h"
#include "nigiri/rt/rt_timetable.h"
#include "nigiri/timetable.h"

namespace nigiri {

void print_transport(timetable const& tt,
                     rt_timetable const* rtt,
                     std::ostream& out,
                     transport const x,
                     interval<stop_idx_t> stop_range,
                     unsigned const indent_width,
                     bool const with_debug) {
  auto const i = x.t_idx_;
  auto const day_idx = x.day_;
  auto const& route_idx = tt.transport_route_.at(i);
  auto const& stop_seq = tt.route_location_seq_.at(route_idx);

  auto rt_t = rt_transport_idx_t::invalid();
  if (rtt != nullptr) {
    auto const it = rtt->static_trip_lookup_.find(x);
    if (it != end(rtt->static_trip_lookup_)) {
      rt_t = it->second;
      auto const rt_stop_seq = rtt->rt_transport_location_seq_[rt_t];
      auto const rt_times = rtt->rt_transport_stop_times_[rt_t];
      utl::verify(utl::equal(rt_stop_seq, stop_seq),
                  "print_transport: rerouting not supported");
      utl::verify(rt_times.size() == rt_stop_seq.size() * 2U - 2U,
                  "print_transport: bad rt times {} vs {}", rt_times.size(),
                  rt_stop_seq.size());
    }
  }

  assert(tt.route_transport_ranges_.at(route_idx).contains(x.t_idx_));

  auto const from =
      std::min(static_cast<stop_idx_t>(stop_seq.size()), stop_range.from_);
  auto const to =
      std::min(static_cast<stop_idx_t>(stop_seq.size()), stop_range.to_);
  for (auto stop_idx = from; stop_idx != to; ++stop_idx) {
    auto const s = stop{stop_seq.at(stop_idx)};
    auto const location_idx = s.location_idx();
    auto const& stop_id = tt.locations_.ids_.at(location_idx).view();
    auto const& parent = tt.locations_.parents_.at(location_idx);
    auto const stop_name = parent == location_idx_t::invalid()
                               ? tt.locations_.names_.at(location_idx).view()
                               : tt.locations_.names_.at(parent).view();

    auto const& tz = tt.locations_.timezones_.at(
        get_transport_stop_tz(tt, i, s.location_idx()));
    indent(out, indent_width);
    fmt::print(out, "{:2}: {:7} {:.<48} ", stop_idx, stop_id, stop_name);

    if (stop_idx != from) {
      if (!s.out_allowed()) {
        fmt::print(" -");
      } else {
        fmt::print("  ");
      }
      auto const t = tt.event_time(x, stop_idx, event_type::kArr);
      date::to_stream(out, "a: %d.%m %R", t);
      date::to_stream(out, " [%d.%m %R]", to_local_time(tz, t));
      if (rt_t != rt_transport_idx_t::invalid()) {
        auto const rt = rtt->unix_event_time(rt_t, stop_idx, event_type::kArr);
        date::to_stream(out, " RT %d.%m %R", rt);
        date::to_stream(out, " [%d.%m %R]", to_local_time(tz, rt));
      }
    } else {
      out << "              ";
      out << "              ";
      if (rtt != nullptr) {
        out << "                 ";
        out << "              ";
      }
    }

    if (stop_idx != to - 1) {
      if (!s.in_allowed()) {
        fmt::print("   -");
      } else {
        fmt::print("    ");
      }

      auto const t = tt.event_time(x, stop_idx, event_type::kDep);
      date::to_stream(out, "  d: %d.%m %R", t);
      date::to_stream(out, " [%d.%m %R]", to_local_time(tz, t));
      if (rt_t != rt_transport_idx_t::invalid()) {
        auto const rt = rtt->unix_event_time(rt_t, stop_idx, event_type::kDep);
        date::to_stream(out, " RT %d.%m %R", rt);
        date::to_stream(out, " [%d.%m %R]", to_local_time(tz, rt));
      }

      auto const& trip_section = tt.transport_to_trip_section_.at(i);
      auto const& merged_trips = tt.merged_trips_.at(
          trip_section.size() == 1U ? trip_section[0]
                                    : trip_section.at(stop_idx));

      auto const transport_line = tt.transport_section_lines_.at(i);
      auto const& line = transport_line.empty()
                             ? ""
                             : tt.trip_lines_
                                   .at(transport_line.size() == 1U
                                           ? transport_line.front()
                                           : transport_line.at(stop_idx))
                                   .view();

      if (with_debug) {
        out << "   line=\"" << line << "\"";
      }

      out << "  [";
      for (auto const& trip_idx : merged_trips) {
        auto j = 0U;

        for (auto const [dbg, id] :
             utl::zip(tt.trip_debug_.at(trip_idx), tt.trip_ids_.at(trip_idx))) {
          if (j++ != 0) {
            out << ", ";
          }
          out << "{name=" << tt.trip_display_names_.at(trip_idx).view()
              << ", day=";
          date::to_stream(
              out, "%F",
              tt.internal_interval_days().from_ + to_idx(day_idx) * 1_days);
          out << ", id=" << tt.trip_id_strings_.at(id).view()
              << ", src=" << static_cast<int>(to_idx(tt.trip_id_src_.at(id)));
          if (with_debug) {
            out << ", debug="
                << tt.source_file_names_.at(dbg.source_file_idx_).view() << ":"
                << dbg.line_number_from_ << ":" << dbg.line_number_to_
                << ", train_nr=" << tt.trip_train_nr_.at(id);
          }
          out << "}";
        }
      }
      out << "]";
    }

    out << "\n";
  }
}

void print_transport(timetable const& tt,
                     rt_timetable const* rtt,
                     std::ostream& out,
                     transport x,
                     bool const with_debug) {
  print_transport(
      tt, rtt, out, x,
      interval{stop_idx_t{0U}, std::numeric_limits<stop_idx_t>::max()}, 0U,
      with_debug);
}

void print_transport(timetable const& tt,
                     rt_timetable const* rtt,
                     std::ostream& out,
                     rt_transport_idx_t rt_t,
                     interval<stop_idx_t> stop_range,
                     unsigned const indent_width,
                     bool const with_debug) {
  utl::verify(rtt != nullptr, "print_transport: rt timetable not set");
  utl::verify(holds_alternative<transport>(
                  rtt->rt_transport_static_transport_.at(rt_t)),
              "print_transport: additional transport not supported");
  auto const x = rtt->rt_transport_static_transport_.at(rt_t).as<transport>();
  print_transport(tt, rtt, out, x, stop_range, indent_width, with_debug);
}

void print_transport(timetable const& tt,
                     rt_timetable const* rtt,
                     std::ostream& out,
                     rt_transport_idx_t x,
                     bool const with_debug) {
  print_transport(
      tt, rtt, out, x,
      interval{stop_idx_t{0U}, std::numeric_limits<stop_idx_t>::max()}, 0U,
      with_debug);
}

}  // namespace nigiri
