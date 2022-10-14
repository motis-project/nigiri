#include "nigiri/print_transport.h"

#include <codecvt>
#include <iomanip>

#include "utl/enumerate.h"
#include "utl/zip.h"

#include "date/date.h"

#include "nigiri/common/indent.h"

namespace nigiri {

void print_transport(timetable const& tt,
                     std::ostream& out,
                     transport x,
                     interval<unsigned> stop_range,
                     unsigned const indent_width,
                     bool const with_debug) {
  std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> utf8_conv;

  auto const i = x.t_idx_;
  auto const day_idx = x.day_;
  auto const& route_idx = tt.transport_route_.at(i);
  auto const& stop_seq = tt.route_location_seq_.at(route_idx);
  auto const& stop_times = tt.transport_stop_times_.at(i);

  auto const from =
      std::min(static_cast<unsigned>(stop_seq.size()), stop_range.from_);
  auto const to =
      std::min(static_cast<unsigned>(stop_seq.size()), stop_range.to_);
  for (auto stop_idx = from; stop_idx != to; ++stop_idx) {
    auto const s = timetable::stop{stop_seq.at(stop_idx)};
    auto const location_idx = s.location_idx();
    auto const& stop_id = tt.locations_.ids_.at(location_idx).view();
    auto const& parent = tt.locations_.parents_.at(location_idx);
    auto const stop_name = parent == location_idx_t::invalid()
                               ? tt.locations_.names_.at(location_idx).view()
                               : tt.locations_.names_.at(parent).view();
    auto const& tz = tt.locations_.timezones_.at(
        tt.locations_.location_timezones_.at(location_idx));
    indent(out, indent_width);
    fmt::print(out, "{:2}: {:7} {:.<48} ", stop_idx, stop_id, stop_name);

    if (stop_idx != from) {
      auto const t = tt.date_range_.from_ + to_idx(day_idx) * 1_days +
                     stop_times.at(2 * stop_idx - 1);
      if (!s.out_allowed()) {
        fmt::print(" -");
      } else {
        fmt::print("  ");
      }
      date::to_stream(out, "a: %d.%m %R", t);
      date::to_stream(out, " [%d.%m %R]", to_local_time(tz, t));
    } else {
      out << "              ";
      out << "              ";
    }

    if (stop_idx != to - 1) {
      auto const t = tt.date_range_.from_ + to_idx(day_idx) * 1_days +
                     stop_times.at(2 * stop_idx);
      if (!s.in_allowed()) {
        fmt::print("   -");
      } else {
        fmt::print("    ");
      }

      date::to_stream(out, "  d: %d.%m %R", t);
      date::to_stream(out, " [%d.%m %R]", to_local_time(tz, t));

      auto const& trip_section = tt.transport_to_trip_section_.at(i);
      auto const& merged_trips = tt.merged_trips_.at(
          trip_section.size() == 1U ? trip_section[0]
                                    : trip_section.at(stop_idx));
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
          date::to_stream(out, "%F",
                          tt.date_range_.from_ + to_idx(day_idx) * 1_days);
          out << ", id=" << tt.trip_id_strings_.at(id).view()
              << ", src=" << static_cast<int>(to_idx(tt.trip_id_src_.at(id)));
          if (with_debug) {
            out << ", debug="
                << tt.source_file_names_.at(dbg.source_file_idx_).view() << ":"
                << dbg.line_number_from_ << ":" << dbg.line_number_to_;
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
                     std::ostream& out,
                     transport x,
                     bool const with_debug) {
  print_transport(tt, out, x,
                  interval{0U, std::numeric_limits<unsigned>::max()}, 0U,
                  with_debug);
}

}  // namespace nigiri
