#include "nigiri/print_transport.h"

#include <codecvt>
#include <iomanip>

#include "utl/enumerate.h"
#include "utl/zip.h"

#include "date/date.h"

namespace nigiri {

void print_transport(timetable const& tt,
                     std::ostream& out,
                     transport_idx_t const i,
                     day_idx_t const day_idx,
                     bool with_debug) {
  std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> utf8_conv;

  auto const& route_idx = tt.transport_route_.at(i);
  auto const& stop_seq = tt.route_location_seq_.at(route_idx);
  auto const& stop_times = tt.transport_stop_times_.at(i);
  out << "ROUTE=" << route_idx << "\n";
  for (auto stop_idx = 0U; stop_idx != stop_seq.size(); ++stop_idx) {
    auto const location_idx = stop_seq.at(stop_idx).location_idx();
    auto const& stop_name = tt.locations_.names_.at(location_idx);
    auto const& stop_id = tt.locations_.ids_.at(location_idx);
    auto const& tz = tt.locations_.timezones_.at(
        tt.locations_.location_timezones_.at(location_idx));
    auto const stop_name_len = utf8_conv.from_bytes(stop_name.str()).size();
    out << std::right << std::setw(2) << std::setfill(' ') << stop_idx << ": "
        << std::left << std::setw(7) << stop_id << " " << std::left
        << std::setw(std::max(
               0, 50 - static_cast<int>(stop_name_len + stop_name.size())))
        << std::setfill('.') << stop_name << std::setfill(' ');

    if (stop_idx != 0U) {
      auto const t = tt.begin_ + to_idx(day_idx) * 1_days +
                     stop_times.at(2 * stop_idx - 1);
      date::to_stream(out, " a: %d.%m %R", t);
      date::to_stream(out, " [%d.%m %R]", to_local_time(tz, t));
    } else {
      out << "               ";
      out << "              ";
    }

    if (stop_idx != stop_seq.size() - 1U) {
      auto const t =
          tt.begin_ + to_idx(day_idx) * 1_days + stop_times.at(2 * stop_idx);
      date::to_stream(out, "  d: %d.%m %R", t);
      date::to_stream(out, " [%d.%m %R]", to_local_time(tz, t));

      auto const& merged_trips =
          tt.merged_trips_.at(tt.transport_to_trip_section_.at(i).at(stop_idx));
      out << "  [";
      for (auto const& trip_idx : merged_trips) {
        auto j = 0U;

        for (auto const [dbg, id] :
             utl::zip(tt.trip_debug_.at(trip_idx), tt.trip_ids_.at(trip_idx))) {
          if (j++ != 0) {
            out << ", ";
          }
          out << "{name=" << tt.trip_display_names_.at(trip_idx) << ", day=";
          date::to_stream(out, "%F", tt.begin_ + to_idx(day_idx) * 1_days);
          out << ", id=" << id.id_
              << ", src=" << static_cast<int>(to_idx(id.src_));
          if (with_debug) {
            out << ", debug=" << dbg;
          }
          out << "}";
        }
      }
      out << "]";
    }

    out << "\n";
  }
}

}  // namespace nigiri
