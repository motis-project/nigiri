#include "nigiri/print_transport.h"

#include <codecvt>
#include <iomanip>

#include "date/date.h"

namespace nigiri {

void print_transport(timetable const& tt,
                     std::ostream& out,
                     trip_idx_t const i,
                     day_idx_t const day_idx) {
  std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> utf8_conv;

  out << tt.trip_display_names_.at(i) << "\n";

  auto const& stop_seq = tt.route_location_seq_.at(tt.trip_route_.at(i));
  auto const& stop_times = tt.trip_stop_times_.at(i);
  for (auto stop_idx = 0U; stop_idx != stop_seq.size(); ++stop_idx) {
    auto const location_idx = stop_seq.at(stop_idx).location_idx();
    auto const& stop_name = tt.locations_.names_.at(location_idx);
    auto const& stop_id = tt.locations_.ids_.at(location_idx);
    auto const stop_name_len = utf8_conv.from_bytes(stop_name.str()).size();
    std::cout << std::right << std::setw(2) << stop_idx << ": " << std::left
              << std::setw(7) << stop_id << " " << std::left
              << std::setw(std::max(0, 50 - static_cast<int>(stop_name_len) +
                                           static_cast<int>(stop_name.size())))
              << std::setfill('.') << stop_name << std::setfill(' ');

    if (stop_idx != 0U) {
      auto const t = tt.begin_ + to_idx(day_idx) * duration_t{1440} +
                     stop_times.at(2 * stop_idx - 1);
      date::to_stream(out, " a: %d.%m %R", t);
    } else {
      out << "               ";
    }

    if (stop_idx != stop_seq.size() - 1U) {
      auto const t = tt.begin_ + to_idx(day_idx) * duration_t{1440} +
                     stop_times.at(2 * stop_idx);
      date::to_stream(out, "  d: %d.%m %R", t);
    }
    out << "\n";
  }
}

}  // namespace nigiri