#include "nigiri/timetable.h"

#include "nigiri/print_transport.h"

namespace nigiri {

std::string reverse(std::string s) {
  std::reverse(s.begin(), s.end());
  return s;
}

std::ostream& operator<<(std::ostream& out, timetable const& tt) {
  auto const num_days = static_cast<size_t>(
      (tt.date_range_.to_ - tt.date_range_.from_ + 1_days) / 1_days);
  for (auto i = 0U; i != tt.transport_stop_times_.size(); ++i) {
    auto const transport_idx = transport_idx_t{i};
    auto const traffic_days =
        tt.bitfields_.at(tt.transport_traffic_days_.at(transport_idx));
    out << "TRANSPORT=" << transport_idx << ", TRAFFIC_DAYS="
        << reverse(
               traffic_days.to_string().substr(traffic_days.size() - num_days))
        << "\n";
    for (auto const& d : tt.date_range_) {
      auto const day_idx = day_idx_t{
          static_cast<day_idx_t::value_t>((d - tt.date_range_.from_) / 1_days)};
      if (traffic_days.test(to_idx(day_idx))) {
        date::to_stream(out, "%F", d);
        out << " (day_idx=" << day_idx << ")\n";
        print_transport(tt, out, {transport_idx, day_idx});
        out << "\n";
      }
    }
    out << "---\n\n";
  }
  return out;
}

}  // namespace nigiri