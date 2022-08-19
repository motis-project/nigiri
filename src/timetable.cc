#include "nigiri/timetable.h"

#include "nigiri/print_transport.h"

namespace nigiri {

std::ostream& operator<<(std::ostream& out, timetable const& tt) {
  auto const num_days = (tt.end_ - tt.begin_ + 1_days) / 1_days;
  for (auto i = 0U; i != tt.transport_stop_times_.size(); ++i) {
    auto const transport_idx = transport_idx_t{i};
    auto const traffic_days =
        tt.bitfields_.at(tt.transport_traffic_days_.at(transport_idx));
    out << "TRANSPORT=" << transport_idx << ", TRAFFIC_DAYS="
        << traffic_days.to_string().substr(traffic_days.size() - num_days)
        << "\n";
    for (auto day = tt.begin_; day <= tt.end_; day += 1_days) {
      auto const day_idx = day_idx_t{
          static_cast<day_idx_t ::value_t>((day - tt.begin_) / 1_days)};
      if (traffic_days.test(to_idx(day_idx))) {
        date::to_stream(out, "%F", day);
        out << " (day_idx=" << day_idx << ")\n";
        print_transport(tt, out, transport_idx, day_idx);
        out << "\n";
      }
    }
    out << "---\n\n";
  }
  return out;
}

}  // namespace nigiri