#include <set>
#include <string>

#include "nigiri/rt/frun.h"
#include "nigiri/timetable.h"

namespace nigiri::loader {

inline std::set<std::string> service_strings(timetable const& tt) {
  auto const reverse = [](std::string s) {
    std::reverse(s.begin(), s.end());
    return s;
  };
  auto const range = tt.internal_interval_days();
  auto const num_days = static_cast<size_t>((range.size() + 2_days) / 1_days);
  auto ret = std::set<std::string>{};
  for (auto i = 0U; i != tt.transport_traffic_days_.size(); ++i) {
    std::stringstream out;
    auto const transport_idx = transport_idx_t{i};
    auto const num_stops =
        tt.route_location_seq_[tt.transport_route_[transport_idx]].size();
    auto const traffic_days =
        tt.bitfields_.at(tt.transport_traffic_days_.at(transport_idx));
    out << "TRAFFIC_DAYS="
        << reverse(
               traffic_days.to_string().substr(traffic_days.size() - num_days))
        << "\n";
    for (auto d = range.from_; d != range.to_; d += std::chrono::days{1}) {
      auto const day_idx = day_idx_t{
          static_cast<day_idx_t::value_t>((d - range.from_) / 1_days)};
      if (traffic_days.test(to_idx(day_idx))) {
        date::to_stream(out, "%F", d);
        out << " (day_idx=" << day_idx << ")\n";
        out << rt::frun{
            tt,
            nullptr,
            {.t_ = transport{transport_idx, day_idx},
             .stop_range_ = {0U, static_cast<stop_idx_t>(num_stops)}}};
      }
    }
    ret.emplace(out.str());
  }
  return ret;
}

}  // namespace nigiri::loader
