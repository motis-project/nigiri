#include "nigiri/lookup/get_transport.h"

#include "nigiri/timetable.h"

namespace nigiri {

std::optional<transport> get_transport(timetable const& tt,
                                       std::string_view trip_idx,
                                       date::year_month_day const day) {
  auto const lb = std::lower_bound(
      begin(tt.trip_id_to_idx_), end(tt.trip_id_to_idx_), trip_idx,
      [&](pair<trip_id_idx_t, trip_idx_t> const& a, std::string_view b) {
        return tt.trip_id_strings_[a.first].view() < b;
      });

  if (lb == end(tt.trip_id_to_idx_) ||
      tt.trip_id_strings_[lb->first].view() != trip_idx) {
    return std::nullopt;
  }

  auto const day_idx = tt.day_idx(day);
  for (auto it = lb; it != end(tt.trip_id_to_idx_); ++it) {
    auto const t = tt.trip_ref_transport_[it->second].first;
    auto const& traffic_days = tt.bitfields_[tt.transport_traffic_days_[t]];
    if (traffic_days.test(to_idx(day_idx))) {
      return transport{.t_idx_ = t, .day_ = day_idx};
    }
  }

  return std::nullopt;
}

}  // namespace nigiri