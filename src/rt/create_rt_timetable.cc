#include "nigiri/rt/create_rt_timetable.h"

#include "nigiri/rt/rt_timetable.h"
#include "nigiri/timetable.h"

namespace nigiri::rt {

rt_timetable create_rt_timetable(timetable const& tt,
                                 date::sys_days const base_day) {
  auto rtt = rt_timetable{};
  rtt.transport_traffic_days_ = tt.transport_traffic_days_;
  rtt.bitfields_ = tt.bitfields_;
  rtt.base_day_ = base_day;
  rtt.base_day_idx_ = tt.day_idx(rtt.base_day_);
  // resize for later memory accesses
  rtt.location_rt_transports_[location_idx_t{tt.n_locations() - 1U}];
  for (auto i = 0U; i != kMaxProfiles; ++i) {
    if (!tt.locations_.footpaths_out_[i].empty()) {
      rtt.has_td_footpaths_out_[i].resize(tt.n_locations());
      rtt.has_td_footpaths_in_[i].resize(tt.n_locations());
      rtt.td_footpaths_out_[i].resize(tt.n_locations());
      rtt.td_footpaths_in_[i].resize(tt.n_locations());
    }
  }
  return rtt;
}

}  // namespace nigiri::rt