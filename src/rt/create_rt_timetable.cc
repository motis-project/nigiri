#include "nigiri/rt/create_rt_timetable.h"

#include "utl/enumerate.h"

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
  rtt.alerts_.route_type_.resize(tt.n_sources());
  rtt.alerts_.route_id_.resize(tt.n_sources());
  for (auto const [src, r] : utl::enumerate(tt.route_ids_)) {
    rtt.alerts_.route_id_[source_idx_t{src}].resize(r.route_id_type_.size());
  }
  rtt.alerts_.location_.resize(tt.n_locations());
  rtt.alerts_.agency_.resize(tt.n_agencies());
  for (auto i = 0U; i != kNProfiles; ++i) {
    if (!tt.locations_.footpaths_out_[i].empty()) {
      rtt.has_td_footpaths_out_[i].resize(tt.n_locations());
      rtt.has_td_footpaths_in_[i].resize(tt.n_locations());
      rtt.td_footpaths_out_[i].resize(tt.n_locations());
      rtt.td_footpaths_in_[i].resize(tt.n_locations());
    }
  }
  rtt.additional_trips_.resize(tt.n_sources());
  rtt.fwd_search_lb_graph_ = tt.fwd_search_lb_graph_;
  rtt.bwd_search_lb_graph_ = tt.bwd_search_lb_graph_;

  return rtt;
}

}  // namespace nigiri::rt