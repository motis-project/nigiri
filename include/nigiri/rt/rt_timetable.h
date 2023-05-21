#pragma once

#include "nigiri/common/delta_t.h"
#include "nigiri/common/interval.h"
#include "nigiri/stop.h"
#include "nigiri/timetable.h"
#include "nigiri/types.h"

namespace nigiri {

struct rt_timetable {
  vector<pair<rt_trip_id_idx_t, rt_trip_idx_t>> trip_id_to_idx_;
  mutable_fws_multimap<rt_trip_idx_t, rt_trip_id_idx_t> trip_ids_;
  vecvec<rt_trip_id_idx_t, char> trip_id_strings_;
  vecvec<rt_trip_idx_t, char> trip_display_names_;
  vector_map<rt_trip_id_idx_t, source_idx_t> trip_id_src_;
  vector_map<rt_trip_idx_t, pair<transport_idx_t, interval<std::uint32_t>>>
      trip_ref_transport_;
  vector_map<rt_route_idx_t, interval<rt_transport_idx_t>>
      route_transport_ranges_;
  vector_map<rt_route_idx_t, interval<unsigned>> route_stop_time_ranges_;
  vector<delta_t> route_stop_times_;
  vecvec<rt_route_idx_t, stop::value_type> route_location_seq_;
  vector_map<rt_transport_idx_t, rt_route_idx_t> transport_route_;
  vecvec<rt_merged_trips_idx_t, rt_trip_idx_t> merged_trips_;
  vecvec<rt_transport_idx_t, rt_merged_trips_idx_t> transport_to_trip_section_;
  vecvec<route_idx_t, clasz> route_section_clasz_;

  day_idx_t base_day_;
};

}  // namespace nigiri