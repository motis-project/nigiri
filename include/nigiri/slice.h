#pragma once

#include "nigiri/common/delta_t.h"
#include "nigiri/footpath.h"
#include "nigiri/stop.h"
#include "nigiri/types.h"

namespace nigiri {

struct timetable;

using l_idx_t = location_idx_t;
using t_idx_t = transport_idx_t;
using r_idx_t = route_idx_t;
using fp = footpath;

struct slice {
  explicit slice(timetable const&, interval<day_idx_t>);

  vector_map<location_idx_t, l_idx_t> location_l_;
  vector_map<l_idx_t, location_idx_t> l_location_;

  vecvec<l_idx_t, r_idx_t> l_r_;

  array<vecvec<l_idx_t, fp>, kMaxProfiles> footpaths_out_;
  array<vecvec<l_idx_t, fp>, kMaxProfiles> footpaths_in_;

  vecvec<r_idx_t, stop::value_type> r_l_seq_;

  vector_map<t_idx_t, transport> t_transport_;
  vector_map<r_idx_t, interval<t_idx_t>> r_transport_ranges_;

  vector_map<r_idx_t, interval<std::uint32_t>> r_stop_time_ranges_;
  vector<delta_t> r_stop_times_;
  day_idx_t base_day_;
};

}  // namespace nigiri