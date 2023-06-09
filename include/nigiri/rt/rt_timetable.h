#pragma once

#include "nigiri/common/delta_t.h"
#include "nigiri/common/interval.h"
#include "nigiri/stop.h"
#include "nigiri/timetable.h"
#include "nigiri/types.h"

namespace nigiri {

// General note:
// - The real-time timetable does not use bitfields. It requires an initial copy
//   of the bitfields from the static timetable to be able to deactivate bits
//   for transports that are updated with delays, rerouting (incl. track
//   changes) or cancellations (without changing the static timetable).
// - RT transports represent departure and arrival times relative to a base day.
// - RT transports are currently not grouped into routes to simplify the code.
//   If this leads to performance issues during the routing, grouping into
//   routes can be introduced for real-time routing as well.
// - All RT transports can be resolved via their static transport if they were
//   already scheduled in the static timetable.
// - All RT transports that did not exist in the static timetable, can be looked
//   up with their trip_id in the RT timetable.
struct rt_timetable {
  delta_t unix_to_delta(unixtime_t const t) const {
    return (t - std::chrono::time_point_cast<unixtime_t::duration>(base_day_))
        .count();
  }

  // Updated transport traffic days from the static timetable.
  // Initial: 100% copy from static, then adapted according to real-time updates
  vector_map<transport_idx_t, bitfield_idx_t> transport_traffic_days_;
  vector_map<bitfield_idx_t, bitfield> bitfields_;

  // Base-day: all real-time timestamps (departures + arrivals in
  // rt_transport_stop_times_) are given relative to this base day.
  date::sys_days base_day_;
  day_idx_t base_day_idx_;

  // Lookup: static transport -> realtime transport
  // only works for transport that existed in the static timetable
  hash_map<transport, rt_transport_idx_t> static_trip_lookup_;

  // Lookup: additional trip index -> realtime transport
  hash_map<rt_add_trip_id_idx_t, rt_transport_idx_t> additional_trips_lookup_;

  // RT transport -> static transport (not for additional trips)
  vector_map<rt_transport_idx_t, variant<transport, rt_add_trip_id_idx_t>>
      rt_transport_static_transport_;

  // RT trip ID index -> ID strings + source
  vecvec<rt_add_trip_id_idx_t, char> trip_id_strings_;
  vector_map<rt_transport_idx_t, source_idx_t> rt_transport_src_;

  // RT trip ID index -> train number, if available (otherwise 0)
  vector_map<rt_transport_idx_t, std::uint32_t> rt_transport_train_nr_;

  // RT transport -> event times (dep, arr, dep, arr, ...)
  vecvec<rt_transport_idx_t, delta_t> rt_transport_stop_times_;
  vecvec<rt_transport_idx_t, stop::value_type> rt_transport_location_seq_;

  // RT trip index -> display name (empty if not changed)
  vecvec<rt_transport_idx_t, char> rt_transport_display_names_;

  // RT transport -> vehicle clasz for each section
  vecvec<rt_transport_idx_t, clasz> rt_transport_section_clasz_;

  vecvec<rt_transport_idx_t, rt_merged_trips_idx_t>
      rt_transport_to_trip_section_;

  vecvec<rt_merged_trips_idx_t, variant<transport, rt_add_trip_id_idx_t>>
      merged_trips_;

  rt_transport_idx_t next_rt_transport_idx_{0U};
};

}  // namespace nigiri