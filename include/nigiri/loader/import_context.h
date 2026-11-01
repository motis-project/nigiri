#pragma once

#include "nigiri/common/dedup_vecvec.h"
#include "nigiri/timetable.h"
#include "nigiri/types.h"

namespace nigiri::loader {

// Loader-only data: filled by the loaders, consumed by finalize.
struct import_context {
  explicit import_context(timetable&);
  import_context(import_context const&) = delete;
  import_context(import_context&&) = delete;
  import_context& operator=(import_context const&) = delete;
  import_context& operator=(import_context&&) = delete;

  // Route 1:
  //   stop-1-dep: [trip1, trip2, ..., tripN]
  //   stop-2-arr: [trip1, trip2, ..., tripN]
  //   ...
  // Route 2:
  //  stop-1-dep: [...]
  // ...
  // RouteN: ...
  //
  // -> filled by loaders during import
  // -> finalize: converts to compressed stop times
  vector_map<route_idx_t, interval<std::uint32_t>> route_stop_time_ranges_;
  vector<delta> route_stop_times_;

  paged_vecvec<trip_idx_t, transport_range_t> trip_transport_ranges_;
  mutable_fws_multimap<trip_idx_t, trip_id_idx_t> trip_ids_;
  mutable_fws_multimap<trip_idx_t, trip_debug> trip_debug_;

  mutable_fws_multimap<location_idx_t, location_idx_t> equivalences_;
  mutable_fws_multimap<location_idx_t, location_idx_t> children_;
  mutable_fws_multimap<location_idx_t, footpath> preprocessing_footpaths_out_;
  mutable_fws_multimap<location_idx_t, footpath> preprocessing_footpaths_in_;

  vector_map<source_idx_t, paged_vecvec<route_id_idx_t, trip_idx_t>>
      route_id_trips_;

  paged_vecvec<location_group_idx_t, location_idx_t> location_group_locations_;
  paged_vecvec<location_idx_t, location_group_idx_t> location_location_groups_;
  paged_vecvec<location_group_idx_t, flex_transport_idx_t>
      location_group_transports_;
  paged_vecvec<flex_area_idx_t, flex_transport_idx_t> flex_area_transports_;

  decltype(timetable::trip_stop_seq_numbers_)::cache_t trip_stop_seq_numbers_;

  decltype(timetable::transport_section_providers_)::cache_t section_providers_;
  decltype(timetable::transport_section_directions_)::cache_t
      section_directions_;

  hash_map<bitfield, bitfield_idx_t> bitfields_;
};

bitfield_idx_t register_bitfield(timetable&, import_context&, bitfield const&);

void add_transport(timetable&, import_context&, timetable::transport&&);

}  // namespace nigiri::loader
