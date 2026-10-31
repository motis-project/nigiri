#include "nigiri/loader/import_context.h"

#include <cassert>

#include "utl/get_or_create.h"

#include "nigiri/timetable.h"

namespace nigiri::loader {

import_context::import_context(timetable& tt)
    : trip_stop_seq_numbers_{tt.trip_stop_seq_numbers_.make_cache()},
      section_providers_{tt.transport_section_providers_.make_cache()},
      section_directions_{tt.transport_section_directions_.make_cache()} {}

bitfield_idx_t register_bitfield(timetable& tt,
                                 import_context& ctx,
                                 bitfield const& b) {
  return utl::get_or_create(ctx.bitfields_, b,
                            [&]() { return tt.register_bitfield(b); });
}

void add_transport(timetable& tt,
                   import_context& ctx,
                   timetable::transport&& t) {
  tt.transport_first_dep_offset_.emplace_back(t.first_dep_offset_);
  tt.transport_traffic_days_.emplace_back(t.bitfield_idx_);
  tt.transport_route_.emplace_back(t.route_idx_);
  tt.transport_to_trip_section_.emplace_back(t.external_trip_ids_);
  tt.transport_section_attributes_.emplace_back(t.section_attributes_);
  tt.transport_section_providers_.emplace_back(ctx.section_providers_,
                                               t.section_providers_);
  tt.transport_section_directions_.emplace_back(ctx.section_directions_,
                                                t.section_directions_);

  assert(tt.transport_traffic_days_.size() == tt.transport_route_.size());
  assert(tt.transport_traffic_days_.size() ==
         tt.transport_to_trip_section_.size());
  assert(t.section_directions_.size() == 0U ||
         t.section_directions_.size() == 1U ||
         t.section_directions_.size() ==
             tt.route_location_seq_.at(t.route_idx_).size() - 1U);
}

}  // namespace nigiri::loader
