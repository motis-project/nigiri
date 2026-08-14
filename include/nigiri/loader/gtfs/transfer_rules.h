#pragma once

#include <string_view>

#include "nigiri/loader/gtfs/route.h"
#include "nigiri/loader/gtfs/stop.h"
#include "nigiri/loader/gtfs/trip.h"

namespace nigiri {
struct timetable;
}

namespace nigiri::loader::gtfs {

// Reads and applies transfers.txt. Has to run after stops, routes, trips and
// stop times are read and before routes are built (it rewrites trip stop
// sequences).
//
// - stay-seated transfers (transfer_type=4) are wired into
//   trip::seated_in_/seated_out_
// - same-stop rows set the stop transfer time
// - cross-stop rows are added as footpaths
// - transfer_type=3 (transfer not possible): directed marker edge with
//   duration footpath::kMaxDuration
// - route-/trip-qualified rules (from_route_id/to_route_id,
//   from_trip_id/to_trip_id): the qualified trips are split off to virtual
//   child locations (location_type::kVirt) so that the rule only applies to
//   them
//
// Rule matching does not trust location_type: a rule stop matches an event
// location if they are related via the parent chain, ranked by specificity
// (exact match beats a station level cascade). Qualifier specificity:
// trip > route > unqualified.
//
// All transfer times given in the data are authoritative: they are emitted as
// directed edges into timetable::locations::transfer_rule_fps_ which override
// computed footpaths. Street routing may only fill transfers without a rule
// and never a forbidden pair.
void read_transfers(timetable&,
                    std::string_view file_content,
                    stops_map_t const&,
                    route_map_t const&,
                    trip_data&);

}  // namespace nigiri::loader::gtfs
