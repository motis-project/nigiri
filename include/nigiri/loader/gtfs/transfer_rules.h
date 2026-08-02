#pragma once

#include <vector>

#include "nigiri/loader/gtfs/route.h"
#include "nigiri/loader/gtfs/stop.h"
#include "nigiri/loader/gtfs/trip.h"

namespace nigiri {
struct timetable;
}

namespace nigiri::loader::gtfs {

// Applies the transfers.txt rules that cannot be expressed as plain stop-pair
// footpaths:
//
// - transfer_type=3 (transfer not possible): directed marker edge with
//   duration footpath::kMaxDuration
// - route-/trip-qualified rules (from_route_id/to_route_id,
//   from_trip_id/to_trip_id): the qualified trips are split off to virtual
//   child locations (location_type::kGeneratedTransfer) so the rule only
//   applies to them
//
// Matching does not trust location_type: a rule stop matches an event
// location if they are related via the parent chain, ranked by specificity
// (exact match > rule stop is ancestor = station-level cascade > rule stop is
// descendant = data error, applied defensively). Qualifier specificity:
// trip > route > unqualified.
//
// All resulting directed edges are emitted into
// timetable::locations::transfer_rule_fps_. These are authoritative: street
// routing only fills transfers without a rule and never fills forbidden
// pairs.
void build_transfer_rules(timetable&,
                          std::vector<raw_transfer_rule> const&,
                          stops_map_t const&,
                          route_map_t const&,
                          trip_data&);

}  // namespace nigiri::loader::gtfs
