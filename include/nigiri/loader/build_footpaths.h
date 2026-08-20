#pragma once

#include "nigiri/timetable.h"

namespace nigiri::loader {

struct finalize_options {
  bool adjust_footpaths_{true};
  bool merge_dupes_intra_src_{true};
  bool merge_dupes_inter_src_{true};
  std::uint16_t max_footpath_length_{20};
};

void build_footpaths(timetable& tt, finalize_options);

// Rebuild one profile's hubs from a walking layer computed elsewhere (street
// routing). The rule hubs of the default profile are kept verbatim - a rule
// fixes the transfer time regardless of how the walks were derived - and the
// walk hubs are derived from `fps`, which may gain the pairs no hub covers.
void build_profile_hubs(timetable&,
                        profile_idx_t,
                        vector_map<location_idx_t, std::vector<footpath>>& fps);

// Drop the default profile's footpaths that a hub already hands out at the
// same weight or better.
void prune_hub_covered_footpaths(timetable&);

// Reference mode: write every pair the default profile's hubs stand for as an
// ordinary footpath. The hubs stay - see the comment at the definition.
void expand_hubs_into_footpaths(timetable&);

}  // namespace nigiri::loader
