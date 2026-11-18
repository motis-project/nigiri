#pragma once

#include "nigiri/loader/station_footpaths.h"
#include "nigiri/footpath.h"
#include "nigiri/timetable.h"
#include "nigiri/types.h"

namespace nigiri::loader {

void expand_profile_to_virts(timetable&, profile_idx_t);

void write_profiles(
    timetable&,
    mutable_fws_multimap<location_idx_t, footpath> const& fallback_fps,
    mutable_fws_multimap<location_idx_t, footpath> const& qualified_rule_fps,
    station_footpaths_t const& = {},
    rule_pairs_t const& = {});

}  // namespace nigiri::loader
