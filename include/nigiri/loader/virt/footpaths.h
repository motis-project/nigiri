#pragma once

#include "nigiri/footpath.h"
#include "nigiri/timetable.h"
#include "nigiri/types.h"

namespace nigiri::loader {

void expand_profile_to_virts(timetable&, profile_idx_t);

void write_profiles(
    timetable&,
    mutable_fws_multimap<location_idx_t, footpath> const& walks,
    mutable_fws_multimap<location_idx_t, footpath> const& virt_footpaths);

}  // namespace nigiri::loader
