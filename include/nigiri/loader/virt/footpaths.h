#pragma once

#include <vector>

#include "nigiri/footpath.h"
#include "nigiri/timetable.h"
#include "nigiri/types.h"

namespace nigiri::loader {

void expand_profile_to_virts(
    timetable&,
    profile_idx_t,
    std::vector<source_idx_t> const& excluded_sources = {});

void write_profiles(
    timetable&,
    mutable_fws_multimap<location_idx_t, footpath> const& fallback_fps,
    mutable_fws_multimap<location_idx_t, footpath> const& qualified_rule_fps,
    std::vector<source_idx_t> const& excluded_sources);

}  // namespace nigiri::loader
