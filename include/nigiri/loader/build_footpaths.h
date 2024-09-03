#pragma once

#include "nigiri/timetable.h"

namespace nigiri::loader {

void build_footpaths(timetable& tt,
                     bool adjust_footpaths,
                     bool merge_dupes_intra_src,
                     bool merge_dupes_inter_src,
                     std::uint16_t max_footpath_length);

}  // namespace nigiri::loader
