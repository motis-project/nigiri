#pragma once

#include <cinttypes>
#include <limits>

#include "nigiri/loader/build_footpaths.h"

namespace nigiri {
struct timetable;
}

namespace nigiri::loader {

void register_special_stations(timetable&);
void finalize(timetable&, finalize_options);
void finalize(timetable&,
              bool adjust_footpaths = false,
              bool merge_dupes_intra_src = false,
              bool merge_dupes_inter_src = false,
              bool permutate_loc = false,
              std::uint16_t max_footpath_length =
                  std::numeric_limits<std::uint16_t>::max());

}  // namespace nigiri::loader