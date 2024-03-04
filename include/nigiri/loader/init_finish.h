#pragma once

#include <cinttypes>
#include <limits>

namespace nigiri {
struct timetable;
}

namespace nigiri::loader {

void register_special_stations(timetable&);
void finalize(timetable&,
              bool adjust_footpaths = false,
              bool merge_duplicates = false,
              std::uint16_t max_footpath_length =
                  std::numeric_limits<std::uint16_t>::max());

}  // namespace nigiri::loader