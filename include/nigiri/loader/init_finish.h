#pragma once

namespace nigiri {
struct timetable;
}

namespace nigiri::loader {

void register_special_stations(timetable&);
void finalize(timetable&,
              bool adjust_footpaths = false,
              bool merge_duplicates = false);

}  // namespace nigiri::loader