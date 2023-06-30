#pragma once

namespace nigiri {
struct timetable;
}

namespace nigiri::loader {

void register_special_stations(timetable&);
void finalize(timetable&);

}  // namespace nigiri::loader