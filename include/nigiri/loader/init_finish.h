#pragma once

#include <stdint.h>

namespace nigiri {
struct timetable;
}

namespace nigiri::loader {

void register_special_stations(timetable&);
void finalize(timetable&, uint16_t const& no_profiles);

}  // namespace nigiri::loader