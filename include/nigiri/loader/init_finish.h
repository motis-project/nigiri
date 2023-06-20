#pragma once

#include <stdint.h>

namespace nigiri {
struct timetable;
}

namespace nigiri::loader {

void register_special_stations(timetable&);
void finalize(timetable&, uint16_t const& no_profiles = 1);

/**
 * Reinitialization of the profile-based footpaths. A separate footpath_out_
 * and footpath_in is created for each profile.
 *
 * Works in place and updates tt.locations_.footpaths_[out, in]_
 *
 * After calling this method it holds:
 * - tt.locations_.footpaths_[out, in].size() == no_profiles; no_profiles >= 1
 * - all footpaths are initialized using the default footpaths.
 */
void reinitialize_footpaths(timetable&, uint16_t const& no_profiles);

}  // namespace nigiri::loader