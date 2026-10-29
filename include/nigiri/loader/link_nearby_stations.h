#include "nigiri/footpath.h"
#include "nigiri/types.h"

namespace nigiri {
struct timetable;
}

namespace nigiri::loader {

void link_nearby_stations(
    timetable&, mutable_fws_multimap<location_idx_t, footpath>& walks);

}  // namespace nigiri::loader
