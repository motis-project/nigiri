#include "nigiri/footpath.h"
#include "nigiri/types.h"

namespace nigiri {
struct timetable;
}

namespace nigiri::loader {

struct import_context;

void link_nearby_stations(timetable&,
                          import_context&,
                          mutable_fws_multimap<location_idx_t, footpath>& walks);

}  // namespace nigiri::loader
