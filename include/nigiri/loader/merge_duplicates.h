#include "nigiri/types.h"

namespace nigiri {
struct timetable;
}

namespace nigiri::loader {

void find_intra_route_duplicates(timetable&);
unsigned find_duplicates(timetable&, location_idx_t a, location_idx_t b);

}  // namespace nigiri::loader