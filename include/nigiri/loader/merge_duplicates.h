#include "nigiri/types.h"

namespace nigiri {
struct timetable;
}

namespace nigiri::loader {

unsigned find_duplicates(timetable& tt, location_idx_t a, location_idx_t b);

}  // namespace nigiri::loader