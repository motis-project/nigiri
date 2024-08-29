#include "nigiri/types.h"

namespace nigiri {
struct timetable;
}

namespace nigiri::loader {

using match_set_t = hash_set<pair<location_idx_t, location_idx_t>>;

unsigned find_duplicates(timetable& tt,
                         match_set_t const& matches,
                         location_idx_t a,
                         location_idx_t b);

}  // namespace nigiri::loader