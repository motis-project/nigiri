#include "nigiri/types.h"

namespace nigiri {
struct timetable;
}

namespace nigiri::loader {

using merge_threshold_t = std::array<duration_t, kNumClasses>;

void find_intra_route_duplicates(timetable&, merge_threshold_t const&);

unsigned merge_duplicates(timetable&,
                          merge_threshold_t const&,
                          location_idx_t,
                          location_idx_t);

}  // namespace nigiri::loader