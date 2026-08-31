#include "nigiri/types.h"

namespace nigiri {
struct timetable;
}

namespace nigiri::loader {

unsigned find_duplicates(timetable& tt, location_idx_t a, location_idx_t b);

void connect_by_train_nr(timetable& tt,
                         location_idx_t const a,
                         location_idx_t const b);

}  // namespace nigiri::loader
