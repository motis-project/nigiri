#include "nigiri/loader/match_set.h"

namespace nigiri {
struct timetable;
}

namespace nigiri::loader {

match_set_t link_nearby_stations(timetable&, bool const store_matches);

}  // namespace nigiri::loader