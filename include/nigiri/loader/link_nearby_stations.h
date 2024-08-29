#include "nigiri/loader/match_set.h"

namespace nigiri {
struct timetable;
}

namespace nigiri::loader {

template <bool MatchIntraSrc, bool MatchInterSrc>
match_set_t link_nearby_stations(timetable&);

}  // namespace nigiri::loader