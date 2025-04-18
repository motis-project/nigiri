#pragma once

#ifdef NO_DATA
#undef NO_DATA
#endif
#include "gtfsrt/gtfs-realtime.pb.h"

#include "nigiri/rt/gtfsrt_update.h"
#include "nigiri/string_store.h"
#include "nigiri/types.h"

namespace nigiri {
struct rt_timetable;
struct timetable;
}  // namespace nigiri

namespace nigiri::rt {

void handle_alert(date::sys_days const today,
                  timetable const&,
                  rt_timetable&,
                  string_cache_t&,
                  source_idx_t const,
                  std::string_view tag,
                  transit_realtime::Alert const&,
                  statistics&);

}  // namespace nigiri::rt