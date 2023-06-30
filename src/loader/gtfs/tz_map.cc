#include "nigiri/loader/gtfs/tz_map.h"

#include "utl/get_or_create.h"
#include "utl/verify.h"

#include "nigiri/timetable.h"

namespace nigiri::loader::gtfs {

timezone_idx_t get_tz_idx(timetable& tt,
                          tz_map& timezones,
                          std::string_view tz_name) {
  utl::verify(!tz_name.empty(), "timezone not set");
  return utl::get_or_create(timezones, tz_name, [&]() {
    return tt.locations_.register_timezone(timezone{
        cista::pair{string{tz_name},
                    static_cast<void const*>(date::locate_zone(tz_name))}});
  });
}

}  // namespace nigiri::loader::gtfs
