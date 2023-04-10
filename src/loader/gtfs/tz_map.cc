#include "nigiri/loader/gtfs/tz_map.h"

#include "utl/get_or_create.h"

#include "nigiri/timetable.h"

namespace nigiri::loader::gtfs {

timezone_idx_t get_tz_idx(timetable& tt,
                          tz_map& timezones,
                          std::string_view tz_name) {
  return tz_name.empty()
             ? timezone_idx_t::invalid()
             : utl::get_or_create(timezones, tz_name, [&]() {
                 return tt.locations_.register_timezone(timezone{
                     static_cast<void const*>(date::locate_zone(tz_name))});
               });
}

}  // namespace nigiri::loader::gtfs
