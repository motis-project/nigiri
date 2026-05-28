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
    return tt.register_timezone(timezone{
        cista::pair{string{tz_name},
                    static_cast<void const*>(date::locate_zone(tz_name))}});
  });
}

std::optional<std::string_view> get_timezone_name(timetable const& tt,
                                                  timezone_idx_t const i) {
  return i == timezone_idx_t::invalid()
             ? std::nullopt
             : std::optional{tt.timezones_[i].apply(
                   utl::overloaded{[](tz_offsets const&) {
                                     assert(false);
                                     return "";
                                   },
                                   [](pair<string, void const*> const& x) {
                                     return x.first.view();
                                   }})};
}

}  // namespace nigiri::loader::gtfs
