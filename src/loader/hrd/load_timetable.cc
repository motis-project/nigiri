#include "nigiri/loader/hrd/load_timetable.h"

#include "nigiri/loader/hrd/bitfield.h"
#include "nigiri/loader/hrd/service.h"
#include "nigiri/loader/hrd/station.h"
#include "nigiri/loader/hrd/timezone.h"

namespace nigiri::loader::hrd {

std::shared_ptr<timetable> load_timetable(
    config const& c,
    file_list const& files,
    std::vector<std::string_view> const& services) {
  auto tt = std::make_shared<timetable>();

  auto const timezones =
      parse_timezones(c, *tt, files.at(TIMEZONES).value_or(""));
  auto const locations = parse_stations(
      c, source_idx_t{0U}, timezones, *tt, files.at(STATIONS).value(),
      files.at(COORDINATES).value(), files.at(FOOTPATHS).value());
  auto const bitfields = parse_bitfields(c, *tt, files.at(BITFIELDS).value());
  auto const categories = parse_categories(c, files.at(CATEGORIES).value());
  auto const providers = parse_providers(c, files.at(PROVIDERS).value());
  auto const interval = parse_interval(files.at(BASIC_DATA).value());

  tt->begin_ = std::chrono::sys_days{interval.first};
  tt->end_ = std::chrono::sys_days{interval.second};

  for (auto const& s : services) {
    write_services(c, source_idx_t{0}, "services.txt", interval, bitfields,
                   timezones, locations, categories, providers, s, *tt,
                   [](std::size_t) {});
  }

  return tt;
}

}  // namespace nigiri::loader::hrd