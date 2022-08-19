#include "nigiri/loader/hrd/load_timetable.h"

#include "fmt/ranges.h"

#include "nigiri/loader/hrd/bitfield.h"
#include "nigiri/loader/hrd/service.h"
#include "nigiri/loader/hrd/station.h"
#include "nigiri/loader/hrd/timezone.h"

namespace nigiri::loader::hrd {

std::vector<file> load_files(config const& c, dir const& d) {
  return utl::to_vec(c.required_files_,
                     [&](std::vector<std::string> const& alt) {
                       if (alt.empty()) {
                         return file{};
                       }
                       for (auto const& file : alt) {
                         try {
                           return d.get_file(c.core_data_ / file);
                         } catch (...) {
                         }
                       }
                       throw utl::fail("no file available: {}", alt);
                     });
}

std::shared_ptr<timetable> load_timetable(config const& c, dir const& d) {
  auto tt = std::make_shared<timetable>();

  auto const src = source_idx_t{0U};
  auto const files = load_files(c, d);
  auto const timezones = parse_timezones(c, *tt, files.at(TIMEZONES).data());
  auto const locations = parse_stations(
      c, source_idx_t{0U}, timezones, *tt, files.at(STATIONS).data(),
      files.at(COORDINATES).data(), files.at(FOOTPATHS).data());
  auto const bitfields = parse_bitfields(c, *tt, files.at(BITFIELDS).data());
  auto const categories = parse_categories(c, files.at(CATEGORIES).data());
  auto const providers = parse_providers(c, files.at(PROVIDERS).data());
  auto const interval = parse_interval(files.at(BASIC_DATA).data());

  tt->begin_ = std::chrono::sys_days{interval.first};
  tt->end_ = std::chrono::sys_days{interval.second};

  std::vector<file> service_files;
  service_builder sb{*tt, {}};
  for (auto const& s : d.list_files(c.fplan_)) {
    auto const& f = service_files.emplace_back(d.get_file(s));
    sb.add_services(c, f.filename(), interval, bitfields, timezones, locations,
                    f.data(), [](std::size_t) {});
  }
  sb.write_services(src, categories, providers);

  return tt;
}

}  // namespace nigiri::loader::hrd