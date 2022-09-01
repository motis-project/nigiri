#include "nigiri/loader/hrd/stamm/stamm.h"

#include "fmt/ranges.h"

#include "utl/to_vec.h"

#include "nigiri/loader/hrd/stamm/basic_info.h"

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

stamm::stamm(config const& c, timetable& tt, dir const& d) : tt_{tt} {
  auto const files = load_files(c, d);
  timezones_ = parse_timezones(c, tt, files.at(TIMEZONES).data());
  locations_ = parse_stations(
      c, source_idx_t{0U}, timezones_, tt, files.at(STATIONS).data(),
      files.at(COORDINATES).data(), files.at(FOOTPATHS).data());
  bitfields_ = parse_bitfields(c, tt, files.at(BITFIELDS).data());
  categories_ = parse_categories(c, files.at(CATEGORIES).data());
  providers_ = parse_providers(c, tt, files.at(PROVIDERS).data());
  attributes_ = parse_attributes(c, tt, files.at(ATTRIBUTES).data());
  directions_ = parse_directions(c, tt, files.at(DIRECTIONS).data());
  tt.date_range_ = parse_interval(files.at(BASIC_DATA).data());
}

}  // namespace nigiri::loader::hrd