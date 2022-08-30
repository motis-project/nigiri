#include "nigiri/loader/hrd/load_timetable.h"

#include "fmt/ranges.h"

#include "utl/helpers/algorithm.h"

#include "nigiri/loader/hrd/bitfield.h"
#include "nigiri/loader/hrd/service.h"
#include "nigiri/loader/hrd/station.h"
#include "nigiri/loader/hrd/timezone.h"

namespace nigiri::loader::hrd {

std::vector<file> load_files(config const& c, dir const& d) {
  return utl::to_vec(
      c.required_files_, [&](std::vector<std::string> const& alt) {
        if (alt.empty()) {
          return file{};
        }
        for (auto const& file : alt) {
          try {
            auto f = d.get_file(c.core_data_ / file);
            log(log_lvl::info, "nigiri.loader.hrd.load_files",
                "loaded {}: {} bytes", c.core_data_ / file, f.data().size());
            return f;
          } catch (...) {
          }
        }
        throw utl::fail("no file available: {}", alt);
      });
}

bool applicable(config const& c, dir const& d) {
  return utl::all_of(
      c.required_files_, [&](std::vector<std::string> const& alt) {
        return alt.empty() || utl::any_of(alt, [&](std::string const& file) {
                 auto const exists = d.exists(c.core_data_ / file);
                 if (!exists) {
                   std::clog << "missing file for config " << c.version_.view()
                             << ": " << (c.core_data_ / file) << "\n";
                 }
                 return exists;
               });
      });
}

void load_timetable(source_idx_t const src,
                    config const& c,
                    dir const& d,
                    timetable& tt) {
  auto const files = load_files(c, d);
  auto const timezones = parse_timezones(c, tt, files.at(TIMEZONES).data());
  auto const locations = parse_stations(
      c, source_idx_t{0U}, timezones, tt, files.at(STATIONS).data(),
      files.at(COORDINATES).data(), files.at(FOOTPATHS).data());
  auto const bitfields = parse_bitfields(c, tt, files.at(BITFIELDS).data());
  auto const categories = parse_categories(c, files.at(CATEGORIES).data());
  auto const providers = parse_providers(c, tt, files.at(PROVIDERS).data());
  auto const attributes = parse_attributes(c, tt, files.at(ATTRIBUTES).data());
  auto const directions = parse_directions(c, files.at(DIRECTIONS).data());
  auto const interval = parse_interval(files.at(BASIC_DATA).data());

  tt.date_range_ = interval;
  tt.n_days_ = static_cast<std::uint16_t>(tt.date_range_.size().count());

  std::vector<file> service_files;
  service_builder sb{tt};
  for (auto const& s : d.list_files(c.fplan_)) {
    auto const& f = service_files.emplace_back(d.get_file(s));
    sb.add_services(c, f.filename(), interval, bitfields, timezones, locations,
                    f.data(), [](std::size_t) {});
  }
  sb.write_services(src, locations, categories, providers, attributes,
                    directions);
}

}  // namespace nigiri::loader::hrd