#include "nigiri/loader/gtfs/validate.h"
#include "nigiri/loader/gtfs/load_timetable.h"
#include "nigiri/loader/gtfs/files.h"
#include "nigiri/loader/gtfs/agency.h"
#include "nigiri/loader/gtfs/route.h"

namespace nigiri::loader::gtfs {

    std::vector<timetable_source> validate_sources(std::vector<timetable_source> const& sources) {
      auto accepted = std::vector<timetable_source>{};
      for (auto const& source : sources) {
          auto const dir = find_gtfs_root(make_dir(source.path_));

          if(!applicable(*dir)) {
            accepted.emplace_back(source);
            continue;
          }
          auto const agency_file = dir->get_file(kAgencyFile);
          auto const routes_file = dir->get_file(kRoutesFile);
          auto const agency_ids = read_agency_ids(agency_file.data());
          auto const default_tz = source.loader_config_.default_tz_;
          if(!has_invalid_agency_timezone(agency_file.data(), default_tz) &&
            !has_unknown_route_agency(routes_file.data(), agency_ids, default_tz)) {
            accepted.emplace_back(source);
          }
      }
      return accepted;
    }

} // namespace nigiri::loader::gtfs