#include "nigiri/loader/gtfs/validate.h"
#include "nigiri/loader/gtfs/load_timetable.h"
#include "nigiri/loader/gtfs/files.h"
#include "nigiri/loader/gtfs/agency.h"

namespace nigiri::loader::gtfs {

    std::vector<timetable_source> validate_sources(std::vector<timetable_source> const& sources) {
      auto accepted = std::vector<timetable_source>{};
      for (auto const& source : sources) {
          auto const dir = make_dir(source.path_);

          if(!applicable(*dir)) {
            accepted.emplace_back(source);
            continue;
          }
          auto const agency_file = dir->get_file(kAgencyFile);
          if(!has_invalid_agency_timezone(agency_file.data(), source.loader_config_.default_tz_)) {
            accepted.emplace_back(source);
          }
      }
      return accepted;
    }

} // namespace nigiri::loader::gtfs