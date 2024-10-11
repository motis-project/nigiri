#include "nigiri/loader/gtfs-flex/load_td_timetable.h"

#include <filesystem>

#include "utl/get_or_create.h"
#include "utl/progress_tracker.h"

#include "cista/hash.h"

#include "wyhash.h"

#include "nigiri/loader/get_index.h"
#include "nigiri/loader/gtfs/files.h"
#include "nigiri/loader/gtfs/local_to_utc.h"
#include "nigiri/loader/gtfs/noon_offsets.h"
#include "nigiri/loader/gtfs/stop_seq_number_encoding.h"
#include "nigiri/loader/loader_interface.h"
#include "nigiri/common/sort_by.h"
#include "nigiri/logging.h"

namespace fs = std::filesystem;

namespace nigiri::loader::gtfs_flex {

constexpr auto const required_files = { k_td_StopTimesFile, k_td_BookingRulesFile};
constexpr auto const conditionally_required_files = { k_td_CalenderFile, k_td_CalendarDatesFile,
                                                      k_td_LocationGeojsonFile, k_td_LocationGroupsFile,
                                                      k_td_LocationGroupStopsFile, k_td_StopFile,
                                                      k_td_StopAreasFile, k_td_TripsFile};

cista::hash_t hash(dir const& d) {
  if (d.type() == dir_type::kZip) {
    return d.hash();
  }

  auto h = std::uint64_t{0U};
  auto const hash_file = [&](fs::path const& p) {
    if (!d.exists(p)) {
      h = wyhash64(h, _wyp[0]);
    } else {
      auto const f = d.get_file(p);
      auto const data = f.data();
      h = wyhash(data.data(), data.size(), h, _wyp);
    }
  };

  hash_file(k_td_StopTimesFile);
  hash_file(k_td_BookingRulesFile);
  hash_file(k_td_CalenderFile);
  hash_file(k_td_CalendarDatesFile);
  hash_file(k_td_LocationGeojsonFile);
  hash_file(k_td_LocationGroupsFile);
  hash_file(k_td_LocationGroupStopsFile);
  hash_file(k_td_StopFile);
  hash_file(k_td_StopAreasFile);
  hash_file(k_td_TripsFile);

  return h;
}

bool applicable(dir const& d) {
  for (auto const& file_name : required_files) {
    if (!d.exists(file_name)) {
      return false;
    }
  }
  const auto calendarExists = d.exists(k_td_CalenderFile);
  const auto calendarDatesExists = d.exists(k_td_CalendarDatesFile);
  const auto tripsExists = d.exists(k_td_TripsFile);
  const auto locationGeojsonExists = d.exists(k_td_LocationGeojsonFile);
  const auto locationGroupStopsExists = d.exists(k_td_LocationGroupStopsFile);
  const auto stopsExists = d.exists(k_td_StopFile);
  const auto stopAreasExists = d.exists(k_td_StopAreasFile);

  const auto condition1 = !(calendarExists || calendarDatesExists) || tripsExists;
  const auto condition2 = locationGeojsonExists || stopsExists;
  const auto condition3 = !locationGroupStopsExists || stopsExists;
  const auto condition4 = !stopAreasExists || stopsExists;

  return condition1 && condition2 && condition3 && condition4;
}

td_timetable_map_t load_td_timetable(loader_config const& config,
                    source_idx_t src,
                    dir const& d,
                    assistance_times *assistance) {
  auto local_bitfield_indices = hash_map<bitfield, bitfield_idx_t>{};
  load_td_timetable(config, src, d, local_bitfield_indices, assistance);
}

td_timetable_map_t load_td_timetable(loader_config const& config,
                    source_idx_t src,
                    dir const& d,
                    hash_map<bitfield, bitfield_idx_t>& h,
                    assistance_times* assistance) {
  scoped_timer const global_timer{"gtfs-flex parser"};

  auto const load = [&](std::string_view file_name) -> file {
    return d.exists(file_name) ? d.get_file(file_name) : file{};
  };

  auto stop_times = read_td_stop_times(load(k_td_StopTimesFile).data());
  auto booking_rules = read_td_booking_rules(load(k_td_BookingRulesFile).data());

  auto calendar = read_td_calendar(load(k_td_CalenderFile).data());
  auto calendar_dates = read_td_calendar_date(load(k_td_CalendarDatesFile).data());
  auto location_geojson = read_td_location_geojson(load(k_td_LocationGeojsonFile).data());
  auto location_groups = read_td_location_groups(load(k_td_LocationGroupsFile).data());
  auto location_group_stops = read_td_location_group_stops(load(k_td_LocationGroupStopsFile).data());
  auto stops = read_td_stops(load(k_td_StopFile).data());
  auto stop_areas = read_td_stop_areas(load(k_td_StopAreasFile).data());
  auto trips = read_td_trips(load(k_td_TripsFile).data());

  return create_td_timetable( booking_rules, calendar, calendar_dates, location_groups, location_group_stops,
                              location_geojson, stops, stop_areas, stop_times, trips);
}

}  // namespace nigiri::loader::gtfs_flex