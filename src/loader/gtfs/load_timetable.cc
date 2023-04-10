#include "nigiri/loader/gtfs/load_timetable.h"

#include <filesystem>
#include <numeric>
#include <string>

#include "boost/algorithm/string.hpp"

#include "utl/erase_if.h"
#include "utl/get_or_create.h"
#include "utl/pairwise.h"
#include "utl/parallel_for.h"
#include "utl/parser/cstr.h"
#include "utl/pipes/accumulate.h"
#include "utl/pipes/all.h"
#include "utl/pipes/remove_if.h"
#include "utl/pipes/transform.h"
#include "utl/pipes/vec.h"
#include "utl/progress_tracker.h"

#include "cista/hash.h"
#include "cista/mmap.h"

#include "geo/latlng.h"
#include "geo/point_rtree.h"

#include "nigiri/loader/get_index.h"
#include "nigiri/loader/gtfs/agency.h"
#include "nigiri/loader/gtfs/calendar.h"
#include "nigiri/loader/gtfs/calendar_date.h"
#include "nigiri/loader/gtfs/files.h"
#include "nigiri/loader/gtfs/local_to_utc.h"
#include "nigiri/loader/gtfs/route.h"
#include "nigiri/loader/gtfs/services.h"
#include "nigiri/loader/gtfs/stop.h"
#include "nigiri/loader/gtfs/stop_time.h"
#include "nigiri/loader/gtfs/trip.h"
#include "nigiri/logging.h"
#include "nigiri/timetable.h"

namespace fs = std::filesystem;

namespace nigiri::loader::gtfs {

constexpr auto const required_files = {kAgencyFile, kStopFile, kRoutesFile,
                                       kTripsFile, kStopTimesFile};

cista::hash_t hash(fs::path const& path) {
  auto hash = cista::BASE_HASH;
  auto const hash_file = [&](fs::path const& p) {
    if (!fs::is_regular_file(p)) {
      return;
    }
    cista::mmap m{p.generic_string().c_str(), cista::mmap::protection::READ};
    hash = cista::hash_combine(
        cista::hash(std::string_view{
            reinterpret_cast<char const*>(m.begin()),
            std::min(static_cast<size_t>(50 * 1024 * 1024), m.size())}),
        hash);
  };

  for (auto const& file_name : required_files) {
    hash_file(path / file_name);
  }
  hash_file(path / kCalenderFile);
  hash_file(path / kCalendarDatesFile);

  return hash;
}

bool applicable(dir const& d) {
  for (auto const& file_name : required_files) {
    if (!d.exists(file_name)) {
      return false;
    }
  }
  return d.exists(kCalenderFile) || d.exists(kCalendarDatesFile);
}

void load_timetable(source_idx_t const src, dir const& d, timetable& tt) {
  nigiri::scoped_timer const global_timer{"gtfs parser"};

  auto const load = [&](std::string_view file_name) -> file {
    return d.exists(file_name) ? d.get_file(file_name) : file{};
  };

  tz_map timezones;

  auto const agencies = read_agencies(tt, timezones, load(kAgencyFile).data());
  auto const stops = read_stops(src, tt, timezones, load(kStopFile).data(),
                                load(kTransfersFile).data());
  auto const routes = read_routes(agencies, load(kRoutesFile).data());
  auto const calendar = read_calendar(load(kCalenderFile).data());
  auto const dates = read_calendar_date(load(kCalendarDatesFile).data());
  auto const traffic_days = merge_traffic_days(calendar, dates);
  auto [trips, blocks] =
      read_trips(routes, traffic_days, load(kTripsFile).data());
  read_frequencies(trips, load(kFrequenciesFile).data());
  read_stop_times(trips, stops, load(kStopTimesFile).data());

  for (auto& [_, t] : trips) {
    t->interpolate();
  }

  std::map<std::pair<std::basic_string<timetable::stop::value_type>, clasz>,
           std::vector<std::vector<utc_trip>>>
      route_services;
  for (auto const& [_, t] : trips) {
    expand_trip(
        tt, t.get(), traffic_days.interval_, tt.date_range_,
        [&](utc_trip const& s) {
          auto const route_key =
              std::pair{s.orig_->stops(), s.orig_->route_->clasz_};
          if (auto const it = route_services.find(route_key);
              it != end(route_services)) {
            for (auto& r : it->second) {
              auto const idx = get_index(r, s);
              if (idx.has_value()) {
                r.insert(begin(r) + *idx, s);
                return;
              }
            }
            it->second.emplace_back(std::vector<utc_trip>{std::move(s)});
          } else {
            route_services.emplace(
                route_key, std::vector<std::vector<utc_trip>>{{std::move(s)}});
          }
        });
  }

  auto const source_file_idx = tt.register_source_file("trips.txt");
  auto const attributes = std::basic_string<attribute_combination_idx_t>{};
  auto bitfield_indices = hash_map<bitfield, bitfield_idx_t>{};
  auto lines = hash_map<std::string, trip_line_idx_t>{};
  auto directions = hash_map<std::string, trip_direction_idx_t>{};
  for (auto const& [key, sub_routes] : route_services) {
    for (auto const& services : sub_routes) {
      auto const& [stop_seq, sections_clasz] = key;
      auto const route_idx = tt.register_route(
          stop_seq, std::basic_string<clasz>({sections_clasz}));
      for (auto const& s : services) {
        auto const id = tt.register_trip_id(
            s.orig_->id_, src, s.orig_->short_name_,
            {source_file_idx, s.orig_->line_, s.orig_->line_},
            tt.next_transport_idx(),
            {0U, static_cast<unsigned>(stop_seq.size())});
        auto const merged_trip = tt.register_merged_trip({id});
        tt.add_transport(timetable::transport{
            .bitfield_idx_ = utl::get_or_create(
                bitfield_indices, s.utc_traffic_days_,
                [&]() { return tt.register_bitfield(s.utc_traffic_days_); }),
            .route_idx_ = route_idx,
            .external_trip_ids_ = {merged_trip},
            .section_attributes_ = attributes,
            .section_providers_ = {s.orig_->route_->agency_},
            .section_directions_ = {utl::get_or_create(
                directions, s.orig_->headsign_,
                [&]() {
                  auto const trip_dir_str =
                      tt.register_trip_direction_string(s.orig_->headsign_);
                  auto const idx = tt.trip_directions_.size();
                  tt.trip_directions_.emplace_back(trip_dir_str);
                  return trip_direction_idx_t{idx};
                })},
            .section_lines_ = {
                utl::get_or_create(lines, s.orig_->route_->short_name_, [&]() {
                  auto const idx = trip_line_idx_t{tt.trip_lines_.size()};
                  tt.trip_lines_.emplace_back(s.orig_->route_->short_name_);
                  return idx;
                })}});
      }
    }
  }
}

}  // namespace nigiri::loader::gtfs
