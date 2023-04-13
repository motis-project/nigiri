#include "nigiri/loader/gtfs/load_timetable.h"

#include <filesystem>
#include <numeric>
#include <string>

#include "utl/get_or_create.h"
#include "utl/progress_tracker.h"

#include "tsl/hopscotch_map.h"

#include "cista/hash.h"
#include "cista/mmap.h"

#include "nigiri/loader/get_index.h"
#include "nigiri/loader/gtfs/agency.h"
#include "nigiri/loader/gtfs/calendar.h"
#include "nigiri/loader/gtfs/calendar_date.h"
#include "nigiri/loader/gtfs/files.h"
#include "nigiri/loader/gtfs/local_to_utc.h"
#include "nigiri/loader/gtfs/noon_offsets.h"
#include "nigiri/loader/gtfs/route.h"
#include "nigiri/loader/gtfs/services.h"
#include "nigiri/loader/gtfs/stop.h"
#include "nigiri/loader/gtfs/stop_time.h"
#include "nigiri/loader/gtfs/trip.h"
#include "nigiri/common/sort_by.h"
#include "nigiri/logging.h"
#include "nigiri/timetable.h"

namespace fs = std::filesystem;

namespace nigiri::loader::gtfs {

constexpr auto const required_files = {kAgencyFile, kStopFile, kRoutesFile,
                                       kTripsFile, kStopTimesFile};

cista::hash_t hash(dir const& d) {
  auto hash = cista::BASE_HASH;
  auto const hash_file = [&](fs::path const& p) {
    if (!d.exists(p)) {
      return;
    }
    hash = cista::hash_combine(cista::hash(d.get_file(p).data()), hash);
  };

  for (auto const& file_name : required_files) {
    hash_file(file_name);
  }
  hash_file(kCalenderFile);
  hash_file(kCalendarDatesFile);

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

using route_key_t =
    std::pair<std::basic_string<timetable::stop::value_type>, clasz>;

struct hash_route_key {
  std::size_t operator()(route_key_t const& x) const {
    return cista::hashing<route_key_t>{}(x);
  }
};

void load_timetable(source_idx_t const src, dir const& d, timetable& tt) {
  auto bars = utl::global_progress_bars{false};

  nigiri::scoped_timer const global_timer{"gtfs parser"};

  auto const load = [&](std::string_view file_name) -> file {
    return d.exists(file_name) ? d.get_file(file_name) : file{};
  };

  auto progress_tracker = utl::activate_progress_tracker("nigiri");
  auto timezones = tz_map{};
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

  {
    auto const timer = scoped_timer{"loader.gtfs.trips.sort"};
    for (auto& [_, t] : trips) {
      if (t->requires_sorting_) {
        t->stop_headsigns_.resize(t->seq_numbers_.size());
        std::tie(t->seq_numbers_, t->stop_seq_, t->event_times_,
                 t->stop_headsigns_) =
            sort_by(t->seq_numbers_, t->stop_seq_, t->event_times_,
                    t->stop_headsigns_);
      }
    }
  }

  {
    auto const timer = scoped_timer{"loader.gtfs.trips.interpolate"};
    for (auto& [_, t] : trips) {
      t->interpolate();
    }
  }

  tsl::hopscotch_map<
      std::pair<std::basic_string<timetable::stop::value_type>, clasz>,
      std::vector<std::vector<utc_trip>>, hash_route_key>
      route_services;

  {
    progress_tracker->status("Expand Trips")
        .out_bounds(70.F, 85.F)
        .in_high(trips.size());

    auto const timer = scoped_timer{"loader.gtfs.trips.expand"};

    auto const noon_offsets =
        precompute_noon_offsets(tt, traffic_days.interval_, agencies);
    for (auto const& [_, t] : trips) {
      expand_trip(
          noon_offsets, tt, t.get(), traffic_days.interval_, tt.date_range_,
          [&](utc_trip&& s) {
            auto const route_key =
                std::pair{s.orig_->stop_seq_, s.orig_->route_->clasz_};
            auto const it = route_services.find(route_key);
            if (it != end(route_services)) {
              for (auto& r : it.value()) {
                auto const idx = get_index(r, s);
                if (idx.has_value()) {
                  r.insert(std::next(begin(r), static_cast<int>(*idx)), s);
                  return;
                }
              }
              it.value().emplace_back(std::vector<utc_trip>{std::move(s)});
            } else {
              route_services.emplace(
                  route_key,
                  std::vector<std::vector<utc_trip>>{{std::move(s)}});
            }
          });
      progress_tracker->increment();
    }
  }

  {
    progress_tracker->status("Write Trips")
        .out_bounds(85.F, 100.F)
        .in_high(route_services.size());

    auto const timer = scoped_timer{"loader.gtfs.routes.build"};

    auto const source_file_idx = tt.register_source_file("trips.txt");
    auto const attributes = std::basic_string<attribute_combination_idx_t>{};
    auto bitfield_indices = hash_map<bitfield, bitfield_idx_t>{};
    auto lines = hash_map<std::string, trip_line_idx_t>{};
    auto directions = hash_map<std::string, trip_direction_idx_t>{};
    for (auto const& [key, sub_routes] : route_services) {
      for (auto const& services : sub_routes) {
        auto const& [stop_seq, sections_clasz] = key;
        auto const route_idx = tt.register_route(stop_seq, {sections_clasz});
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
              .section_lines_ = {utl::get_or_create(
                  lines, s.orig_->route_->short_name_, [&]() {
                    auto const idx = trip_line_idx_t{tt.trip_lines_.size()};
                    tt.trip_lines_.emplace_back(s.orig_->route_->short_name_);
                    return idx;
                  })}});
        }

        tt.finish_route();

        auto const stop_times_begin = tt.route_stop_times_.size();
        for (auto const [from, to] :
             utl::pairwise(interval{std::size_t{0U}, stop_seq.size()})) {
          // Write departure times of all route services at stop i.
          for (auto const& s : services) {
            tt.route_stop_times_.emplace_back(s.utc_times_[from * 2]);
          }

          // Write arrival times of all route services at stop i+1.
          for (auto const& s : services) {
            tt.route_stop_times_.emplace_back(s.utc_times_[to * 2 - 1]);
          }
        }
        auto const stop_times_end = tt.route_stop_times_.size();
        tt.route_stop_time_ranges_.emplace_back(
            interval{stop_times_begin, stop_times_end});
      }

      progress_tracker->increment();
    }
  }
}

}  // namespace nigiri::loader::gtfs
