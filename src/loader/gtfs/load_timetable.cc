#include "nigiri/loader/gtfs/load_timetable.h"

#include <filesystem>
#include <numeric>
#include <string>

#include "utl/get_or_create.h"
#include "utl/progress_tracker.h"

#include "cista/hash.h"
#include "cista/mmap.h"

#include "wyhash.h"

#include "nigiri/loader/get_index.h"
#include "nigiri/loader/gtfs/agency.h"
#include "nigiri/loader/gtfs/calendar.h"
#include "nigiri/loader/gtfs/calendar_date.h"
#include "nigiri/loader/gtfs/files.h"
#include "nigiri/loader/gtfs/local_to_utc.h"
#include "nigiri/loader/gtfs/noon_offsets.h"
#include "nigiri/loader/gtfs/route.h"
#include "nigiri/loader/gtfs/route_key.h"
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

  hash_file(kAgencyFile);
  hash_file(kStopFile);
  hash_file(kRoutesFile);
  hash_file(kTripsFile);
  hash_file(kStopTimesFile);
  hash_file(kCalenderFile);
  hash_file(kCalendarDatesFile);
  hash_file(kTransfersFile);
  hash_file(kFeedInfoFile);
  hash_file(kFrequenciesFile);

  return h;
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

  auto const progress_tracker = utl::get_active_progress_tracker();
  auto timezones = tz_map{};
  auto agencies = read_agencies(tt, timezones, load(kAgencyFile).data());
  auto const stops = read_stops(src, tt, timezones, load(kStopFile).data(),
                                load(kTransfersFile).data());
  auto const routes =
      read_routes(tt, timezones, agencies, load(kRoutesFile).data());
  auto const calendar = read_calendar(load(kCalenderFile).data());
  auto const dates = read_calendar_date(load(kCalendarDatesFile).data());
  auto const service = merge_traffic_days(calendar, dates);
  auto trip_data = read_trips(routes, service, load(kTripsFile).data());
  read_frequencies(trip_data, load(kFrequenciesFile).data());
  read_stop_times(trip_data, stops, load(kStopTimesFile).data());

  {
    auto const timer = scoped_timer{"loader.gtfs.trips.sort"};
    for (auto& t : trip_data.data_) {
      if (t.requires_sorting_) {
        t.stop_headsigns_.resize(t.seq_numbers_.size());
        std::tie(t.seq_numbers_, t.stop_seq_, t.event_times_,
                 t.stop_headsigns_) =
            sort_by(t.seq_numbers_, t.stop_seq_, t.event_times_,
                    t.stop_headsigns_);
      }
    }
  }

  {
    auto const timer = scoped_timer{"loader.gtfs.trips.interpolate"};
    for (auto& t : trip_data.data_) {
      t.interpolate();
    }
  }

  hash_map<route_key_t, std::vector<std::vector<utc_trip>>, route_key_hash,
           route_key_equals>
      route_services;

  auto const noon_offsets =
      precompute_noon_offsets(tt, service.interval_, agencies);

  stop_seq_t stop_seq_cache;
  auto const get_route_key =
      [&](std::basic_string<gtfs_trip_idx_t> const& trips) {
        if (trips.size() == 1U) {
          return &trip_data.get(trips.front()).stop_seq_;
        } else {
          stop_seq_cache.clear();
          for (auto const [i, t_idx] : utl::enumerate(trips)) {
            auto const& trp = trip_data.get(t_idx);
            stop_seq_cache.insert(
                end(stop_seq_cache),
                i == 0 ? begin(trp.stop_seq_) : std::next(begin(trp.stop_seq_)),
                end(trp.stop_seq_));
          }
          return &stop_seq_cache;
        }
      };

  std::basic_string<minutes_after_midnight_t> utc_time_mem;
  auto const add_trip = [&](std::basic_string<gtfs_trip_idx_t> const& trips,
                            bitfield const* traffic_days) {
    expand_trip(
        trip_data, noon_offsets, tt, trips, traffic_days, service.interval_,
        tt.date_range_, utc_time_mem, [&](utc_trip&& s) {
          auto const* stop_seq = get_route_key(s.trips_);
          auto const clasz = trip_data.get(s.trips_.front()).route_->clasz_;
          auto const it = route_services.find(std::pair{clasz, stop_seq});
          if (it != end(route_services)) {
            for (auto& r : it->second) {
              auto const idx = get_index(r, s);
              if (idx.has_value()) {
                r.insert(std::next(begin(r), static_cast<int>(*idx)), s);
                return;
              }
            }
            it->second.emplace_back(std::vector<utc_trip>{std::move(s)});
          } else {
            route_services.emplace(
                std::pair{clasz, *stop_seq},
                std::vector<std::vector<utc_trip>>{{std::move(s)}});
          }
        });
  };

  {
    progress_tracker->status("Expand Trips")
        .out_bounds(70.F, 85.F)
        .in_high(trip_data.data_.size());
    auto const timer = scoped_timer{"loader.gtfs.trips.expand"};

    for (auto const [i, t] : utl::enumerate(trip_data.data_)) {
      if (t.block_ != nullptr) {
        continue;
      }
      add_trip({gtfs_trip_idx_t{i}}, t.service_);
      progress_tracker->increment();
    }
  }

  {
    progress_tracker->status("Stay Seated")
        .out_bounds(85.F, 87.F)
        .in_high(route_services.size());
    auto const timer = scoped_timer{"loader.gtfs.trips.block_id"};

    for (auto const& [_, blk] : trip_data.blocks_) {
      for (auto const& [trips, traffic_days] : blk->rule_services(trip_data)) {
        add_trip(trips, &traffic_days);
      }
    }
  }

  {
    progress_tracker->status("Write Trips")
        .out_bounds(87.F, 100.F)
        .in_high(route_services.size());

    auto const is_train_number = [](auto const& s) {
      return !s.empty() && std::all_of(begin(s), end(s), [](auto&& c) -> bool {
        return std::isdigit(c);
      });
    };

    auto trip_id_buf = fmt::memory_buffer{};
    auto const timer = scoped_timer{"loader.gtfs.routes.build"};
    auto const source_file_idx =
        tt.register_source_file((d.path() / kStopTimesFile).generic_string());
    auto const attributes = std::basic_string<attribute_combination_idx_t>{};
    auto bitfield_indices = hash_map<bitfield, bitfield_idx_t>{};
    auto lines = hash_map<std::string, trip_line_idx_t>{};
    auto directions = hash_map<std::string, trip_direction_idx_t>{};
    auto section_directions = std::basic_string<trip_direction_idx_t>{};
    auto section_lines = std::basic_string<trip_line_idx_t>{};
    auto external_trip_ids = std::basic_string<merged_trips_idx_t>{};
    for (auto const& [key, sub_routes] : route_services) {
      for (auto const& services : sub_routes) {
        auto const& [sections_clasz, stop_seq] = key;
        auto const route_idx = tt.register_route(stop_seq, {sections_clasz});
        for (auto const& s : services) {
          auto const& first = trip_data.get(s.trips_.front());

          int train_nr = 0;
          if (is_train_number(first.short_name_)) {
            train_nr = std::stoi(first.short_name_);
          } else if (is_train_number(first.headsign_)) {
            train_nr = std::stoi(first.headsign_);
          }

          external_trip_ids.clear();
          section_directions.clear();
          section_lines.clear();
          auto prev_end = 0U;
          for (auto const [i, t] : utl::enumerate(s.trips_)) {
            auto const& trp = trip_data.get(t);

            trip_id_buf.clear();
            fmt::format_to(trip_id_buf, "{}/{}", train_nr, trp.id_);

            auto const end =
                static_cast<unsigned>(prev_end + trp.stop_seq_.size());
            auto const id = tt.register_trip_id(
                trip_id_buf, src, trp.display_name(tt),
                {source_file_idx, trp.from_line_, trp.to_line_},
                tt.next_transport_idx(), {prev_end, end});
            prev_end = end - 1;

            auto const direction =
                utl::get_or_create(directions, trp.headsign_, [&]() {
                  auto const trip_dir_str =
                      tt.register_trip_direction_string(trp.headsign_);
                  auto const idx = tt.trip_directions_.size();
                  tt.trip_directions_.emplace_back(trip_dir_str);
                  return trip_direction_idx_t{idx};
                });

            auto const line =
                utl::get_or_create(lines, trp.route_->short_name_, [&]() {
                  auto const idx = trip_line_idx_t{tt.trip_lines_.size()};
                  tt.trip_lines_.emplace_back(trp.route_->short_name_);
                  return idx;
                });

            auto const merged_trip = tt.register_merged_trip({id});
            if (s.trips_.size() == 1U) {
              external_trip_ids.push_back(merged_trip);
              section_directions.push_back(direction);
              section_lines.push_back(line);
            } else {
              for (auto section = 0U; section != trp.stop_seq_.size() - 1;
                   ++section) {
                external_trip_ids.push_back(merged_trip);
                section_directions.push_back(direction);
                section_lines.push_back(line);
              }
            }
          }

          tt.add_transport(timetable::transport{
              .bitfield_idx_ = utl::get_or_create(
                  bitfield_indices, s.utc_traffic_days_,
                  [&]() { return tt.register_bitfield(s.utc_traffic_days_); }),
              .route_idx_ = route_idx,
              .external_trip_ids_ = external_trip_ids,
              .section_attributes_ = attributes,
              .section_providers_ = {first.route_->agency_},
              .section_directions_ = section_directions,
              .section_lines_ = section_lines});
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
