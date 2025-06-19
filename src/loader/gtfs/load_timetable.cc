#include "nigiri/loader/gtfs/load_timetable.h"

#include <charconv>
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
#include "nigiri/loader/gtfs/fares.h"
#include "nigiri/loader/gtfs/files.h"
#include "nigiri/loader/gtfs/flex.h"
#include "nigiri/loader/gtfs/local_to_utc.h"
#include "nigiri/loader/gtfs/noon_offsets.h"
#include "nigiri/loader/gtfs/route.h"
#include "nigiri/loader/gtfs/route_key.h"
#include "nigiri/loader/gtfs/services.h"
#include "nigiri/loader/gtfs/shape.h"
#include "nigiri/loader/gtfs/shape_prepare.h"
#include "nigiri/loader/gtfs/stop.h"
#include "nigiri/loader/gtfs/stop_seq_number_encoding.h"
#include "nigiri/loader/gtfs/stop_time.h"
#include "nigiri/loader/gtfs/trip.h"
#include "nigiri/loader/loader_interface.h"
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

void load_timetable(loader_config const& config,
                    source_idx_t const src,
                    dir const& d,
                    timetable& tt,
                    assistance_times* assistance,
                    shapes_storage* shapes_data) {
  auto local_bitfield_indices = hash_map<bitfield, bitfield_idx_t>{};
  load_timetable(config, src, d, tt, local_bitfield_indices, assistance,
                 shapes_data);
}

void load_timetable(loader_config const& config,
                    source_idx_t const src,
                    dir const& d,
                    timetable& tt,
                    hash_map<bitfield, bitfield_idx_t>& bitfield_indices,
                    assistance_times* assistance,
                    shapes_storage* shapes_data) {
  auto const global_timer = nigiri::scoped_timer{"gtfs parser"};

  auto const load = [&](std::string_view file_name) -> file {
    return d.exists(file_name) ? d.get_file(file_name) : file{};
  };

  auto const progress_tracker = utl::get_active_progress_tracker();
  auto timezones = tz_map{};
  auto agencies = read_agencies(tt, timezones, load(kAgencyFile).data());
  auto const [stops, seated_transfers] =
      read_stops(src, tt, timezones, load(kStopFile).data(),
                 load(kTransfersFile).data(), config.link_stop_distance_);
  auto const routes = read_routes(src, tt, timezones, agencies,
                                  load(kRoutesFile).data(), config.default_tz_);
  auto const calendar = read_calendar(load(kCalenderFile).data());
  auto const dates = read_calendar_date(load(kCalendarDatesFile).data());
  auto const service =
      merge_traffic_days(tt.internal_interval_days(), calendar, dates);
  auto const shape_states =
      (shapes_data != nullptr)
          ? parse_shapes(load(kShapesFile).data(), *shapes_data)
          : shape_loader_state{};
  auto trip_data =
      read_trips(tt, routes, service, shape_states, load(kTripsFile).data(),
                 config.bikes_allowed_default_, config.cars_allowed_default_);
  auto const booking_rules = parse_booking_rules(
      tt, load(kBookingRulesFile).data(), service, bitfield_indices);
  auto const location_groups =
      parse_location_groups(tt, load(kLocationGroupsFile).data());
  auto const flex_areas =
      parse_flex_areas(tt, src, load(kLocationsFile).data());
  parse_location_group_stops(tt, load(kLocationGroupStopsFile).data(),
                             location_groups, stops);
  read_frequencies(trip_data, load(kFrequenciesFile).data());
  read_stop_times(tt, trip_data, stops, flex_areas, booking_rules,
                  location_groups, load(kStopTimesFile).data(),
                  shapes_data != nullptr);
  load_fares(tt, d, service, routes, stops);
  utl::verify(tt.fares_.size() == to_idx(src) + 1U, "fares: size={} src={}",
              tt.fares_.size(), src);

  {
    auto const timer = scoped_timer{"loader.gtfs.trips.sort"};
    for (auto& t : trip_data.data_) {
      if (t.requires_sorting_ &&
          (t.event_times_.empty() || t.flex_time_windows_.empty())) {
        t.stop_headsigns_.resize(t.seq_numbers_.size());
        std::tie(t.seq_numbers_, t.stop_seq_, t.event_times_,
                 t.flex_time_windows_, t.stop_headsigns_,
                 t.distance_traveled_) =
            sort_by(t.seq_numbers_, t.stop_seq_, t.event_times_,
                    t.flex_time_windows_, t.stop_headsigns_,
                    t.distance_traveled_);
      }
    }
  }

  {
    auto const timer = scoped_timer{"loader.gtfs.trips.interpolate"};
    for (auto& t : trip_data.data_) {
      t.interpolate();
    }
  }

  {
    for (auto const& [from_trip_id, to_trip_ids] : seated_transfers) {
      auto const from_it = trip_data.trips_.find(from_trip_id);
      if (from_it == end(trip_data.trips_)) {
        log(log_lvl::error, "nigiri.loader.gtfs.seated", "trip {} not found",
            from_trip_id);
        continue;
      }

      auto& from_trip = trip_data.get(from_trip_id);
      for (auto const& to_trip_id : to_trip_ids) {
        auto const to_it = trip_data.trips_.find(to_trip_id);
        if (to_it == end(trip_data.trips_)) {
          log(log_lvl::error, "nigiri.loader.gtfs.seated", "trip {} not found",
              to_trip_id);
          continue;
        }

        auto& to_trip = trip_data.data_[to_it->second];
        to_trip.seated_out_.push_back(&from_trip);
        from_trip.seated_in_.push_back(&to_trip);
      }

      auto const cmp_stop_seq = [](trip const* a, trip const* b) {
        return a->route_key() < b->route_key();
      };
      for (auto& trp : trip_data.data_) {
        utl::sort(trp.seated_out_, cmp_stop_seq);
        utl::sort(trp.seated_in_, cmp_stop_seq);
      }
    }
  }

  auto route_services = hash_map<
      trip const* /* representative; only trip::route_key() is relevant */,
      std::vector<std::vector<utc_trip>>, route_key_hash, route_key_equals>{};

  auto const noon_offsets = precompute_noon_offsets(tt, agencies);
  auto const add_trip = [&](basic_string<gtfs_trip_idx_t> const& trips,
                            bitfield const* traffic_days) {
    expand_trip(
        trip_data, noon_offsets, tt, trips, traffic_days, tt.date_range_,
        assistance, [&](utc_trip&& s) {
          auto const& t = trip_data.get(s.trip_);
          auto const it = route_services.find(&t);
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
            route_services.emplace(&t, std::vector<std::vector<utc_trip>>{{s}});
          }
        });
  };

  {
    progress_tracker->status("Expand Trips")
        .out_bounds(68.F, 83.F)
        .in_high(trip_data.data_.size());
    auto const timer = scoped_timer{"loader.gtfs.trips.expand"};

    for (auto const [i, t] : utl::enumerate(trip_data.data_)) {
      if (t.block_ != nullptr || !t.flex_time_windows_.empty()) {
        continue;
      }
      add_trip({gtfs_trip_idx_t{i}}, t.service_);
      progress_tracker->increment();
    }
  }

  {
    progress_tracker->status("Stay Seated")
        .out_bounds(83.F, 85.F)
        .in_high(route_services.size());
    auto const timer = scoped_timer{"loader.gtfs.trips.block_id"};

    for (auto const& [_, blk] : trip_data.blocks_) {
      for (auto const& [trips, traffic_days] : blk->rule_services(trip_data)) {
        add_trip(trips, &traffic_days);
      }
    }
  }

  {
    auto const timer = scoped_timer{"loader.gtfs.write_trips"};

    progress_tracker->status("Write Trips")
        .out_bounds(85.F, 96.F)
        .in_high(route_services.size());

    auto const is_train_number = [](auto const& s) {
      return !s.empty() && std::all_of(begin(s), end(s), [](auto&& c) -> bool {
        return std::isdigit(c);
      });
    };

    auto stop_seq_numbers = basic_string<stop_idx_t>{};
    auto const source_file_idx =
        tt.register_source_file((d.path() / kStopTimesFile).generic_string());
    tt.trip_direction_id_.resize(tt.n_trips() + trip_data.data_.size());
    for (auto& trp : trip_data.data_) {

      encode_seq_numbers(trp.seq_numbers_, stop_seq_numbers);
      trp.trip_idx_ = tt.register_trip(
          trp.id_, trp.display_name(), trp.route_->route_id_idx_, src,
          {source_file_idx, trp.from_line_, trp.to_line_}, stop_seq_numbers,
          trp.direction_id_);
    }

    auto location_routes = mutable_fws_multimap<location_idx_t, route_idx_t>{};
    for (auto const& [k, sub_routes] : route_services) {
      auto const& [clasz, stop_seq, bikes_allowed, cars_allowed] =
          k->route_key();
      for (auto const& services : sub_routes) {
        auto const route_idx =
            tt.register_route(stop_seq, clasz, bikes_allowed, cars_allowed);

        for (auto const& s : stop_seq) {
          auto s_routes = location_routes[stop{s}.location_idx()];
          if (s_routes.empty() || s_routes.back() != route_idx) {
            s_routes.emplace_back(route_idx);
          }
        }

        for (auto const& s : services) {
          auto const& trp = trip_data.get(s.trip_);
          tt.add_transport(timetable::transport{
              .trip_ = trp.trip_idx_,
              .bitfield_ = utl::get_or_create(
                  bitfield_indices, s.utc_traffic_days_,
                  [&]() { return tt.register_bitfield(s.utc_traffic_days_); }),
              .route_ = route_idx,
              .first_dep_offset_ = s.first_dep_offset_});
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

    if (shapes_data != nullptr) {
      calculate_shape_offsets_and_bboxes(tt, *shapes_data, shape_states,
                                         trip_data.data_);
    }

    // Build location_routes map
    for (auto l = tt.location_routes_.size(); l != tt.n_locations(); ++l) {
      tt.location_routes_.emplace_back(location_routes[location_idx_t{l}]);
      assert(tt.location_routes_.size() == l + 1U);
    }

    // Build transport ranges.
    for (auto const& t : trip_data.data_) {
      tt.trip_transport_ranges_.emplace_back(t.transport_ranges_);
    }
  }

  {
    progress_tracker->status("Flex")
        .out_bounds(97.F, 98.F)
        .in_high(route_services.size());

    auto const timer = scoped_timer{"loader.gtfs.write_flex"};

    auto stop_seq = stop_seq_map_t{};
    for (auto const& trp : trip_data.data_) {
      if (trp.flex_time_windows_.empty()) {
        continue;
      }
      expand_flex_trip(tt, bitfield_indices, stop_seq, noon_offsets,
                       tt.date_range_, trp);
    }
  }
}

}  // namespace nigiri::loader::gtfs
