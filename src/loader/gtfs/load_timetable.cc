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
#include "nigiri/loader/gtfs/feed_info_test.h"
#include "nigiri/loader/gtfs/files.h"
#include "nigiri/loader/gtfs/flex.h"
#include "nigiri/loader/gtfs/local_to_utc.h"
#include "nigiri/loader/gtfs/noon_offsets.h"
#include "nigiri/loader/gtfs/route.h"
#include "nigiri/loader/gtfs/route_key.h"
#include "nigiri/loader/gtfs/seated.h"
#include "nigiri/loader/gtfs/services.h"
#include "nigiri/loader/gtfs/shape.h"
#include "nigiri/loader/gtfs/shape_prepare.h"
#include "nigiri/loader/gtfs/stop.h"
#include "nigiri/loader/gtfs/stop_seq_number_encoding.h"
#include "nigiri/loader/gtfs/stop_time.h"
#include "nigiri/loader/gtfs/trip.h"
#include "nigiri/loader/loader_interface.h"
#include "nigiri/loader/register.h"

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

  auto const user_script = script_runner{config.user_script_};
  auto const progress_tracker = utl::get_active_progress_tracker();
  auto timezones = tz_map{};
  auto agencies =
      read_agencies(src, tt, timezones, load(kAgencyFile).data(), user_script);
  auto const [stops, seated_transfers] = read_stops(
      src, tt, timezones, load(kStopFile).data(), load(kTransfersFile).data(),
      config.link_stop_distance_, user_script);
  auto const routes =
      read_routes(src, tt, timezones, agencies, load(kRoutesFile).data(),
                  config.default_tz_, user_script);
  auto const calendar = read_calendar(load(kCalenderFile).data());
  auto const dates = read_calendar_date(load(kCalendarDatesFile).data());
  auto const feed_info = read_feed_info(load(kFeedInfoFile).data());
  tt.src_end_date_.push_back(
      feed_info.feed_end_date_.value_or(date::sys_days::max()));
  auto const service = merge_traffic_days(
      tt.internal_interval_days(), calendar, dates,
      config.extend_calendar_ ? feed_info.feed_end_date_ : std::nullopt);
  auto const shape_states =
      (shapes_data != nullptr)
          ? parse_shapes(load(kShapesFile).data(), *shapes_data)
          : shape_loader_state{};
  auto trip_data = read_trips(
      src, tt, routes, service, shape_states, load(kTripsFile).data(),
      config.bikes_allowed_default_, config.cars_allowed_default_, user_script);
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
      if (utl::all_of(t.stop_headsigns_,
                      [&](auto x) { return x == t.headsign_; })) {
        t.stop_headsigns_.clear();
      }
      if (!t.stop_headsigns_.empty()) {
        t.stop_headsigns_.resize(t.seq_numbers_.size(), t.headsign_);
      }
      if (t.requires_sorting_ &&
          (t.event_times_.empty() || t.flex_time_windows_.empty())) {
        if (t.stop_headsigns_.empty()) {
          // without stop headsigns
          std::tie(t.seq_numbers_, t.stop_seq_, t.event_times_,
                   t.flex_time_windows_, t.distance_traveled_) =
              sort_by(t.seq_numbers_, t.stop_seq_, t.event_times_,
                      t.flex_time_windows_, t.distance_traveled_);
        } else {
          // with stop headsigns
          std::tie(t.seq_numbers_, t.stop_seq_, t.event_times_,
                   t.flex_time_windows_, t.stop_headsigns_,
                   t.distance_traveled_) =
              sort_by(t.seq_numbers_, t.stop_seq_, t.event_times_,
                      t.flex_time_windows_, t.stop_headsigns_,
                      t.distance_traveled_);
        }
      }

      if (!t.stop_headsigns_.empty()) {
        t.stop_headsigns_.resize(t.seq_numbers_.size() - 1U);
      }
    }
  }

  {
    auto const timer = scoped_timer{"loader.gtfs.trips.interpolate"};
    for (auto& t : trip_data.data_) {
      t.interpolate();
    }
  }

  {  // Resolve stay-seated transfers (transfer_type=4).
    auto const timer =
        scoped_timer{"loader.gtfs.trips.resolve_seated_transfers"};

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
        to_trip.seated_in_.push_back(
            gtfs_trip_idx_t{&from_trip - trip_data.data_.data()});
        from_trip.seated_out_.push_back(
            gtfs_trip_idx_t{&to_trip - trip_data.data_.data()});
      }
    }
  }

  hash_map<route_key_t, std::vector<std::vector<utc_trip>>, route_key_hash,
           route_key_equals>
      route_services;

  auto const noon_offsets = precompute_noon_offsets(tt, agencies);

  stop_seq_t stop_seq_cache;
  bitvec bikes_allowed_seq_cache;
  auto const get_bikes_allowed_seq =
      [&](basic_string<gtfs_trip_idx_t> const& trips) -> bitvec const* {
    if (trips.size() == 1U) {
      return trip_data.get(trips.front()).bikes_allowed_
                 ? &kSingleTripBikesAllowed
                 : &kSingleTripBikesNotAllowed;
    } else {
      bikes_allowed_seq_cache.resize(0);
      for (auto const [i, t_idx] : utl::enumerate(trips)) {
        auto const& trp = trip_data.get(t_idx);
        auto const stop_count = trp.stop_seq_.size();
        auto const offset = bikes_allowed_seq_cache.size();
        bikes_allowed_seq_cache.resize(
            static_cast<bitvec::size_type>(offset + stop_count - 1));
        for (auto j = 0U; j < stop_count - 1; ++j) {
          bikes_allowed_seq_cache.set(offset + j, trp.bikes_allowed_);
        }
      }
      return &bikes_allowed_seq_cache;
    }
  };

  bitvec cars_allowed_seq_cache;
  auto const get_cars_allowed_seq =
      [&](basic_string<gtfs_trip_idx_t> const& trips) -> bitvec const* {
    if (trips.size() == 1U) {
      return trip_data.get(trips.front()).cars_allowed_
                 ? &kSingleTripBikesAllowed
                 : &kSingleTripBikesNotAllowed;
    } else {
      cars_allowed_seq_cache.resize(0);
      for (auto const [i, t_idx] : utl::enumerate(trips)) {
        auto const& trp = trip_data.get(t_idx);
        auto const stop_count = trp.stop_seq_.size();
        auto const offset = cars_allowed_seq_cache.size();
        cars_allowed_seq_cache.resize(
            static_cast<bitvec::size_type>(offset + stop_count - 1));
        for (auto j = 0U; j < stop_count - 1; ++j) {
          cars_allowed_seq_cache.set(offset + j, trp.cars_allowed_);
        }
      }
      return &cars_allowed_seq_cache;
    }
  };

  auto const add_expanded_trip = [&](utc_trip&& s) {
    auto const* stop_seq = get_stop_seq(trip_data, s, stop_seq_cache);
    auto const clasz = trip_data.get(s.trips_.front()).get_clasz(tt);
    auto const* bikes_allowed_seq = get_bikes_allowed_seq(s.trips_);
    auto const* cars_allowed_seq = get_cars_allowed_seq(s.trips_);
    auto const it = route_services.find(
        route_key_ptr_t{clasz, stop_seq, bikes_allowed_seq, cars_allowed_seq});
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
          route_key_t{clasz, *stop_seq, *bikes_allowed_seq, *cars_allowed_seq},
          std::vector<std::vector<utc_trip>>{{s}});
    }
  };

  auto const add_trip = [&](basic_string<gtfs_trip_idx_t> const& trips,
                            bitfield const* traffic_days) {
    expand_trip(trip_data, noon_offsets, tt, trips, traffic_days,
                tt.date_range_, assistance, add_expanded_trip);
  };

  {
    progress_tracker->status("Expand Trips")
        .out_bounds(68.F, 83.F)
        .in_high(trip_data.data_.size());
    auto const timer = scoped_timer{"loader.gtfs.trips.expand"};

    for (auto const [i, t] : utl::enumerate(trip_data.data_)) {
      if (t.block_ != nullptr || t.has_seated_transfers() ||
          !t.flex_time_windows_.empty()) {
        continue;
      }
      if (t.trip_idx_ != trip_idx_t::invalid()) {
        add_trip({gtfs_trip_idx_t{i}}, t.service_);
        progress_tracker->increment();
      }
    }
  }

  {
    progress_tracker->status("Stay Seated")
        .out_bounds(83.F, 85.F)
        .in_high(route_services.size());
    auto const timer = scoped_timer{"loader.gtfs.trips.block_id"};

    for (auto const& [_, blk] : trip_data.blocks_) {
      // If a trip has both block_id and transfer_type=4
      // -> prefer transfer_type=4, ignore block_id
      if (utl::any_of(blk->trips_, [&](gtfs_trip_idx_t const idx) {
            return trip_data.data_[idx].has_seated_transfers();
          })) {
        for (auto const& trip : blk->trips_) {
          auto const& trp = trip_data.get(trip);
          if (!trp.has_seated_transfers()) {
            // One of the block_id trips has no stay-seated transfer.
            // -> build it separately
            add_trip({trip}, trp.service_);
          }
        }
        continue;
      }

      for (auto const& [trips, traffic_days] : blk->rule_services(trip_data)) {
        add_trip(trips, &traffic_days);
      }
    }
  }

  {
    auto stop_seq_numbers = basic_string<stop_idx_t>{};
    auto const source_file_idx =
        tt.register_source_file((d.path() / kStopTimesFile).generic_string());
    for (auto& trp : trip_data.data_) {
      encode_seq_numbers(trp.seq_numbers_, stop_seq_numbers);

      tt.trip_debug_.emplace_back().emplace_back(
          trip_debug{source_file_idx, trp.from_line_, trp.to_line_});
      tt.trip_stop_seq_numbers_.emplace_back(trp.seq_numbers_);
    }
  }

  {
    auto const timer = scoped_timer{"loader.gtfs.write_seated"};
    auto expanded_seated = expand_seated_trips(
        trip_data, [&](gtfs_trip_idx_t const i, auto&& consume) {
          expand_trip(trip_data, noon_offsets, tt, {i},
                      trip_data.get(i).service_, tt.date_range_, assistance,
                      [&](utc_trip&& s) { consume(std::move(s)); });
        });
    build_seated_trips(tt, trip_data, expanded_seated, add_expanded_trip);
  }

  {
    auto const timer = scoped_timer{"loader.gtfs.write_trips"};

    progress_tracker->status("Write Trips")
        .out_bounds(85.F, 97.F)
        .in_high(route_services.size());

    auto const attributes = basic_string<attribute_combination_idx_t>{};
    auto lines = hash_map<std::string, trip_line_idx_t>{};
    auto section_directions = basic_string<trip_direction_idx_t>{};
    auto section_lines = basic_string<trip_line_idx_t>{};
    auto route_colors = basic_string<route_color>{};
    auto external_trip_ids = basic_string<merged_trips_idx_t>{};
    auto location_routes = mutable_fws_multimap<location_idx_t, route_idx_t>{};
    for (auto const& [key, sub_routes] : route_services) {
      for (auto const& services : sub_routes) {
        auto const route_idx = tt.register_route(
            key.stop_seq_, {key.clasz_}, key.bikes_allowed_, key.cars_allowed_);

        for (auto const& s : key.stop_seq_) {
          auto s_routes = location_routes[stop{s}.location_idx()];
          if (s_routes.empty() || s_routes.back() != route_idx) {
            s_routes.emplace_back(route_idx);
          }
        }

        for (auto const& s : services) {
          auto const& first = trip_data.get(s.trips_.front());

          external_trip_ids.clear();
          section_directions.clear();
          section_lines.clear();
          route_colors.clear();
          auto prev_end = std::uint16_t{0U};
          for (auto const [i, t] : utl::enumerate(s.trips_)) {
            auto& trp = trip_data.get(t);

            auto const end =
                static_cast<std::uint16_t>(prev_end + trp.stop_seq_.size());

            trp.transport_ranges_.emplace_back(
                transport_range_t{tt.next_transport_idx(), {prev_end, end}});
            prev_end = end - 1;

            auto const line =
                utl::get_or_create(lines, trp.route_->short_name_, [&]() {
                  auto const idx = trip_line_idx_t{tt.trip_lines_.size()};
                  tt.trip_lines_.emplace_back(trp.route_->short_name_);
                  return idx;
                });

            auto const merged_trip = tt.register_merged_trip({trp.trip_idx_});
            if (s.trips_.size() == 1U) {
              external_trip_ids.push_back(merged_trip);
              if (trp.stop_headsigns_.empty()) {
                section_directions.push_back(trp.headsign_);
              } else {
                section_directions.insert(std::end(section_directions),
                                          std::begin(trp.stop_headsigns_),
                                          std::end(trp.stop_headsigns_));
              }
              section_lines.push_back(line);
              route_colors.push_back(
                  {trp.route_->color_, trp.route_->text_color_});
            } else {
              for (auto section = 0U; section != trp.stop_seq_.size() - 1;
                   ++section) {
                external_trip_ids.push_back(merged_trip);
                section_directions.push_back(
                    trp.stop_headsigns_.empty()
                        ? trp.headsign_
                        : trp.stop_headsigns_.at(section));
                section_lines.push_back(line);
                route_colors.push_back(
                    {trp.route_->color_, trp.route_->text_color_});
              }
            }
          }

          assert(s.first_dep_offset_.count() >= -1);
          tt.add_transport(timetable::transport{
              .bitfield_idx_ = utl::get_or_create(
                  bitfield_indices, s.utc_traffic_days_,
                  [&]() { return tt.register_bitfield(s.utc_traffic_days_); }),
              .route_idx_ = route_idx,
              .first_dep_offset_ = {s.first_dep_offset_, s.tz_offset_},
              .external_trip_ids_ = external_trip_ids,
              .section_attributes_ = attributes,
              .section_providers_ = {first.route_->agency_},
              .section_directions_ = section_directions,
              .section_lines_ = section_lines,
              .route_colors_ = route_colors});
        }

        tt.finish_route();

        auto const stop_times_begin = tt.route_stop_times_.size();
        for (auto const [from, to] :
             utl::pairwise(interval{std::size_t{0U}, key.stop_seq_.size()})) {
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
