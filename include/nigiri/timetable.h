#pragma once

#include <compare>
#include <filesystem>
#include <span>
#include <type_traits>

#include "cista/memory_holder.h"
#include "cista/reflection/printable.h"

#include "utl/verify.h"
#include "utl/zip.h"

#include "geo/box.h"
#include "geo/latlng.h"

#include "nigiri/common/interval.h"
#include "nigiri/fares.h"
#include "nigiri/footpath.h"
#include "nigiri/location.h"
#include "nigiri/logging.h"
#include "nigiri/stop.h"
#include "nigiri/string_store.h"
#include "nigiri/td_footpath.h"
#include "nigiri/types.h"

namespace nigiri {

struct timetable {
  struct locations {
    timezone_idx_t register_timezone(timezone tz) {
      auto const idx = timezone_idx_t{
          static_cast<timezone_idx_t::value_t>(timezones_.size())};
      timezones_.emplace_back(std::move(tz));
      return idx;
    }

    location_idx_t register_location(location const& l) {
      auto const next_idx = static_cast<location_idx_t::value_t>(names_.size());
      auto const l_idx = location_idx_t{next_idx};
      auto const [it, is_new] = location_id_to_idx_.emplace(
          location_id{.id_ = l.id_, .src_ = l.src_}, l_idx);

      if (is_new) {
        names_.emplace_back(l.name_);
        coordinates_.emplace_back(l.pos_);
        ids_.emplace_back(l.id_);
        src_.emplace_back(l.src_);
        types_.emplace_back(l.type_);
        location_timezones_.emplace_back(l.timezone_idx_);
        equivalences_.emplace_back();
        children_.emplace_back();
        preprocessing_footpaths_out_.emplace_back();
        preprocessing_footpaths_in_.emplace_back();
        transfer_time_.emplace_back(l.transfer_time_);
        parents_.emplace_back(l.parent_);
      } else {
        log(log_lvl::error, "timetable.register_location",
            "duplicate station {}", l.id_);
      }

      assert(names_.size() == next_idx + 1);
      assert(coordinates_.size() == next_idx + 1);
      assert(ids_.size() == next_idx + 1);
      assert(src_.size() == next_idx + 1);
      assert(types_.size() == next_idx + 1);
      assert(location_timezones_.size() == next_idx + 1);
      assert(equivalences_.size() == next_idx + 1);
      assert(children_.size() == next_idx + 1);
      assert(preprocessing_footpaths_out_.size() == next_idx + 1);
      assert(preprocessing_footpaths_in_.size() == next_idx + 1);
      assert(transfer_time_.size() == next_idx + 1);
      assert(parents_.size() == next_idx + 1);

      return it->second;
    }

    location get(location_idx_t const idx) const {
      auto l = location{ids_[idx].view(),
                        names_[idx].view(),
                        coordinates_[idx],
                        src_[idx],
                        types_[idx],
                        parents_[idx],
                        location_timezones_[idx],
                        transfer_time_[idx],
                        it_range{equivalences_[idx]}};
      l.l_ = idx;
      return l;
    }

    location get(location_id const& id) const {
      return get(location_id_to_idx_.at(id));
    }

    std::optional<location> find(location_id const& id) const {
      auto const it = location_id_to_idx_.find(id);
      return it == end(location_id_to_idx_) ? std::nullopt
                                            : std::optional{get(it->second)};
    }

    // Station access: external station id -> internal station idx
    hash_map<location_id, location_idx_t> location_id_to_idx_;
    vecvec<location_idx_t, char> names_;
    vecvec<location_idx_t, char> ids_;
    vector_map<location_idx_t, geo::latlng> coordinates_;
    vector_map<location_idx_t, source_idx_t> src_;
    vector_map<location_idx_t, u8_minutes> transfer_time_;
    vector_map<location_idx_t, location_type> types_;
    vector_map<location_idx_t, location_idx_t> parents_;
    vector_map<location_idx_t, timezone_idx_t> location_timezones_;
    mutable_fws_multimap<location_idx_t, location_idx_t> equivalences_;
    mutable_fws_multimap<location_idx_t, location_idx_t> children_;
    mutable_fws_multimap<location_idx_t, footpath> preprocessing_footpaths_out_;
    mutable_fws_multimap<location_idx_t, footpath> preprocessing_footpaths_in_;
    array<vecvec<location_idx_t, footpath>, kMaxProfiles> footpaths_out_;
    array<vecvec<location_idx_t, footpath>, kMaxProfiles> footpaths_in_;
    vector_map<timezone_idx_t, timezone> timezones_;
    rtree<location_idx_t> rtree_;
  } locations_;

  struct transport {
    bitfield_idx_t bitfield_idx_;
    route_idx_t route_idx_;
    duration_t first_dep_offset_;
    basic_string<merged_trips_idx_t> const& external_trip_ids_;
    basic_string<attribute_combination_idx_t> const& section_attributes_;
    basic_string<provider_idx_t> const& section_providers_;
    basic_string<trip_direction_idx_t> const& section_directions_;
    basic_string<trip_line_idx_t> const& section_lines_;
    basic_string<stop_idx_t> const& stop_seq_numbers_;
    basic_string<route_color> const& route_colors_;
  };

  template <typename TripId>
  trip_idx_t register_trip_id(TripId const& trip_id_str,
                              route_id_idx_t const route_id_idx,
                              source_idx_t const src,
                              std::string const& display_name,
                              trip_debug const dbg,
                              std::uint32_t const train_nr,
                              std::span<stop_idx_t> seq_numbers,
                              direction_id_t const direction_id) {
    auto const trip_idx = trip_idx_t{trip_ids_.size()};

    auto const trip_id_idx = trip_id_idx_t{trip_id_strings_.size()};

    if (route_id_idx != route_id_idx_t::invalid()) {  // HRD
      route_ids_[src].route_id_trips_[route_id_idx].push_back(trip_idx);
      trip_direction_id_.set(trip_idx, direction_id == direction_id_t{1U});
    }
    trip_route_id_.emplace_back(route_id_idx);

    trip_id_strings_.emplace_back(trip_id_str);
    trip_id_src_.emplace_back(src);

    trip_id_to_idx_.emplace_back(trip_id_idx, trip_idx);
    trip_display_names_.emplace_back(display_name);
    trip_debug_.emplace_back().emplace_back(dbg);
    trip_ids_.emplace_back().emplace_back(trip_id_idx);
    trip_train_nr_.emplace_back(train_nr);
    trip_stop_seq_numbers_.emplace_back(seq_numbers);

    return trip_idx;
  }

  void resolve();

  bitfield_idx_t register_bitfield(bitfield const& b) {
    auto const idx = bitfield_idx_t{bitfields_.size()};
    bitfields_.emplace_back(b);
    return idx;
  }

  template <typename T>
  trip_direction_string_idx_t register_trip_direction_string(T&& s) {
    auto const idx =
        trip_direction_string_idx_t{trip_direction_strings_.size()};
    trip_direction_strings_.emplace_back(s);
    return idx;
  }

  route_idx_t register_route(basic_string<stop::value_type> const& stop_seq,
                             basic_string<clasz> const& clasz_sections,
                             bitvec const& bikes_allowed_per_section) {
    assert(stop_seq.size() > 1U);
    assert(!clasz_sections.empty());

    auto const idx = route_location_seq_.size();
    route_transport_ranges_.emplace_back(
        transport_idx_t{transport_traffic_days_.size()},
        transport_idx_t::invalid());
    route_location_seq_.emplace_back(stop_seq);
    route_section_clasz_.emplace_back(clasz_sections);
    route_clasz_.emplace_back(clasz_sections[0]);

    auto const sections = bikes_allowed_per_section.size();
    auto const sections_with_bikes_allowed = bikes_allowed_per_section.count();
    auto const bikes_allowed_on_all_sections =
        sections_with_bikes_allowed == sections && sections != 0;
    auto const bikes_allowed_on_some_sections =
        sections_with_bikes_allowed != 0U;
    route_bikes_allowed_.resize(route_bikes_allowed_.size() + 2U);
    route_bikes_allowed_.set(idx * 2, bikes_allowed_on_all_sections);
    route_bikes_allowed_.set(idx * 2 + 1, bikes_allowed_on_some_sections);

    route_bikes_allowed_per_section_.resize(idx + 1);
    if (bikes_allowed_on_some_sections && !bikes_allowed_on_all_sections) {
      auto bucket = route_bikes_allowed_per_section_[route_idx_t{idx}];
      for (auto i = 0U; i < bikes_allowed_per_section.size(); ++i) {
        bucket.push_back(bikes_allowed_per_section[i]);
      }
    }

    return route_idx_t{idx};
  }

  void finish_route() {
    route_transport_ranges_.back().to_ =
        transport_idx_t{transport_traffic_days_.size()};
  }

  merged_trips_idx_t register_merged_trip(
      basic_string<trip_idx_t> const& trip_ids) {
    auto const idx = merged_trips_.size();
    merged_trips_.emplace_back(trip_ids);
    return merged_trips_idx_t{static_cast<merged_trips_idx_t::value_t>(idx)};
  }

  source_file_idx_t register_source_file(std::string_view path) {
    auto const idx = source_file_idx_t{source_file_names_.size()};
    source_file_names_.emplace_back(path);
    return idx;
  }

  provider_idx_t register_provider(provider&& p) {
    auto const idx = providers_.size();
    providers_.emplace_back(std::move(p));
    provider_id_to_idx_.emplace_back(idx);
    return provider_idx_t{idx};
  }

  void add_transport(transport&& t) {
    transport_first_dep_offset_.emplace_back(t.first_dep_offset_);
    transport_traffic_days_.emplace_back(t.bitfield_idx_);
    transport_route_.emplace_back(t.route_idx_);
    transport_to_trip_section_.emplace_back(t.external_trip_ids_);
    transport_section_attributes_.emplace_back(t.section_attributes_);
    transport_section_providers_.emplace_back(t.section_providers_);
    transport_section_directions_.emplace_back(t.section_directions_);
    transport_section_lines_.emplace_back(t.section_lines_);
    transport_section_route_colors_.emplace_back(t.route_colors_);

    assert(transport_traffic_days_.size() == transport_route_.size());
    assert(transport_traffic_days_.size() == transport_to_trip_section_.size());
    assert(transport_section_directions_.back().size() == 0U ||
           transport_section_directions_.back().size() == 1U ||
           transport_section_directions_.back().size() ==
               route_location_seq_.at(transport_route_.back()).size() - 1U);
  }

  transport_idx_t next_transport_idx() const {
    return transport_idx_t{transport_traffic_days_.size()};
  }

  std::span<delta const> event_times_at_stop(route_idx_t const r,
                                             stop_idx_t const stop_idx,
                                             event_type const ev_type) const {
    auto const n_transports =
        static_cast<unsigned>(route_transport_ranges_[r].size());
    auto const idx = static_cast<unsigned>(
        route_stop_time_ranges_[r].from_ +
        n_transports * (stop_idx * 2 - (ev_type == event_type::kArr ? 1 : 0)));
    return std::span<delta const>{&route_stop_times_[idx], n_transports};
  }

  delta event_mam(route_idx_t const r,
                  transport_idx_t t,
                  stop_idx_t const stop_idx,
                  event_type const ev_type) const {
    auto const range = route_transport_ranges_[r];
    auto const n_transports = static_cast<unsigned>(range.size());
    auto const route_stop_begin = static_cast<unsigned>(
        route_stop_time_ranges_[r].from_ +
        n_transports * (stop_idx * 2 - (ev_type == event_type::kArr ? 1 : 0)));
    auto const t_idx_in_route = to_idx(t) - to_idx(range.from_);
    return route_stop_times_[route_stop_begin + t_idx_in_route];
  }

  delta event_mam(transport_idx_t t,
                  stop_idx_t const stop_idx,
                  event_type const ev_type) const {
    return event_mam(transport_route_[t], t, stop_idx, ev_type);
  }

  unixtime_t event_time(nigiri::transport t,
                        stop_idx_t const stop_idx,
                        event_type const ev_type) const {
    return unixtime_t{internal_interval_days().from_ + to_idx(t.day_) * 1_days +
                      event_mam(t.t_idx_, stop_idx, ev_type).as_duration()};
  }

  day_idx_t day_idx(date::year_month_day const day) const {
    return day_idx(date::sys_days{day});
  }

  day_idx_t day_idx(date::sys_days const day) const {
    return day_idx_t{(day - (date_range_.from_ - kTimetableOffset)).count()};
  }

  std::pair<day_idx_t, minutes_after_midnight_t> day_idx_mam(
      unixtime_t const t) const {
    auto const minutes_since_timetable_begin =
        (t - internal_interval().from_).count();
    auto const d =
        static_cast<day_idx_t::value_t>(minutes_since_timetable_begin / 1440);
    auto const m = minutes_since_timetable_begin % 1440;
    return {day_idx_t{d}, minutes_after_midnight_t{m}};
  }

  unixtime_t to_unixtime(day_idx_t const d,
                         minutes_after_midnight_t const m = 0_minutes) const {
    return internal_interval_days().from_ + to_idx(d) * 1_days + m;
  }

  cista::base_t<trip_idx_t> n_trips() const {
    return trip_display_names_.size();
  }

  cista::base_t<location_idx_t> n_locations() const {
    return locations_.names_.size();
  }

  cista::base_t<route_idx_t> n_routes() const {
    return route_location_seq_.size();
  }

  cista::base_t<source_idx_t> n_sources() const {
    return source_file_names_.size();
  }

  cista::base_t<provider_idx_t> n_agencies() const { return providers_.size(); }

  interval<unixtime_t> external_interval() const {
    return {std::chrono::time_point_cast<i32_minutes>(date_range_.from_),
            std::chrono::time_point_cast<i32_minutes>(date_range_.to_)};
  }

  interval<date::sys_days> internal_interval_days() const {
    return {date_range_.from_ - kTimetableOffset,
            date_range_.to_ + date::days{1}};
  }

  constexpr interval<unixtime_t> internal_interval() const {
    return {
        std::chrono::time_point_cast<i32_minutes>(date_range_.from_ -
                                                  kTimetableOffset),
        std::chrono::time_point_cast<i32_minutes>(date_range_.to_ + 1_days)};
  }

  std::string_view trip_direction(trip_direction_idx_t const i) const {
    return trip_directions_.at(i).apply(
        utl::overloaded{[&](trip_direction_string_idx_t s_idx) {
                          return trip_direction_strings_.at(s_idx).view();
                        },
                        [&](location_idx_t const l) {
                          return locations_.names_.at(l).view();
                        }});
  }

  std::string_view transport_name(transport_idx_t const t) const {
    return trip_display_names_
        [merged_trips_[transport_to_trip_section_[t].front()].front()]
            .view();
  }

  debug dbg(transport_idx_t const t) const {
    auto const trip_idx =
        merged_trips_[transport_to_trip_section_[t].front()].front();
    return debug{
        source_file_names_[trip_debug_[trip_idx].front().source_file_idx_]
            .view(),
        trip_debug_[trip_idx].front().line_number_from_,
        trip_debug_[trip_idx].front().line_number_to_};
  }

  friend std::ostream& operator<<(std::ostream&, timetable const&);

  void write(cista::memory_holder&) const;
  void write(std::filesystem::path const&) const;
  static cista::wrapped<timetable> read(std::filesystem::path const&);

  // Schedule range.
  interval<date::sys_days> date_range_;

  // Trip access: external trip id -> internal trip index
  vector<pair<trip_id_idx_t, trip_idx_t>> trip_id_to_idx_;

  // Trip index -> list of external trip ids
  mutable_fws_multimap<trip_idx_t, trip_id_idx_t> trip_ids_;

  // Storage for trip id strings + source
  vecvec<trip_id_idx_t, char> trip_id_strings_;
  vector_map<trip_id_idx_t, source_idx_t> trip_id_src_;

  // Trip -> direction (valid options 0 or 1)
  bitvec_map<trip_idx_t> trip_direction_id_;

  // Trip train number, if available (otherwise 0)
  vector_map<trip_id_idx_t, std::uint32_t> trip_train_nr_;

  // Trip -> route name
  vector_map<trip_idx_t, route_id_idx_t> trip_route_id_;

  // External route id
  struct route_ids {
    route_id_idx_t add(std::string_view id,
                       std::string_view short_name,
                       std::string_view long_name,
                       provider_idx_t const provider,
                       std::uint16_t const type) {
      auto const idx = ids_.store(id);
      route_id_short_names_.emplace_back(short_name);
      route_id_long_names_.emplace_back(long_name);
      route_id_type_.emplace_back(type);
      route_id_provider_.emplace_back(provider);
      route_id_trips_.emplace_back(std::initializer_list<trip_idx_t>{});
      return idx;
    }

    vecvec<route_id_idx_t, char> route_id_short_names_;
    vecvec<route_id_idx_t, char> route_id_long_names_;
    vector_map<route_id_idx_t, route_type_t> route_id_type_;
    vector_map<route_id_idx_t, provider_idx_t> route_id_provider_;
    paged_vecvec<route_id_idx_t, trip_idx_t> route_id_trips_;
    string_store<route_id_idx_t> ids_;
  };
  vector_map<source_idx_t, route_ids> route_ids_;

  // Trip index -> all transports with a stop interval
  paged_vecvec<trip_idx_t, transport_range_t> trip_transport_ranges_;

  // Transport -> stop sequence numbers (relevant for GTFS-RT stop matching)
  // Compaction:
  // - empty = zero-based sequence 0,1,2,...
  // - only one '1' entry = one-based sequence 1,2,3,...
  // - only one '10' entry = 10-based sequence 10,20,30,...
  // - more than one entry: exact sequence number for each stop
  vecvec<trip_idx_t, stop_idx_t> trip_stop_seq_numbers_;

  // Trip -> debug info
  mutable_fws_multimap<trip_idx_t, trip_debug> trip_debug_;
  vecvec<source_file_idx_t, char> source_file_names_;

  // Trip index -> display name
  vecvec<trip_idx_t, char> trip_display_names_;

  // Route -> range of transports in this route (from/to transport_idx_t)
  vector_map<route_idx_t, interval<transport_idx_t>> route_transport_ranges_;

  // Route -> list of stops
  vecvec<route_idx_t, stop::value_type> route_location_seq_;

  // Route -> clasz
  vector_map<route_idx_t, clasz> route_clasz_;

  // Route -> clasz per section
  vecvec<route_idx_t, clasz> route_section_clasz_;

  // Route * 2 -> bikes allowed along the route
  // Route * 2 + 1 -> bikes along parts of the route
  bitvec route_bikes_allowed_;

  // Route -> bikes allowed per section
  // Only set for routes where the entry in route_bikes_allowed_bitvec_
  // is set to "bikes along parts of the route"
  vecvec<route_idx_t, bool> route_bikes_allowed_per_section_;

  // Location -> list of routes
  vecvec<location_idx_t, route_idx_t> location_routes_;

  // Route 1:
  //   stop-1-dep: [trip1, trip2, ..., tripN]
  //   stop-2-arr: [trip1, trip2, ..., tripN]
  //   ...
  // Route 2:
  //  stop-1-dep: [...]
  // ...
  // RouteN: ...
  vector_map<route_idx_t, interval<std::uint32_t>> route_stop_time_ranges_;
  vector<delta> route_stop_times_;

  // Offset between the stored time and the time given in the GTFS timetable.
  // Required to match GTFS-RT with GTFS-static trips.
  vector_map<transport_idx_t, duration_t> transport_first_dep_offset_;

  // Services in GTFS can start with a first departure time > 24:00:00
  // The loader transforms this into a time <24:00:00 and shifts the bits in the
  // bitset accordingly. To still be able to match the traffic day from the
  // corresponding service_id, it's necessary to store the number of days which
  // is floor(stop_times.txt:departure_time/1440)
  vector_map<transport_idx_t, std::uint8_t> initial_day_offset_;

  // Trip index -> traffic day bitfield
  vector_map<transport_idx_t, bitfield_idx_t> transport_traffic_days_;

  // Unique bitfields
  vector_map<bitfield_idx_t, bitfield> bitfields_;

  // For each trip the corresponding route
  vector_map<transport_idx_t, route_idx_t> transport_route_;

  // Trip index -> merged trips
  vecvec<transport_idx_t, merged_trips_idx_t> transport_to_trip_section_;

  // Merged trips info
  vecvec<merged_trips_idx_t, trip_idx_t> merged_trips_;

  // Section meta infos:
  vector_map<attribute_idx_t, attribute> attributes_;
  vecvec<attribute_combination_idx_t, attribute_idx_t> attribute_combinations_;
  vector_map<provider_idx_t, provider> providers_;
  vector<provider_idx_t> provider_id_to_idx_;
  vecvec<trip_direction_string_idx_t, char> trip_direction_strings_;
  vector_map<trip_direction_idx_t, trip_direction_t> trip_directions_;
  vecvec<trip_line_idx_t, char> trip_lines_;

  // Transport to section meta infos; Compaction:
  // - only one value = value is valid for the whole run
  // - multiple values = one value for each section
  vecvec<transport_idx_t, attribute_combination_idx_t>
      transport_section_attributes_;
  vecvec<transport_idx_t, provider_idx_t> transport_section_providers_;
  vecvec<transport_idx_t, trip_direction_idx_t> transport_section_directions_;
  vecvec<transport_idx_t, trip_line_idx_t> transport_section_lines_;
  vecvec<transport_idx_t, route_color> transport_section_route_colors_;

  // Lower bound graph.
  vecvec<location_idx_t, footpath> fwd_search_lb_graph_;
  vecvec<location_idx_t, footpath> bwd_search_lb_graph_;

  // profile name -> profile_idx_t
  hash_map<string, profile_idx_t> profiles_;

  // Fares
  vector_map<source_idx_t, fares> fares_;
  vector_map<area_idx_t, area> areas_;
  vecvec<location_idx_t, area_idx_t> location_areas_;
  string_store<string_idx_t> strings_;

  // Flex
  paged_vecvec<location_group_idx_t, location_idx_t> location_group_;
  vector_map<location_group_idx_t, string_idx_t> location_group_name_;
  vector_map<flex_area_idx_t, geo::box> flex_area_bbox_;
  vecvec<flex_area_idx_t, location_idx_t> flex_area_locations_;
  nvec<flex_area_idx_t, geo::latlng, 2U> flex_area_outers_;
  nvec<flex_area_idx_t, geo::latlng, 3U> flex_area_inners_;
  rtree<flex_area_idx_t> flex_area_rtree_;
  vector_map<flex_transport_idx_t, bitfield_idx_t> flex_traffic_days_;
  vecvec<flex_transport_idx_t, interval<duration_t>> flex_stop_time_windows_;
  vecvec<flex_transport_idx_t, flex_stop_t> flex_stops_;
  vecvec<flex_transport_idx_t, booking_rule_idx_t> flex_pickup_booking_rule_;
  vecvec<flex_transport_idx_t, booking_rule_idx_t> flex_drop_off_booking_rule_;
  vector_map<booking_rule_idx_t, booking_rule> booking_rules_;
};

}  // namespace nigiri