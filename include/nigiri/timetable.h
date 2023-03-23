#pragma once

#include <compare>
#include <filesystem>
#include <span>
#include <type_traits>

#include "cista/memory_holder.h"
#include "cista/reflection/printable.h"

#include "utl/verify.h"
#include "utl/zip.h"

#include "geo/latlng.h"

#include "nigiri/common/interval.h"
#include "nigiri/footpath.h"
#include "nigiri/location.h"
#include "nigiri/logging.h"
#include "nigiri/types.h"

namespace nigiri {

struct timetable {
  struct stop {
    using value_type = location_idx_t::value_t;

    stop(location_idx_t::value_t const val) {
      std::memcpy(this, &val, sizeof(value_type));
    }

    stop(location_idx_t const location,
         bool const in_allowed,
         bool const out_allowed)
        : location_{location},
          in_allowed_{in_allowed ? 1U : 0U},
          out_allowed_{out_allowed ? 1U : 0U} {}

    location_idx_t location_idx() const { return location_idx_t{location_}; }
    bool in_allowed() const { return in_allowed_ != 0U; }
    bool out_allowed() const { return out_allowed_ != 0U; }

    cista::hash_t hash() const {
      return cista::hash_combine(cista::BASE_HASH, value());
    }

    location_idx_t::value_t value() const {
      return *reinterpret_cast<location_idx_t::value_t const*>(this);
    }

    friend auto operator<=>(stop const&, stop const&) = default;

    location_idx_t::value_t location_ : 30;
    location_idx_t::value_t in_allowed_ : 1;
    location_idx_t::value_t out_allowed_ : 1;
  };
  static_assert(sizeof(stop) == sizeof(location_idx_t));

  struct locations {
    timezone_idx_t register_timezone(timezone tz) {
      auto const idx = timezone_idx_t{
          static_cast<timezone_idx_t::value_t>(timezones_.size())};
      timezones_.emplace_back(std::move(tz));
      return idx;
    }

    location_idx_t register_location(location&& l) {
      auto const next_idx =
          location_idx_t{static_cast<location_idx_t::value_t>(names_.size())};
      auto const [it, is_new] = location_id_to_idx_.emplace(
          location_id{.id_ = l.id_, .src_ = l.src_}, next_idx);

      if (is_new) {
        names_.emplace_back(l.name_);
        coordinates_.emplace_back(l.pos_);
        ids_.emplace_back(l.id_);
        src_.emplace_back(l.src_);
        types_.emplace_back(l.type_);
        location_timezones_.emplace_back(l.timezone_idx_);
        equivalences_.emplace_back();
        children_.emplace_back();
        footpaths_out_.emplace_back();
        footpaths_in_.emplace_back();
        transfer_time_.emplace_back(l.transfer_time_);  // TODO(felix)
        osm_ids_.emplace_back(osm_node_id_t::invalid());  // TODO(felix)
        parents_.emplace_back(l.parent_);
      }

      assert(names_.size() == next_idx + 1);
      assert(coordinates_.size() == next_idx + 1);
      assert(ids_.size() == next_idx + 1);
      assert(src_.size() == next_idx + 1);
      assert(types_.size() == next_idx + 1);
      assert(location_timezones_.size() == next_idx + 1);
      assert(equivalences_.size() == next_idx + 1);
      assert(children_.size() == next_idx + 1);
      assert(footpaths_out_.size() == next_idx + 1);
      assert(footpaths_in_.size() == next_idx + 1);
      assert(transfer_time_.size() == next_idx + 1);
      assert(osm_ids_.size() == next_idx + 1);
      assert(parents_.size() == next_idx + 1);

      return it->second;
    }

    location get(location_idx_t const idx) {
      return location{ids_[idx].view(),
                      names_[idx].view(),
                      coordinates_[idx],
                      src_[idx],
                      types_[idx],
                      osm_ids_[idx],
                      parents_[idx],
                      location_timezones_[idx],
                      transfer_time_[idx],
                      it_range{equivalences_[idx]},
                      it_range{footpaths_out_[idx]},
                      it_range{footpaths_in_[idx]}};
    }

    location get(location_id const& id) {
      return get(location_id_to_idx_.at(id));
    }

    // Station access: external station id -> internal station idx
    hash_map<location_id, location_idx_t> location_id_to_idx_;
    vecvec<location_idx_t, char> names_;
    vecvec<location_idx_t, char> ids_;
    vector_map<location_idx_t, geo::latlng> coordinates_;
    vector_map<location_idx_t, source_idx_t> src_;
    vector_map<location_idx_t, u8_minutes> transfer_time_;
    vector_map<location_idx_t, location_type> types_;
    vector_map<location_idx_t, osm_node_id_t> osm_ids_;
    vector_map<location_idx_t, location_idx_t> parents_;
    vector_map<location_idx_t, timezone_idx_t> location_timezones_;
    mutable_fws_multimap<location_idx_t, location_idx_t> equivalences_;
    mutable_fws_multimap<location_idx_t, location_idx_t> children_;
    mutable_fws_multimap<location_idx_t, footpath> footpaths_out_;
    mutable_fws_multimap<location_idx_t, footpath> footpaths_in_;
    vector_map<timezone_idx_t, timezone> timezones_;
  } locations_;

  struct transport {
    bitfield_idx_t bitfield_idx_;
    route_idx_t route_idx_;
    std::basic_string<merged_trips_idx_t> const& external_trip_ids_;
    std::basic_string<attribute_combination_idx_t> const& section_attributes_;
    std::basic_string<provider_idx_t> const& section_providers_;
    std::basic_string<trip_direction_idx_t> const& section_directions_;
    std::basic_string<trip_line_idx_t> const& section_lines_;
  };

  trip_idx_t register_trip_id(
      fmt::memory_buffer const& trip_id_str,
      source_idx_t const src,
      std::string const& display_name,
      trip_debug const dbg,
      transport_idx_t const ref_transport,
      interval<std::uint32_t> ref_transport_stop_range) {
    auto const trip_idx = trip_idx_t{trip_ids_.size()};

    auto const trip_id_idx = trip_id_idx_t{trip_id_strings_.size()};
    trip_id_strings_.emplace_back(trip_id_str);
    trip_id_src_.emplace_back(src);

    trip_id_to_idx_.emplace_back(trip_id_idx, trip_idx);
    trip_display_names_.emplace_back(display_name);
    trip_debug_.emplace_back().emplace_back(dbg);
    trip_ids_.emplace_back().emplace_back(trip_id_idx);
    trip_ref_transport_.emplace_back(ref_transport, ref_transport_stop_range);

    return trip_idx;
  }

  bitfield_idx_t register_bitfield(bitfield const& b) {
    auto const idx = bitfield_idx_t{bitfields_.size()};
    bitfields_.emplace_back(b);
    return idx;
  }

  template <typename T>
  trip_direction_string_idx_t register_trip_direction_string(T&& s) {
    auto const idx = trip_direction_string_idx_t{bitfields_.size()};
    trip_direction_strings_.emplace_back(s);
    return idx;
  }

  route_idx_t register_route(
      std::basic_string<stop::value_type> const& stop_seq,
      std::basic_string<clasz> const& clasz_sections) {
    auto const idx = route_location_seq_.size();
    for (auto const& s : stop_seq) {
      location_routes_[timetable::stop{s}.location_idx()].emplace_back(idx);
    }
    route_transport_ranges_.emplace_back(
        transport_idx_t{transport_traffic_days_.size()},
        transport_idx_t::invalid());
    route_location_seq_.emplace_back(stop_seq);
    route_section_clasz_.emplace_back(clasz_sections);
    return route_idx_t{idx};
  }

  void finish_route() {
    route_transport_ranges_.back().to_ =
        transport_idx_t{transport_traffic_days_.size()};
  }

  merged_trips_idx_t register_merged_trip(
      std::basic_string<trip_idx_t> const& trip_ids) {
    auto const idx = merged_trips_.size();
    merged_trips_.emplace_back(trip_ids);
    return merged_trips_idx_t{static_cast<merged_trips_idx_t::value_t>(idx)};
  }

  source_file_idx_t register_source_file(char const* path) {
    auto const idx = source_file_idx_t{source_file_names_.size()};
    source_file_names_.emplace_back(path);
    return idx;
  }

  void add_transport(transport&& t) {
    transport_traffic_days_.emplace_back(t.bitfield_idx_);
    transport_route_.emplace_back(t.route_idx_);
    transport_to_trip_section_.emplace_back(t.external_trip_ids_);
    transport_section_attributes_.emplace_back(t.section_attributes_);
    transport_section_providers_.emplace_back(t.section_providers_);
    transport_section_directions_.emplace_back(t.section_directions_);
    transport_section_lines_.emplace_back(t.section_lines_);

    assert(transport_traffic_days_.size() == transport_route_.size());
    assert(transport_traffic_days_.size() == transport_to_trip_section_.size());
  }

  transport_idx_t next_transport_idx() const {
    return transport_idx_t{transport_traffic_days_.size()};
  }

  std::span<minutes_after_midnight_t const> event_times_at_stop(
      route_idx_t const r,
      std::size_t const stop_idx,
      event_type const ev_type) const {
    auto const n_transports =
        static_cast<unsigned>(route_transport_ranges_[r].size());
    auto const idx = static_cast<unsigned>(
        route_stop_time_ranges_[r].from_ +
        n_transports * (stop_idx * 2 - (ev_type == event_type::kArr ? 1 : 0)));
    return std::span<minutes_after_midnight_t const>{&route_stop_times_[idx],
                                                     n_transports};
  }

  minutes_after_midnight_t event_mam(route_idx_t const r,
                                     transport_idx_t t,
                                     std::size_t const stop_idx,
                                     event_type const ev_type) const {
    auto const range = route_transport_ranges_[r];
    auto const n_transports = static_cast<unsigned>(range.size());
    auto const route_stop_begin = static_cast<unsigned>(
        route_stop_time_ranges_[r].from_ +
        n_transports * (stop_idx * 2 - (ev_type == event_type::kArr ? 1 : 0)));
    auto const t_idx_in_route = to_idx(t) - to_idx(range.from_);
    return route_stop_times_[route_stop_begin + t_idx_in_route];
  }

  minutes_after_midnight_t event_mam(transport_idx_t t,
                                     std::size_t const stop_idx,
                                     event_type const ev_type) const {
    return event_mam(transport_route_[t], t, stop_idx, ev_type);
  }

  unixtime_t event_time(nigiri::transport t,
                        size_t const stop_idx,
                        event_type const ev_type) const {
    return unixtime_t{internal_interval_days().from_ + to_idx(t.day_) * 1_days +
                      event_mam(t.t_idx_, stop_idx, ev_type)};
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
                         minutes_after_midnight_t const m) const {
    return internal_interval_days().from_ + to_idx(d) * 1_days + m;
  }

  cista::base_t<location_idx_t> n_locations() const {
    return locations_.names_.size();
  }

  cista::base_t<route_idx_t> n_routes() const {
    return route_location_seq_.size();
  }

  interval<unixtime_t> external_interval() const {
    return {std::chrono::time_point_cast<i32_minutes>(date_range_.from_),
            std::chrono::time_point_cast<i32_minutes>(date_range_.to_)};
  }

  interval<unixtime_t> internal_interval_days() const {
    return {date_range_.from_ - kTimetableOffset, date_range_.to_ + 1_days};
  }

  constexpr interval<unixtime_t> internal_interval() const {
    return {
        std::chrono::time_point_cast<i32_minutes>(date_range_.from_ -
                                                  kTimetableOffset),
        std::chrono::time_point_cast<i32_minutes>(date_range_.to_ + 1_days)};
  }

  friend std::ostream& operator<<(std::ostream&, timetable const&);
  friend void print_1(std::ostream&, timetable const&);

  void write(std::filesystem::path const&) const;
  static cista::wrapped<timetable> read(cista::memory_holder);

  // Schedule range.
  interval<std::chrono::sys_days> date_range_;

  // Trip access: external trip id -> internal trip index
  vector<pair<trip_id_idx_t, trip_idx_t>> trip_id_to_idx_;

  // Trip index -> list of external trip ids
  mutable_fws_multimap<trip_idx_t, trip_id_idx_t> trip_ids_;

  // Storage for trip id strings + source
  vecvec<trip_id_idx_t, char> trip_id_strings_;
  vector_map<trip_id_idx_t, source_idx_t> trip_id_src_;

  // Trip index -> reference transport + stop range
  vector_map<trip_idx_t, pair<transport_idx_t, interval<std::uint32_t>>>
      trip_ref_transport_;

  // Trip -> debug info
  mutable_fws_multimap<trip_idx_t, trip_debug> trip_debug_;
  vecvec<source_file_idx_t, char> source_file_names_;

  // Trip index -> display name
  vecvec<trip_idx_t, char> trip_display_names_;

  // Route -> From (inclusive) and to index (exclusive) of expanded trips
  vector_map<route_idx_t, interval<transport_idx_t>> route_transport_ranges_;

  // Route -> list of stops
  vecvec<route_idx_t, stop::value_type> route_location_seq_;

  // Route -> clasz per section
  vecvec<route_idx_t, clasz> route_section_clasz_;

  // Location -> list of routes
  mutable_fws_multimap<location_idx_t, route_idx_t> location_routes_;

  // Route 1:
  //   stop-1-dep: [trip1, trip2, ..., tripN]
  //   stop-2-arr: [trip1, trip2, ..., tripN]
  //   ...
  // Route 2:
  //  stop-1-dep: [...]
  // ...
  // RouteN: ...
  vector_map<route_idx_t, interval<unsigned>> route_stop_time_ranges_;
  vector<minutes_after_midnight_t> route_stop_times_;

  // Trip index -> traffic day bitfield
  vector_map<transport_idx_t, bitfield_idx_t> transport_traffic_days_;

  // Trip index -> sequence of stop times
  vecvec<rt_trip_idx_t, unixtime_t> rt_transport_stop_times_;

  // Unique bitfields
  vector_map<bitfield_idx_t, bitfield> bitfields_;

  // bitfields_[1][day_1], bitfields_2[2][day_1], ...
  // bitfields_[1][day_2], bitfields_2[2][day_2], ...
  bitvec col_bitfields_;

  // For each trip the corresponding route
  vector_map<transport_idx_t, route_idx_t> transport_route_;

  // Trip index -> merged trips
  vecvec<transport_idx_t, merged_trips_idx_t> transport_to_trip_section_;

  // Merged trips info
  vecvec<merged_trips_idx_t, trip_idx_t> merged_trips_;

  // Trip index -> list of section ranges where this trip was expanded
  mutable_fws_multimap<trip_idx_t,
                       pair<transport_idx_t, interval<std::uint32_t>>>
      trip_idx_to_transport_idx_;

  // Track names
  vecvec<track_name_idx_t, char> track_names_;

  // Section meta infos:
  vector_map<attribute_idx_t, attribute> attributes_;
  vecvec<attribute_combination_idx_t, attribute_idx_t> attribute_combinations_;
  vector_map<provider_idx_t, provider> providers_;
  vecvec<trip_direction_string_idx_t, char> trip_direction_strings_;
  vector_map<trip_direction_idx_t, trip_direction_t> trip_directions_;
  vecvec<trip_line_idx_t, char> trip_lines_;

  vecvec<transport_idx_t, attribute_combination_idx_t>
      transport_section_attributes_;
  vecvec<transport_idx_t, provider_idx_t> transport_section_providers_;
  vecvec<transport_idx_t, trip_direction_idx_t> transport_section_directions_;
  vecvec<transport_idx_t, trip_line_idx_t> transport_section_lines_;

  // Lower bound graph.
  vecvec<location_idx_t, footpath> fwd_search_lb_graph_;
  vecvec<location_idx_t, footpath> bwd_search_lb_graph_;
};

}  // namespace nigiri
