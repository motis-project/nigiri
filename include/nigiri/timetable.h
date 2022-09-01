#pragma once

#include <compare>
#include <type_traits>

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
      return cista::hash_combine(
          cista::BASE_HASH,
          *reinterpret_cast<location_idx_t::value_t const*>(this));
    }

    friend auto operator<=>(stop const&, stop const&) = default;

  private:
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
        types_.emplace_back(location_type::kStation);
        location_timezones_.emplace_back(l.timezone_idx_);
        equivalences_.emplace_back();
        children_.emplace_back();
        footpaths_out_.emplace_back();
        footpaths_in_.emplace_back();
        transfer_time_.emplace_back(2);  // TODO(felix)
        osm_ids_.emplace_back(osm_node_id_t::invalid());  // TODO(felix)
        parents_.emplace_back(location_idx_t::invalid());  // TODO(felix)
      }

      return it->second;
    }

    location get(location_idx_t const idx) {
      return location{ids_[idx],
                      names_[idx],
                      coordinates_[idx],
                      src_[idx],
                      types_[idx],
                      osm_ids_[idx],
                      parents_[idx],
                      location_timezones_[idx],
                      it_range{equivalences_[idx]},
                      it_range{footpaths_out_[idx]},
                      it_range{footpaths_in_[idx]}};
    }

    location get(location_id const& id) {
      return get(location_id_to_idx_.at(id));
    }

    // Station access: external station id -> internal station idx
    hash_map<location_id, location_idx_t> location_id_to_idx_;
    vector_map<location_idx_t, string> names_;
    vector_map<location_idx_t, geo::latlng> coordinates_;
    vector_map<location_idx_t, string> ids_;
    vector_map<location_idx_t, source_idx_t> src_;
    vector_map<location_idx_t, duration_t> transfer_time_;
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
    vector<minutes_after_midnight_t> stop_times_;
    vector<merged_trips_idx_t> external_trip_ids_;
    vector<attribute_combination_idx_t> section_attributes_;
    vector<provider_idx_t> section_providers_;
    vector<trip_direction_idx_t> section_directions_;
  };

  trip_idx_t register_trip_id(
      trip_id const& id,
      string display_name,
      trip_debug const dbg,
      transport_idx_t const ref_transport,
      interval<std::uint32_t> ref_transport_stop_range) {
    auto const idx = trip_idx_t{trip_ids_.size()};
    auto& trips = trip_id_to_idx_[id];
    trips.emplace_back(idx);
    trip_display_names_.emplace_back(std::move(display_name));
    trip_debug_.emplace_back().emplace_back(dbg);
    trip_ids_.emplace_back().emplace_back(id);
    trip_ref_transport_.emplace_back(ref_transport, ref_transport_stop_range);
    return idx;
  }

  bitfield_idx_t register_bitfield(bitfield const& b) {
    auto const idx = bitfield_idx_t{bitfields_.size()};
    bitfields_.emplace_back(b);
    return idx;
  }

  trip_direction_string_idx_t register_trip_direction_string(string&& s) {
    auto const idx = trip_direction_string_idx_t{bitfields_.size()};
    trip_direction_strings_.emplace_back(std::move(s));
    return idx;
  }

  route_idx_t register_route(vector<stop> stop_seq,
                             vector<clasz> clasz_sections) {
    auto const idx = route_location_seq_.size();
    for (auto const& s : stop_seq) {
      location_routes_[s.location_idx()].emplace_back(idx);
    }
    route_transport_ranges_.emplace_back(
        transport_idx_t{transport_traffic_days_.size()},
        transport_idx_t::invalid());
    route_location_seq_.emplace_back(std::move(stop_seq));
    route_section_clasz_.emplace_back(std::move(clasz_sections));
    return route_idx_t{idx};
  }

  void finish_route() {
    route_transport_ranges_.back().to_ =
        transport_idx_t{transport_traffic_days_.size()};
  }

  merged_trips_idx_t register_merged_trip(vector<trip_idx_t> trip_ids) {
    auto const idx = merged_trips_.size();
    merged_trips_.emplace_back(std::move(trip_ids));
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
    transport_stop_times_.emplace_back(std::move(t.stop_times_));
    transport_to_trip_section_.emplace_back(std::move(t.external_trip_ids_));
    transport_section_attributes_.emplace_back(
        std::move(t.section_attributes_));
    transport_section_providers_.emplace_back(std::move(t.section_providers_));
    transport_section_directions_.emplace_back(
        std::move(t.section_directions_));

    assert(transport_traffic_days_.size() == transport_route_.size());
    assert(transport_traffic_days_.size() == transport_stop_times_.size());
    assert(transport_traffic_days_.size() == transport_to_trip_section_.size());
    assert(t.stop_times_.size() ==
           route_location_seq_.at(t.route_idx_).size() * 2 - 2);
    assert(t.external_trip_ids_.size() == 1U ||
           t.external_trip_ids_.size() == t.stop_times_.size() / 2);
    assert(t.section_attributes_.size() == 1U ||
           t.section_attributes_.size() == t.stop_times_.size() / 2);
    assert(t.section_providers_.size() == 1U ||
           t.section_providers_.size() == t.stop_times_.size() / 2);
    assert(t.section_directions_.size() == 1U ||
           t.section_directions_.size() == t.stop_times_.size() / 2);
  }

  transport_idx_t next_transport_idx() const {
    return transport_idx_t{transport_traffic_days_.size()};
  }

  minutes_after_midnight_t event_mam(transport_idx_t const transport_idx,
                                     size_t const stop_idx,
                                     event_type const ev_type) const {
    // Event times are stored alternatingly:
    // departure (D), arrival (A), ..., arrival (A)
    // event type: D A D A D A D A
    // stop index: 0 1 1 2 2 3 3 4
    // event time: 0 1 2 3 4 5 6 7
    // --> A at stop i = i x 2 - 1
    // --> D at stop i = i x 2
    // There's no arrival at the first stop and no departure at the last stop.
    assert(!(stop_idx == 0 && ev_type == event_type::kArr));
    assert(!(stop_idx == transport_stop_times_[transport_idx].size() - 1 &&
             ev_type == event_type::kDep));
    auto const idx = stop_idx * 2 - (ev_type == event_type::kArr ? 1 : 0);
    return transport_stop_times_[transport_idx][idx];
  }

  unixtime_t event_time(nigiri::transport t,
                        size_t const stop_idx,
                        event_type const ev_type) const {
    return unixtime_t{date_range_.from_ + to_idx(t.day_) * 1_days +
                      event_mam(t.t_idx_, stop_idx, ev_type)};
  }

  std::pair<day_idx_t, minutes_after_midnight_t> day_idx_mam(
      unixtime_t const t) const {
    auto const minutes_since_timetable_begin = (t - date_range_.from_).count();
    auto const d =
        static_cast<day_idx_t::value_t>(minutes_since_timetable_begin / 1440);
    auto const m = minutes_since_timetable_begin % 1440;
    return {day_idx_t{d}, minutes_after_midnight_t{m}};
  }

  unixtime_t to_unixtime(day_idx_t const d,
                         minutes_after_midnight_t const m) const {
    return date_range_.from_ + to_idx(d) * 1_days + m;
  }

  cista::base_t<location_idx_t> n_locations() const {
    return locations_.names_.size();
  }

  cista::base_t<route_idx_t> n_routes() const {
    return route_location_seq_.size();
  }

  unixtime_t begin() const {
    return unixtime_t{std::chrono::duration_cast<i32_minutes>(
        date_range_.from_.time_since_epoch())};
  }

  unixtime_t end() const {
    return unixtime_t{std::chrono::duration_cast<i32_minutes>(
        date_range_.to_.time_since_epoch())};
  }

  friend std::ostream& operator<<(std::ostream&, timetable const&);

  // Schedule range.
  interval<std::chrono::sys_days> date_range_;

  // Trip access: external trip id -> internal trip index
  hash_map<trip_id, vector<trip_idx_t>> trip_id_to_idx_;

  // Trip index -> list of external trip ids (HRD + RI Basis)
  mutable_fws_multimap<trip_idx_t, trip_id> trip_ids_;

  // Trip index -> reference transport + stop range
  vector_map<trip_idx_t, pair<transport_idx_t, interval<std::uint32_t>>>
      trip_ref_transport_;

  // Trip -> debug info
  mutable_fws_multimap<trip_idx_t, trip_debug> trip_debug_;
  vecvec<source_file_idx_t, char> source_file_names_;

  // Trip index -> display name
  vector_map<trip_idx_t, string> trip_display_names_;

  // Route -> From (inclusive) and to index (exclusive) of expanded trips
  vector_map<route_idx_t, interval<transport_idx_t>> route_transport_ranges_;

  // Route -> list of stops
  vecvec<route_idx_t, stop> route_location_seq_;

  // Route -> clasz per section
  vecvec<route_idx_t, clasz> route_section_clasz_;

  // Location -> list of routes
  mutable_fws_multimap<location_idx_t, route_idx_t> location_routes_;

  // Trip index -> sequence of stop times
  vecvec<transport_idx_t, minutes_after_midnight_t> transport_stop_times_;

  // Trip index -> traffic day bitfield
  vector_map<transport_idx_t, bitfield_idx_t> transport_traffic_days_;

  // Trip index -> sequence of stop times
  vecvec<rt_trip_idx_t, unixtime_t> rt_transport_stop_times_;

  // Unique bitfields
  vector_map<bitfield_idx_t, bitfield> bitfields_;

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

  // Section meta infos:
  vector_map<attribute_idx_t, attribute> attributes_;
  vecvec<attribute_combination_idx_t, attribute_idx_t> attribute_combinations_;
  vector_map<provider_idx_t, provider> providers_;
  vecvec<trip_direction_string_idx_t, char> trip_direction_strings_;
  vector_map<trip_direction_idx_t, trip_direction_t> trip_directions_;
  vecvec<line_idx_t, char> lines_;

  vecvec<transport_idx_t, attribute_combination_idx_t>
      transport_section_attributes_;
  vecvec<transport_idx_t, provider_idx_t> transport_section_providers_;
  vecvec<transport_idx_t, trip_direction_idx_t> transport_section_directions_;
  vecvec<transport_idx_t, line_idx_t> transport_section_lines_;
};

}  // namespace nigiri
