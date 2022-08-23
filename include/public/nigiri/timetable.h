#pragma once

#include <compare>
#include <type_traits>

#include "cista/reflection/printable.h"

#include "utl/verify.h"
#include "utl/zip.h"

#include "geo/latlng.h"

#include "nigiri/logging.h"
#include "nigiri/section_db.h"
#include "nigiri/types.h"

namespace nigiri {

template <typename BeginIt, typename EndIt = BeginIt>
struct it_range {
  template <typename Collection>
  explicit it_range(Collection&& c)
      : begin_{std::begin(c)}, end_{std::end(c)} {}
  explicit it_range(BeginIt begin, EndIt end)
      : begin_{std::move(begin)}, end_{std::move(end)} {}
  BeginIt begin() const { return begin_; }
  EndIt end() const { return end_; }
  friend BeginIt begin(it_range const& r) { return r.begin(); }
  friend EndIt end(it_range const& r) { return r.end(); }
  BeginIt begin_;
  EndIt end_;
};

template <typename Collection>
it_range(Collection const&) -> it_range<typename Collection::iterator>;

template <typename BeginIt, typename EndIt>
it_range(BeginIt, EndIt) -> it_range<BeginIt, EndIt>;

struct footpath {
  CISTA_PRINTABLE(footpath, "target", "duration")
  location_idx_t target_;
  duration_t duration_;
};

template <typename T>
struct interval {
  template <typename X>
  interval operator+(X const& x) const {
    return {from_ + x, to_ + x};
  }

  template <typename X>
  interval operator-(X const& x) const {
    return {from_ - x, to_ - x};
  }

  template <typename X>
    requires std::is_convertible_v<T, X>
  operator interval<X>() {
    return {from_, to_};
  }

  bool contains(unixtime_t const t) const { return t >= from_ && t < to_; }

  T from_{}, to_{};
};

template <typename T>
interval(T, T) -> interval<T>;

struct timetable {
  struct expanded_trip_section {
    transport_idx_t trip_idx_{};
    section_idx_t from_section_idx_{}, to_section_idx_{};
  };

  struct external_trip_section {
    trip_idx_t trip_idx_{};
    section_idx_t section_idx_{};
  };

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

  struct location {
    string const& id_;
    string const& name_;
    geo::latlng pos_;
    source_idx_t src_;
    location_type type_;
    osm_node_id_t osm_id_;
    location_idx_t parent_;
    timezone_idx_t timezone_idx_;
    it_range<vector<location_idx_t>::iterator> equivalences_;
    it_range<vector<footpath>::iterator> footpaths_out_, footpaths_in_;
  };

  struct locations {
    using location_multimap =
        mutable_fws_multimap<location_idx_t, location_idx_t>;
    using footpath_multimap = mutable_fws_multimap<location_idx_t, footpath>;

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
      return location{.id_ = ids_[idx],
                      .name_ = names_[idx],
                      .pos_ = coordinates_[idx],
                      .src_ = src_[idx],
                      .type_ = types_[idx],
                      .osm_id_ = osm_ids_[idx],
                      .parent_ = parents_[idx],
                      .timezone_idx_ = location_timezones_[idx],
                      .equivalences_ = it_range{equivalences_[idx]},
                      .footpaths_out_ = it_range{footpaths_out_[idx]},
                      .footpaths_in_ = it_range{footpaths_in_[idx]}};
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
    vector<section_db_idx_t> meta_data_;
    vector<merged_trips_idx_t> external_trip_ids_;
  };

  trip_idx_t register_trip_id(trip_id const& id,
                              string display_name,
                              string debug) {
    auto const idx = trip_idx_t{trip_ids_.size()};
    auto& trips = trip_id_to_idx_[id];
    trips.emplace_back(idx);
    trip_display_names_.emplace_back(std::move(display_name));
    trip_debug_.emplace_back().emplace_back(std::move(debug));
    trip_ids_.emplace_back().emplace_back(id);
    return idx;
  }

  bitfield_idx_t register_bitfield(bitfield const& b) {
    auto const idx = bitfield_idx_t{bitfields_.size()};
    bitfields_.emplace_back(b);
    return idx;
  }

  route_idx_t register_route(vector<stop> stop_seq) {
    auto const idx = route_location_seq_.size();
    for (auto const& s : stop_seq) {
      location_routes_[s.location_idx()].emplace_back(idx);
    }
    route_transport_ranges_.emplace_back(
        transport_idx_t{transport_traffic_days_.size()},
        transport_idx_t::invalid());
    route_location_seq_.emplace_back(std::move(stop_seq));
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

  void add_transport(transport&& t) {
    transport_traffic_days_.emplace_back(t.bitfield_idx_);
    transport_route_.emplace_back(t.route_idx_);
    transport_stop_times_.emplace_back(std::move(t.stop_times_));
    transport_section_meta_data_.emplace_back(std::move(t.meta_data_));
    transport_to_trip_section_.emplace_back(std::move(t.external_trip_ids_));

    assert(transport_traffic_days_.size() == transport_route_.size());
    assert(transport_traffic_days_.size() == transport_stop_times_.size());
    assert(transport_traffic_days_.size() ==
           transport_section_meta_data_.size());
    assert(transport_traffic_days_.size() == transport_to_trip_section_.size());
    assert(t.stop_times_.size() ==
           route_location_seq_.at(t.route_idx_).size() * 2 - 2);
    assert(t.external_trip_ids_.size() == t.stop_times_.size() / 2);
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
    auto const idx = stop_idx * 2 - (ev_type == event_type::kArr ? 1 : 0);
    return transport_stop_times_[transport_idx][idx];
  }

  std::pair<day_idx_t, minutes_after_midnight_t> day_idx_mam(
      unixtime_t const t) const {
    auto const minutes_since_timetable_begin = (t - begin_).count();
    auto const d =
        static_cast<day_idx_t::value_t>(minutes_since_timetable_begin / 1440);
    auto const m = minutes_since_timetable_begin % 1440;
    return {day_idx_t{d}, minutes_after_midnight_t{m}};
  }

  unixtime_t to_unixtime(day_idx_t const d,
                         minutes_after_midnight_t const m) const {
    return begin_ + to_idx(d) * 1_days + m;
  }

  cista::base_t<location_idx_t> n_locations() const {
    return locations_.names_.size();
  }

  cista::base_t<route_idx_t> n_routes() const {
    return route_location_seq_.size();
  }

  friend std::ostream& operator<<(std::ostream&, timetable const&);

  // Schedule range.
  unixtime_t begin_, end_;
  std::uint16_t n_days_;

  // Trip access: external trip id -> internal trip index
  hash_map<trip_id, vector<trip_idx_t>> trip_id_to_idx_;

  // External trip index -> list of external trip ids (HRD + RI Basis)
  mutable_fws_multimap<trip_idx_t, trip_id> trip_ids_;

  // External trip -> debug info
  mutable_fws_multimap<trip_idx_t, string> trip_debug_;

  // External trip index -> display name
  vector_map<trip_idx_t, string> trip_display_names_;

  // Route -> From (inclusive) and to index (exclusive) of expanded trips
  vector_map<route_idx_t, interval<transport_idx_t>> route_transport_ranges_;

  // Route -> list of stops
  vecvec<route_idx_t, stop> route_location_seq_;

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

  // Trip index -> trip section meta data db index
  vecvec<transport_idx_t, section_db_idx_t> transport_section_meta_data_;

  // Trip index -> merged trips
  vecvec<transport_idx_t, merged_trips_idx_t> transport_to_trip_section_;

  // Merged trips info
  vecvec<merged_trips_idx_t, trip_idx_t> merged_trips_;

  // External trip index -> list of section ranges where this trip was
  // expanded
  mutable_fws_multimap<trip_idx_t, expanded_trip_section>
      trip_idx_to_transport_idx_;
};

}  // namespace nigiri
