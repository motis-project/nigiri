#pragma once

#include "nigiri/types.h"

#include "nigiri/logging.h"
#include "cista/reflection/printable.h"

namespace nigiri {

template <typename BeginIt, typename EndIt = BeginIt>
struct it_range {
  template <typename Collection>
  explicit it_range(Collection&& c)
      : begin_{std::begin(c)}, end_{std::end(c)} {}
  BeginIt begin() const { return begin_; }
  EndIt end() const { return end_; }
  friend BeginIt begin(it_range const& r) { return r.begin(); }
  friend EndIt end(it_range const& r) { return r.end(); }
  BeginIt begin_;
  EndIt end_;
};

template <typename Collection>
it_range(Collection const&) -> it_range<typename Collection::iterator>;

struct footpath {
  CISTA_PRINTABLE(footpath, "target", "duration")
  location_idx_t target_;
  duration_t duration_;
};

struct timetable {
  struct expanded_trip_section {
    trip_idx_t trip_idx_{};
    section_idx_t from_section_idx_{}, to_section_idx_{};
  };

  struct external_trip_section {
    external_trip_idx_t trip_idx_{};
    section_idx_t section_idx_{};
  };

  template <typename T>
  struct index_range {
    T from_{}, to_{};
  };

  struct stop {
    location_idx_t location_idx() const { return location_idx_t{location_}; }
    bool in_allowed() const { return in_allowed_; }
    bool out_allowed() const { return out_allowed_; }

  private:
    location_idx_t::value_t location_ : 30;
    location_idx_t::value_t in_allowed_ : 1;
    location_idx_t::value_t out_allowed_ : 1;
  };
  static_assert(sizeof(stop) == sizeof(location_idx_t));

  struct location {
    string const& id_;
    source_idx_t src_;
    location_type type_;
    osm_node_id_t osm_id_;
    location_idx_t parent_;
    it_range<vector<location_idx_t>::iterator> equivalences_;
    it_range<vector<footpath>::iterator> footpaths_out_, footpaths_in_;
  };

  struct locations {
    using location_multimap =
        mutable_fws_multimap<location_idx_t, location_idx_t>;
    using footpath_multimap = mutable_fws_multimap<location_idx_t, footpath>;

    location_idx_t add(location&& l) {
      auto const [it, is_new] = location_id_to_idx_.emplace(
          location_id{.id_ = l.id_, .src_ = l.src_}, next_id());

      if (is_new) {
        log(log_lvl::info, "nigiri.timetable.location.add", "adding {}",
            location_id{.id_ = l.id_, .src_ = l.src_});
        ids_.emplace_back(l.id_);
        src_.emplace_back(l.src_);
        types_.emplace_back(location_type::station);
        transfer_time_.emplace_back(2);  // TODO(felix)
        osm_ids_.emplace_back(0);  // TODO(felix)
        parents_.emplace_back(0);  // TODO(felix)
      }

      return it->second;
    }

    location get(location_idx_t const idx) {
      return {.id_ = ids_[idx],
              .src_ = src_[idx],
              .type_ = types_[idx],
              .osm_id_ = osm_ids_[idx],
              .parent_ = parents_[idx],
              .equivalences_ = it_range{equivalences_[idx]},
              .footpaths_out_ = it_range{footpaths_out_[idx]},
              .footpaths_in_ = it_range{footpaths_in_[idx]}};
    }

    location get(location_id const& id) {
      return get(location_id_to_idx_.at(id));
    }

    location_idx_t next_id() const {
      return location_idx_t{
          static_cast<location_idx_t::value_t>(location_id_to_idx_.size())};
    }

    // Station access: external station id -> internal station idx
    hash_map<location_id, location_idx_t> location_id_to_idx_;
    vector_map<location_idx_t, string> ids_;
    vector_map<location_idx_t, source_idx_t> src_;
    vector_map<location_idx_t, duration_t> transfer_time_;
    vector_map<location_idx_t, location_type> types_;
    vector_map<location_idx_t, osm_node_id_t> osm_ids_;
    vector_map<location_idx_t, location_idx_t> parents_;
    mutable_fws_multimap<location_idx_t, location_idx_t> equivalences_;
    mutable_fws_multimap<location_idx_t, location_idx_t> children_;
    mutable_fws_multimap<location_idx_t, footpath> footpaths_out_;
    mutable_fws_multimap<location_idx_t, footpath> footpaths_in_;
  } locations_;

  // Trip access: external trip id -> internal trip index
  hash_map<trip_id, external_trip_idx_t> trip_id_to_idx_;

  // External trip index -> list of external trip ids (HRD + RI Basis)
  mutable_fws_multimap<external_trip_idx_t, trip_id> trip_ids_;

  // Route -> From (inclusive) and to index (exclusive) of expanded trips
  fws_multimap<route_idx_t, index_range<trip_idx_t>> route_trip_ranges_;

  // Route -> list of stops
  fws_multimap<route_idx_t, stop> route_location_seq_;

  // Location -> list of routes
  mutable_fws_multimap<location_idx_t, route_idx_t> location_routes_;

  // Trip index -> sequence of stop times
  fws_multimap<trip_idx_t, minutes_after_midnight_t> trip_stop_times_;

  // Trip index -> traffic day bitfield
  vector_map<trip_idx_t, bitfield_idx_t> expanded_trip_traffic_days_;

  // Trip index -> sequence of stop times
  fws_multimap<rt_trip_idx_t, unixtime_t> rt_trip_stop_times_;

  // Unique bitfields
  vector_map<bitfield_idx_t, bitfield> bitfields_;

  // For each trip the corresponding route
  vector_map<trip_idx_t, route_idx_t> trip_route_;

  // Trip index -> trip section meta data db index
  fws_multimap<trip_idx_t, section_db_idx_t> trip_section_meta_data_;

  // Trip index -> merged trips
  fws_multimap<trip_idx_t, merged_trips_idx_t> trip_to_external_trip_section_;

  // Merged trips info
  fws_multimap<merged_trips_idx_t, external_trip_idx_t> merged_trips_;

  // External trip index -> list of section ranges where this trip was
  // expanded
  mutable_fws_multimap<external_trip_idx_t, expanded_trip_section>
      external_trip_idx_to_expanded_trip_idx_;

  // External trip -> debug info
  vector_map<trip_idx_t, string> trip_debug_;
};

}  // namespace nigiri
