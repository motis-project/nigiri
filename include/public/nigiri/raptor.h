#pragma once

#include "utl/enumerate.h"

#include "nigiri/timetable.h"

namespace nigiri {

struct query {
  struct edge {
    location_idx_t location_;
    duration_t offset_;
    std::uint8_t type_;
  };

  direction search_dir_;
  unixtime_t interval_begin_;
  unixtime_t interval_end_;
  vector<edge> start_;
  vector<vector<edge>> destinations_;
  vector<vector<edge>> via_destinations_;
  cista::bitset<kNumClasses> allowed_classes_;
  std::uint8_t max_transfers_;
  std::uint8_t min_connection_count_;
  bool extend_interval_earlier_;
  bool extend_interval_later_;
};

struct journey {
  struct leg {
    location_idx_t from_, to_;
    unixtime_t dep_time_, arr_time_;
    variant<route_idx_t, footpath_idx_t> uses_;
  };

  std::vector<leg> legs_;
};

struct routing_result {
  vector<journey> journeys_;
  unixtime_t interval_begin_;
  unixtime_t interval_end_;
};

template <direction SearchDir>
struct raptor {
  static constexpr auto const kMaxTransfers = 7;

  // Measure time since first day of search
  using raptor_time_t = std::int16_t;

  static constexpr auto const kInvalidTime =
      SearchDir == direction::kForward
          ? std::numeric_limits<raptor_time_t>::max()
          : std::numeric_limits<raptor_time_t>::min();

  struct search_state {
    void reset(size_t const num_locations, size_t const num_routes) {
      station_mark_.resize(num_locations);
      route_mark_.resize(num_routes);

      arrival_times_.resize(num_locations, kMaxTransfers);
      arrival_times_.reset(kInvalidTime);
    }

    void reset() {
      arrival_times_.reset(kInvalidTime);
      std::fill(begin(station_mark_), end(station_mark_), false);
      std::fill(begin(route_mark_), end(route_mark_), false);
    }

    matrix<raptor_time_t> arrival_times_;
    std::vector<bool> station_mark_;
    std::vector<bool> route_mark_;
  };

  raptor(std::shared_ptr<timetable const> tt, search_state& state, query q)
      : tt_mem_{std::move(tt)}, tt_{*tt_mem_}, q_{std::move(q)} {}

  routing_result route(query const& q) {
    init();
    rounds();
    return routing_result{reconstruct_journeys(), begin_, end_};
  }

  void rounds() {}

  void init() {
    std::vector<raptor_time_t> start_times;
    auto const add_start_times = [&](query::edge const& e) {
      for (auto const& r : tt_.location_routes_.at(e.location_)) {
        for (auto const& [i, s] :
             utl::enumerate(tt_.route_location_seq_.at(r))) {
          if (s.location_idx() == e.location_) {
          }
        }
      }
    };

    for (auto const& start : q_.start_) {
      add_start_times(start);
    }
  }

  vector<journey> reconstruct_journeys() { return {}; }

  raptor_time_t to_raptor_time(unixtime_t const t) {
    return (t - first_day_).count();
  }

  unixtime_t to_unixtime(raptor_time_t const t) {
    return first_day_ + t * 1_minutes;
  }

  std::chrono::sys_days first_day_;
  query q_;
  unixtime_t curr_begin_;
  unixtime_t begin_, end_;
  search_state& state_;
  std::shared_ptr<timetable const> tt_mem_;
  timetable const& tt_;
};

}  // namespace nigiri