#pragma once

#include "nigiri/routing/start_times.h"
#include "nigiri/timetable.h"

namespace nigiri::routing {

struct query {
  direction search_dir_;
  unixtime_t interval_begin_;
  unixtime_t interval_end_;
  vector<offset> start_;
  vector<vector<offset>> destinations_;
  vector<vector<offset>> via_destinations_;
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
    get_starts<SearchDir>(tt_, first_day_, q_.interval_begin_, q_.interval_end_,
                          q_.start_);
  }

  vector<journey> reconstruct_journeys() { return {}; }

  std::chrono::sys_days first_day_;
  query q_;
  unixtime_t curr_begin_;
  unixtime_t begin_, end_;
  search_state& state_;
  std::shared_ptr<timetable const> tt_mem_;
  timetable const& tt_;
};

}  // namespace nigiri::routing