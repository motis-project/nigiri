#pragma once

#include "nigiri/routing/start_times.h"
#include "nigiri/timetable.h"
#include "utl/equal_ranges_linear.h"

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

  static constexpr auto const kInvalidTime =
      SearchDir == direction::kForward ? std::numeric_limits<unixtime_t>::max()
                                       : std::numeric_limits<unixtime_t>::min();

  struct search_state {
    void reset(size_t const num_locations, size_t const num_routes) {
      station_mark_.resize(num_locations);
      route_mark_.resize(num_routes);
      std::fill(begin(station_mark_), end(station_mark_), false);
      std::fill(begin(route_mark_), end(route_mark_), false);

      earliest_arrivals_.resize(num_locations);
      std::fill(begin(earliest_arrivals_), end(earliest_arrivals_),
                kInvalidTime);

      arrival_times_.resize(num_locations, kMaxTransfers);
      arrival_times_.reset(kInvalidTime);

      current_start_index_ = 0U;
    }

    std::vector<unixtime_t> earliest_arrivals_;
    matrix<unixtime_t> arrival_times_;
    std::vector<bool> station_mark_;
    std::vector<bool> route_mark_;
    std::vector<start> starts_;
    std::size_t current_start_index_{0U};
  };

  raptor(std::shared_ptr<timetable const> tt, search_state state, query q)
      : tt_mem_{std::move(tt)},
        tt_{*tt_mem_},
        state_{std::move(state)},
        q_{std::move(q)} {}

  routing_result route(query const& q) {
    init();
    rounds();
    return routing_result{reconstruct_journeys(), begin_, end_};
  }

  void update_route(unsigned const k, route_idx_t const r_idx) {}
  void update_footpaths(unsigned const k) {}

  void rounds() {
    for (auto k = 1; k <= kMaxTransfers; ++k) {
      auto any_marked = false;
      for (auto l_idx = location_idx_t{0}; l_idx != state_.station_mark_.size();
           ++l_idx) {
        if (!state_.station_mark_[l_idx]) {
          continue;
        }
        if (!any_marked) {
          any_marked = true;
        }
        for (auto const& r : tt_.location_routes_[l_idx]) {
          state_.route_mark_[r] = true;
        }
      }

      if (!any_marked) {
        break;
      }

      std::fill(begin(state_.station_mark_), end(state_.station_mark_), false);

      for (auto r_id = 0; r_id < tt_.n_routes(); ++r_id) {
        if (!state_.station_mark_[r_id]) {
          continue;
        }

        update_route(k, r_id);
      }

      std::fill(begin(state_.station_mark_), end(state_.station_mark_), false);

      update_footpaths(k);
    }
  }

  void init() {
    state_.reset(tt_.n_locations(), tt_.n_routes());
    get_starts<SearchDir>(tt_, first_day_, q_.interval_begin_, q_.interval_end_,
                          q_.start_, state_.starts_);
    utl::equal_ranges_linear(
        state_.starts_,
        [](start const& a, start const& b) {
          return a.time_at_start_ == b.time_at_start_;
        },
        [&](std::vector<start>::const_iterator const& from_it,
            std::vector<start>::const_iterator const& to_it) {
          for (auto const& s : it_range{from_it, to_it}) {
            state_.arrival_times_[0][s.stop_] = s.time_at_stop_;
            state_.station_mark_[s.stop_] = true;
          }
          rounds();
          reconstruct_journeys();
        });
  }

  void reconstruct_journeys() {}

  std::chrono::sys_days first_day_;
  query q_;
  unixtime_t curr_begin_;
  unixtime_t begin_, end_;
  search_state state_;
  std::shared_ptr<timetable const> tt_mem_;
  timetable const& tt_;
};

}  // namespace nigiri::routing