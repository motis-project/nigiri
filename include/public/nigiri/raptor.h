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
  static constexpr auto const kMaxTransfers = std::uint8_t{7U};

  static constexpr auto const kInvalidTime =
      SearchDir == direction::kForward ? std::numeric_limits<unixtime_t>::max()
                                       : std::numeric_limits<unixtime_t>::min();

  struct search_state {
    void reset(size_t const num_locations, size_t const num_routes) {
      station_mark_.resize(num_locations);
      route_mark_.resize(num_routes);
      std::fill(begin(station_mark_), end(station_mark_), false);
      std::fill(begin(route_mark_), end(route_mark_), false);

      best_.resize(num_locations);
      std::fill(begin(best_), end(best_), kInvalidTime);

      round_times_.resize(num_locations, kMaxTransfers);
      round_times_.reset(kInvalidTime);

      current_start_index_ = 0U;
    }

    std::vector<unixtime_t> best_;
    matrix<unixtime_t> round_times_;
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

  static constexpr event_type ev_type() {
    return SearchDir == direction::kForward ? event_type::kDep
                                            : event_type::kArr;
  }

  void update_route(unsigned const k, route_idx_t const r) {
    auto const& stop_seq = tt_.route_location_seq_[r];

    auto const get_earliest_transport = [&](unsigned const stop_idx) {
      auto const transport_range = tt_.route_transport_ranges_[r];
      for (auto t = transport_range.from_; t != transport_range.to_; ++t) {
        auto const [time_at_stop_day, time_at_stop_mam] =
            tt_.day_idx_mam(state_.round_times_[k - 1]);
        auto const transport_event_mam = tt_.event_mam(t, stop_idx, ev_type());
      }
    };

    auto const time_at = [&](transport_idx_t const t_idx,
                             day_idx_t const day_idx, unsigned const stop_idx) {
      auto const mam = tt_.event_mam(t_idx, stop_idx, ev_type());
      return tt_.to_unixtime(day_idx_t{day_idx}, mam);
    };
    auto const is_better = [](unixtime_t const a, unixtime_t const b) {
      return (SearchDir == direction::kForward) ? a < b : a > b;
    };
    auto const is_better_or_equal = [](unixtime_t const a, unixtime_t const b) {
      return (SearchDir == direction::kForward) ? a <= b : a >= b;
    };
    auto const get_best = [&](unixtime_t const a, unixtime_t const b) {
      return is_better(a, b) ? a : b;
    };

    constexpr auto const kInvalid = std::int32_t{-1};
    auto const earliest_transport = transport_idx_t{kInvalid};
    auto const earliest_day = kInvalid;

    for (auto i = 0U; i != stop_seq.size(); ++i) {
      auto const stop_idx =
          (SearchDir == direction::kForward) ? i : stop_seq.size() - i - 1U;
      auto const l_idx = stop_seq[i];
      auto const current_best =
          get_best(state_.best_[l_idx], state_.round_times_[k - 1][l_idx]);
      if (earliest_transport != transport_idx_t{kInvalid}) {
        auto const time = time_at(earliest_transport, earliest_day, i);
        if (is_better(time, current_best)) {
          state_.best_[l_idx] = time;
          state_.round_times_[k] = time;
        }
      }

      if (is_better_or_equal(
              state_.round_times_[k - 1][l_idx],
              time_at(earliest_transport, earliest_day, stop_idx))) {
        earliest_transport = get_earliest_transport(i);
      }
    }
  }

  void update_footpaths(unsigned const k) {
    for (auto l_idx = location_idx_t{0U}; l_idx != tt_.n_locations(); ++l_idx) {
      if (!state_.station_mark_[to_idx(l_idx)]) {
        continue;
      }

      if constexpr (SearchDir == direction::kForward) {
        for (auto const& fp : tt_.locations_.footpaths_out_[l_idx]) {
          auto& time_at_fp_target = state_.round_times_[k][fp.target_];
          auto const arrival = state_.round_times_[k][l_idx] + fp.duration_;
          if (time_at_fp_target > arrival) {
            time_at_fp_target = arrival;
            state_.station_mark_[fp.target_] = true;
          }
        }
      } else {
        for (auto const& fp : tt_.locations_.footpaths_in_[l_idx]) {
          auto& time_at_fp_target = state_.round_times_[k][fp.target_];
          auto const arrival = state_.round_times_[k][l_idx] - fp.duration_;
          if (time_at_fp_target < arrival) {
            time_at_fp_target = arrival;
            state_.station_mark_[fp.target_] = true;
          }
        }
      }
    }
  }

  void rounds() {
    auto const max_transfers = std::min(kMaxTransfers, q_.max_transfers_);
    for (auto k = 1; k <= max_transfers + 1U; ++k) {
      auto any_marked = false;
      for (auto l_idx = location_idx_t{0U};
           l_idx != state_.station_mark_.size(); ++l_idx) {
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

  void route() {
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
            state_.round_times_[0][s.stop_] = s.time_at_stop_;
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
