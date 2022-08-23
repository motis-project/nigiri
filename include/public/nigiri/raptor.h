#pragma once

#include "fmt/core.h"

#include "utl/equal_ranges_linear.h"

#include "nigiri/routing/start_times.h"
#include "nigiri/timetable.h"

namespace nigiri::routing {

constexpr auto const kTracing = true;

template <typename... Args>
void trace(char const* fmt_str, Args... args) {
  if constexpr (kTracing) {
    fmt::print(std::cerr, fmt_str, std::forward<Args&&>(args)...);
  }
}

struct query {
  direction search_dir_;
  interval<unixtime_t> interval_;
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
  unixtime_t start_;
  unixtime_t dest_time_;
  std::uint8_t transfers_{0U};
};

struct routing_result {
  vector<journey> journeys_;
  unixtime_t interval_begin_;
  unixtime_t interval_end_;
};

struct search_state {
  std::vector<unixtime_t> best_;
  matrix<unixtime_t> round_times_;
  std::vector<bool> station_mark_;
  std::vector<bool> route_mark_;
  std::vector<start> starts_;
  std::vector<journey> results_;
  std::size_t current_start_index_{0U};
};

static constexpr auto const kMaxTransfers = std::uint8_t{7U};

template <direction SearchDir>
struct raptor {
  static constexpr auto const kFwd = SearchDir == direction::kForward;
  static constexpr auto const kInvalidTime =
      SearchDir == direction::kForward ? std::numeric_limits<unixtime_t>::max()
                                       : std::numeric_limits<unixtime_t>::min();

  raptor(std::shared_ptr<timetable const> tt, search_state state, query q)
      : tt_mem_{std::move(tt)},
        tt_{*tt_mem_},
        q_{std::move(q)},
        state_{std::move(state)} {}

  void reset() {
    state_.station_mark_.resize(tt_.n_locations());
    state_.route_mark_.resize(tt_.n_routes());
    std::fill(begin(state_.station_mark_), end(state_.station_mark_), false);
    std::fill(begin(state_.route_mark_), end(state_.route_mark_), false);

    state_.best_.resize(tt_.n_locations());
    std::fill(begin(state_.best_), end(state_.best_), kInvalidTime);

    state_.round_times_.resize(tt_.n_locations(), kMaxTransfers + 1U);
    state_.round_times_.reset(kInvalidTime);

    state_.current_start_index_ = 0U;
  }

  static constexpr event_type ev_type() {
    return kFwd ? event_type::kDep : event_type::kArr;
  }
  static constexpr bool is_better(auto const a, auto const b) {
    return kFwd ? a < b : a > b;
  }
  static constexpr bool is_better_or_equal(auto const a, auto const b) {
    return kFwd ? a <= b : a >= b;
  }
  static constexpr unixtime_t get_best(unixtime_t const a, unixtime_t const b) {
    return is_better(a, b) ? a : b;
  }
  unixtime_t time_at_stop(transport_idx_t const t_idx,
                          day_idx_t const day_idx,
                          unsigned const stop_idx,
                          event_type const ev_type) {
    auto const mam = tt_.event_mam(t_idx, stop_idx, ev_type);
    return tt_.to_unixtime(day_idx_t{day_idx}, mam);
  }

  void update_route(unsigned const k, route_idx_t const r) {
    constexpr auto const kInvalid = std::int32_t{-1};

    auto const& stop_seq = tt_.route_location_seq_[r];

    auto const get_earliest_transport = [&](unsigned const stop_idx,
                                            location_idx_t const l_idx)
        -> std::pair<std::int32_t, std::int32_t> {
      auto const transport_range = tt_.route_transport_ranges_[r];
      auto const [day_at_stop, mam_at_stop] =
          tt_.day_idx_mam(state_.round_times_[k - 1][to_idx(l_idx)]);

      auto const n_days_to_iterate =
          kFwd ? tt_.n_days_ - to_idx(day_at_stop) : to_idx(day_at_stop) + 1U;
      for (auto i = std::uint16_t{0U}; i != n_days_to_iterate; ++i) {
        auto const day = kFwd ? day_at_stop + i : day_at_stop - i;
        for (auto t = transport_range.from_; t != transport_range.to_; ++t) {
          auto const ev = tt_.event_mam(t, stop_idx, ev_type());
          auto const ev_mam = minutes_after_midnight_t{ev.count() % 1440};
          if (day == day_at_stop && !is_better_or_equal(ev_mam, mam_at_stop)) {
            continue;
          }

          auto const ev_day_offset =
              static_cast<cista::base_t<day_idx_t>>(ev.count() / 1440);
          if (!tt_.bitfields_[tt_.transport_traffic_days_[t]].test(
                  to_idx(day) - ev_day_offset)) {
            continue;
          }

          return {to_idx(t), to_idx(day - ev_day_offset)};
        }
      }
      return {kInvalid, kInvalid};
    };

    auto earliest_transport = kInvalid;
    auto earliest_day = kInvalid;

    for (auto i = 0U; i != stop_seq.size(); ++i) {
      auto const stop_idx =
          static_cast<unsigned>(kFwd ? i : stop_seq.size() - i - 1U);
      auto const l_idx = cista::to_idx(stop_seq[i].location_idx());
      auto const current_best =
          get_best(state_.best_[l_idx], state_.round_times_[k - 1][l_idx]);
      if (earliest_transport != kInvalid) {
        auto const by_transport_time = time_at_stop(
            transport_idx_t{earliest_transport}, day_idx_t{earliest_day},
            stop_idx, kFwd ? event_type::kArr : event_type::kDep);
        if (is_better(by_transport_time, current_best)) {
          state_.best_[l_idx] = by_transport_time;
          state_.round_times_[k][l_idx] = by_transport_time;
        }
      }

      if (i != stop_seq.size() - 1 &&
          (earliest_transport == kInvalid ||
           is_better_or_equal(
               state_.round_times_[k - 1][l_idx],
               time_at_stop(transport_idx_t{earliest_transport},
                            day_idx_t{earliest_day},
                            static_cast<unsigned>(stop_idx),
                            kFwd ? event_type::kDep : event_type::kArr)))) {
        std::tie(earliest_transport, earliest_day) =
            get_earliest_transport(i, location_idx_t{l_idx});
      }
    }
  }

  void update_footpaths(unsigned const k) {
    for (auto l_idx = location_idx_t{0U}; l_idx != tt_.n_locations(); ++l_idx) {
      if (!state_.station_mark_[to_idx(l_idx)]) {
        continue;
      }

      auto const& fps = kFwd ? tt_.locations_.footpaths_out_[l_idx]
                             : tt_.locations_.footpaths_in_[l_idx];
      for (auto const& fp : fps) {
        auto& time_at_fp_target = state_.round_times_[k][to_idx(fp.target_)];
        auto const arrival = state_.round_times_[k][to_idx(l_idx)] +
                             (kFwd ? fp.duration_ : -fp.duration_);
        if (is_better(arrival, time_at_fp_target)) {
          time_at_fp_target = arrival;
          state_.station_mark_[to_idx(fp.target_)] = true;
        }
      }
    }
  }

  void rounds() {
    auto const max_transfers = std::min(kMaxTransfers, q_.max_transfers_);
    for (auto k = 1U; k != max_transfers + 1U; ++k) {
      trace("┊ round k={}\n", k);

      auto any_marked = false;
      for (auto l_idx = location_idx_t{0U};
           l_idx != static_cast<cista::base_t<location_idx_t>>(
                        state_.station_mark_.size());
           ++l_idx) {
        if (state_.station_mark_[to_idx(l_idx)]) {
          any_marked = true;
          for (auto const& r : tt_.location_routes_[l_idx]) {
            state_.route_mark_[to_idx(r)] = true;
          }
        }
      }

      if (!any_marked) {
        trace("┊ ╰ nothing marked, exit\n\n");
        break;
      }

      std::fill(begin(state_.station_mark_), end(state_.station_mark_), false);

      for (auto r_id = 0U; r_id != tt_.n_routes(); ++r_id) {
        if (!state_.route_mark_[r_id]) {
          continue;
        }
        trace("┊ ├ updating route {}\n", r_id);
        update_route(k, route_idx_t{r_id});
      }

      std::fill(begin(state_.station_mark_), end(state_.station_mark_), false);

      update_footpaths(k);

      trace("┊ ╰ done\n\n");
    }
  }

  void route() {
    reset();
    get_starts<SearchDir>(tt_, q_.interval_, q_.start_, state_.starts_);
    utl::equal_ranges_linear(
        state_.starts_,
        [](start const& a, start const& b) {
          return a.time_at_start_ == b.time_at_start_;
        },
        [&](std::vector<start>::const_iterator const& from_it,
            std::vector<start>::const_iterator const& to_it) {
          for (auto const& s : it_range{from_it, to_it}) {
            trace("init {} [{}]: {}\n", tt_.locations_.names_[s.stop_],
                  tt_.locations_.ids_[s.stop_], s.time_at_stop_);
            state_.round_times_[0U][to_idx(s.stop_)] = s.time_at_stop_;
            state_.station_mark_[to_idx(s.stop_)] = true;
          }
          rounds();
          reconstruct_journeys();
        });
  }

  void reconstruct_journeys() {
    for (auto const& t : q_.destinations_) {
      auto const dest = t.front().location_;
      for (auto k = 1U; k != kMaxTransfers + 1; ++k) {
        if (state_.round_times_[k][to_idx(dest)] == kInvalidTime) {
          continue;
        }
        state_.results_.emplace_back(
            journey{.dest_time_ = state_.round_times_[k][to_idx(dest)],
                    .transfers_ = static_cast<std::uint8_t>(k - 1)});
      }
    }
  }

  std::shared_ptr<timetable const> tt_mem_;
  timetable const& tt_;
  query q_;
  unixtime_t curr_begin_;
  unixtime_t begin_, end_;
  search_state state_;
};

}  // namespace nigiri::routing
