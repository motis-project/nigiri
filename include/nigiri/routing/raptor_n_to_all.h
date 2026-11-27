#pragma once

#include <cstdint>
#include <utility>
#include <vector>

#include "nigiri/routing/query.h"
#include "nigiri/routing/raptor/raptor_state.h"
#include "nigiri/types.h"

namespace nigiri {
struct timetable;
struct rt_timetable;
}  // namespace nigiri

namespace nigiri::routing {

struct n_to_all_result {
  // Shortest travel time to `l` over all departures, and where it began.
  min_label const& at(location_idx_t const l) const {
    return state_.min_[to_idx(l)];
  }
  // Aggregates of `l` (only with with_aggregates = true), over the minutes
  // t = 0..window of the window: the travel time at t is the best over all
  // origins, waiting for the departure included.
  struct stop_stats {
    std::uint32_t minutes_{0};    // window + 1
    std::uint32_t covered_{0};    // minutes with a travel time <= the limit
    std::int32_t sum_{0};         // sum of the travel times over the covered minutes
    std::int32_t max_{-1};        // worst travel time over the covered minutes
    location_idx_t max_owner_{location_idx_t::invalid()};  // start stop at the worst minute
    bool full() const { return covered_ == minutes_; }
    double avg() const { return covered_ == 0U ? 0.0 : 1.0 * sum_ / covered_; }
  };

  stop_stats stats(location_idx_t const l) const {
    auto s = stop_stats{};
    if (state_.agg_.labels_.empty()) {
      return s;
    }
    auto const& a = state_.agg_.labels_[to_idx(l)];
    s.minutes_ = static_cast<std::uint32_t>(state_.agg_.len_) + 1U;
    s.covered_ = static_cast<std::uint32_t>(a.covered_);
    s.sum_ = a.sum_;
    s.max_ = a.max_;
    s.max_owner_ = a.max_owner_;
    return s;
  }

  // Counters (only if an origin map was passed). n_origins() origins, numbered
  // as in the map. minutes(l, o) = minutes of the window at which origin o is
  // the closest to `l`; over all origins they add up to stats(l).covered_.
  std::uint32_t n_origins() const { return state_.agg_.n_origins_; }

  std::uint16_t minutes(location_idx_t const l, std::uint32_t const o) const {
    return state_.agg_.minutes_.empty()
               ? std::uint16_t{0U}
    : state_.agg_.minutes_[to_idx(l) * n_origins() + o];
  }

  // share of the covered minutes of `l` that belong to origin o (0 if none)
  double share(location_idx_t const l, std::uint32_t const o) const {
    auto const c = stats(l).covered_;
    return c == 0U ? 0.0 : 1.0 * minutes(l, o) / c;
  }
  // Profile of `l` (only with with_profile = true): every run of the search
  // whose vehicle journey to `l` is not dominated, earliest departure first.
  // departure_ strictly increases and arrival_ strictly increases.
  // For a person at the origin at minute t (t <= last departure_), the best
  // vehicle journey is the first entry with departure_ >= t, and the travel
  // time is arrival_ - t (waiting included). owner_ is the start stop it began
  // at. The travel time at t is the smaller of that and profile_start(l).
  // Start label of `l`: reached without a vehicle (the origin itself or a
  // footpath/equivalent neighbour). Constant over the window, no waiting.
  profile_start_label const& profile_start(location_idx_t const l) const {
    static auto const none = profile_start_label{};
    return state_.profile_start_.empty() ? none
                                         : state_.profile_start_[to_idx(l)];
  }

  std::vector<profile_entry> profile(location_idx_t const l) const {
    if (state_.profile_.empty()) {
      return {};
    }
    // stored latest departure first, returned earliest departure first
    auto const& e = state_.profile_[to_idx(l)];
    return std::vector<profile_entry>(e.rbegin(), e.rend());
  }

  raptor_state state_;
  // Interval that was actually searched (equals the requested one unless
  // search<> widened it).
  interval<unixtime_t> searched_;
};

// n-to-all: for every stop, the shortest travel time from any of the start
// offsets in q.start_ / q.td_start_ and the start stop it began at (map start
// stops back to your origins, see raptor_n_to_all_ep.cc).
//
//   q.start_time_  a single time or an interval (range RAPTOR: one RAPTOR run
//                  per departure from the start stops inside the interval)
//   q.start_       start offsets of ALL origins; the destination is ignored
//
// with_profile: additionally record the profile of every stop (see
// n_to_all_result::profile). Costs memory: up to one entry per run and stop.
//
// Forward only, no via stops. Runs search<> with the one-to-all RAPTOR and
// raptor_state::track_min_ switched on.
n_to_all_result raptor_n_to_all(timetable const&,
                                rt_timetable const*,
                                query q,
                                bool with_profile = false,
                                bool with_aggregates = false,
                                std::vector<std::pair<location_idx_t, std::uint8_t>> const&
                                    origin_of_start = {});

}  // namespace nigiri::routing