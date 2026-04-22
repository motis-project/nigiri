#pragma once

#include "nigiri/for_each_meta.h"
#include "nigiri/routing/query.h"
#include "nigiri/timetable.h"

namespace nigiri::routing {

struct lb_transit_legs_state {
  bool any_marked_;
  bitvec station_mark_;
  bitvec prev_station_mark_;
  bitvec route_mark_;
  vector_map<location_idx_t, std::uint8_t> lb_;
};

// SearchDir refers to the direction of the main routing query
// fwd: finds the minimum number of transit legs backward from the destination
// bwd: finds the minimum number of transit legs forward from the destination
template <direction SearchDir>
void lb_transit_legs_round(timetable const&,
                           query const&,
                           lb_transit_legs_state&,
                           std::uint8_t k);

template <direction SearchDir>
struct lb_transit_legs {
  static constexpr auto kUnreachable = std::numeric_limits<std::uint8_t>::max();
  static constexpr auto kUnknown = kUnreachable - 1U;

  lb_transit_legs(timetable const& tt,
                  query const& q,
                  lb_transit_legs_state& state)
      : tt_{tt}, q_{q}, state_{state}, k_{0U} {}

  void init(bool const one_to_all = false) {
    auto const start_time = std::chrono::steady_clock::now();
    state_.lb_.resize(tt_.n_locations());
    if (one_to_all) {
      utl::fill(state_.lb_, 0U);
      return;
    }

    state_.station_mark_.resize(tt_.n_locations());
    utl::fill(state_.station_mark_.blocks_, 0U);
    state_.prev_station_mark_.resize(tt_.n_locations());
    state_.route_mark_.resize(tt_.n_routes());
    utl::fill(state_.lb_, kUnknown);

    auto const set_terminal = [&](auto const i) {
      state_.station_mark_.set(to_idx(i), true);
      state_.lb_[i] = 0U;
    };

    for (auto const& o : q_.destination_) {
      for_each_meta(tt_, q_.dest_match_mode_, o.target(),
                    [&](location_idx_t const meta) { set_terminal(meta); });
    }

    for (auto const& [l, tds] : q_.td_dest_) {
      for (auto const& td : tds) {
        if (td.duration() != footpath::kMaxDuration &&
            td.duration() < q_.max_travel_time_) {
          for_each_meta(tt_, q_.dest_match_mode_, l,
                        [&](location_idx_t const meta) { set_terminal(meta); });
        }
      }
    }

    k_ = 1U;
    for (auto const& s : q_.start_) {
      get(s.target());
    }
    for (auto const& td : q_.td_start_) {
      get(td.first);
    }
    fmt::println("[lb_transit_legs] initialized after {} rounds which took {} ",
                 k_,
                 std::chrono::duration_cast<std::chrono::milliseconds>(
                     std::chrono::steady_clock::now() - start_time));
  }

  std::uint8_t get(location_idx_t const l) {
    while (state_.lb_[l] == kUnknown) {
      lb_transit_legs_round<SearchDir>(tt_, q_, state_, k_++);
      if (!state_.any_marked_ ||
          k_ == std::min(q_.max_transfers_, kMaxTransfers) + 2U) {
        for (auto& lb : state_.lb_) {
          if (lb == kUnknown) {
            lb = kUnreachable;
          }
        }
      }
    }
    return state_.lb_[l];
  }

  timetable const& tt_;
  query const& q_;
  lb_transit_legs_state& state_;
  std::uint8_t k_;
};

}  // namespace nigiri::routing