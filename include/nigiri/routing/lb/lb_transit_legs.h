#pragma once

#include "nigiri/for_each_meta.h"
#include "nigiri/routing/query.h"
#include "nigiri/rt/rt_timetable.h"
#include "nigiri/timetable.h"

namespace nigiri::routing {

// SearchDir refers to the direction of the main routing query
// fwd: finds the minimum number of transit legs backward from the destination
// bwd: finds the minimum number of transit legs forward from the destination
template <direction SearchDir>
void lb_transit_legs_round(timetable const&, query const&, std::uint8_t k);

template <direction SearchDir>
struct lb_transit_legs {
  static constexpr auto kUnreachable = std::numeric_limits<std::uint8_t>::max();
  static constexpr auto kUnknown = kUnreachable - 1U;

  lb_transit_legs(timetable const& tt,
                  query const& q,
                  rt_timetable const* rtt = nullptr,
                  bool const disabled = false)
      : tt_{tt},
        rtt_{rtt != nullptr && rtt->n_rt_transports() > 0U ? rtt : nullptr},
        q_{q},
        k_{0U},
        end_k_{static_cast<std::uint8_t>(
            std::min(q.max_transfers_, kMaxTransfers) + 2U)},
        any_marked_{false},
        total_time_{0U} {
    lb_.resize(tt_.n_locations());
    if (disabled) {
      utl::fill(lb_, 0U);
      return;
    }

    station_mark_.resize(tt_.n_locations());
    utl::fill(station_mark_.blocks_, 0U);
    prev_station_mark_.resize(tt_.n_locations());
    ride_expanded_.resize(tt_.n_locations());
    route_mark_.resize(tt_.n_routes());
    if (rtt_ != nullptr) {
      rt_transport_mark_.resize(rtt_->n_rt_transports());
    }
    utl::fill(lb_, kUnknown);

    auto const set_terminal = [&](auto const i) {
      station_mark_.set(to_idx(i), true);
      lb_[i] = 0U;
      // Stops within walking distance of a terminal:
      constexpr auto kFwd = SearchDir == direction::kForward;
      for (auto const fp :
           kFwd ? tt_.locations_.footpaths_in_[q_.prf_idx_][i]
                : tt_.locations_.footpaths_out_[q_.prf_idx_][i]) {
        if (lb_[fp.target()] != 0U) {
          lb_[fp.target()] = 0U;
          station_mark_.set(to_idx(fp.target()), true);
        }
      }
    };

    for (auto const& o : q.destination_) {
      for_each_meta(tt_, q.dest_match_mode_, o.target(),
                    [&](location_idx_t const meta) { set_terminal(meta); });
    }

    for (auto const& [l, tds] : q.td_dest_) {
      for (auto const& td : tds) {
        if (td.duration() != footpath::kMaxDuration &&
            td.duration() < q.max_travel_time_) {
          for_each_meta(tt_, q.dest_match_mode_, l,
                        [&](location_idx_t const meta) { set_terminal(meta); });
        }
      }
    }

    k_ = 1U;
    for (auto const& s : q.start_) {
      get(s.target());
    }
    for (auto const& td : q.td_start_) {
      get(td.first);
    }
  }

  void run_round() {
    constexpr auto kFwd = SearchDir == direction::kForward;
    auto const start_time = std::chrono::steady_clock::now();

    utl::fill(route_mark_.blocks_, 0U);
    utl::fill(ride_expanded_.blocks_, 0U);
    if (rtt_ != nullptr) {
      utl::fill(rt_transport_mark_.blocks_, 0U);
    }

    any_marked_ = false;
    station_mark_.for_each_set_bit([&](std::uint64_t const i) {
      for (auto const r : tt_.location_routes_[location_idx_t{i}]) {
        any_marked_ = true;
        route_mark_.set(to_idx(r), true);
      }
      if (rtt_ != nullptr) {
        for (auto const rt_t :
             rtt_->location_rt_transports_[location_idx_t{i}]) {
          any_marked_ = true;
          rt_transport_mark_.set(to_idx(rt_t), true);
        }
      }
    });
    if (!any_marked_) {
      return;
    }

    std::swap(prev_station_mark_, station_mark_);
    utl::fill(station_mark_.blocks_, 0U);

    any_marked_ = false;
    auto const relax_seq = [&](auto const& seq) {
      for (auto x = 0U; x != seq.size(); ++x) {
        auto const in = kFwd ? seq.size() - x - 1U : x;
        auto const l_in = stop{seq[in]}.location_idx();

        if (!prev_station_mark_.test(to_idx(l_in))) {
          continue;
        }

        auto const step = [&](auto const y) { return kFwd ? y - 1U : y + 1U; };
        for (auto out = step(in); out < seq.size(); out = step(out)) {
          auto const l_out = stop{seq[out]}.location_idx();

          if (!ride_expanded_.test(to_idx(l_out))) {
            ride_expanded_.set(to_idx(l_out), true);
            for (auto const fp :
                 kFwd ? tt_.locations_.footpaths_in_[q_.prf_idx_][l_out]
                      : tt_.locations_.footpaths_out_[q_.prf_idx_][l_out]) {
              if (k_ < lb_[fp.target()]) {
                lb_[fp.target()] = k_;
                station_mark_.set(to_idx(fp.target()), true);
                any_marked_ = true;
              }
            }
          }

          if (k_ < lb_[l_out]) {
            lb_[l_out] = k_;
            station_mark_.set(to_idx(l_out), true);
            any_marked_ = true;
          } else if (k_ > lb_[l_out]) {
            break;
          }
        }
      }
    };

    route_mark_.for_each_set_bit([&](std::uint64_t const i) {
      relax_seq(tt_.route_location_seq_[route_idx_t{i}]);
    });
    if (rtt_ != nullptr) {
      rt_transport_mark_.for_each_set_bit([&](std::uint64_t const i) {
        relax_seq(rtt_->rt_transport_location_seq_[rt_transport_idx_t{i}]);
      });
    }

    total_time_ += std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now() - start_time);
  }

  std::uint8_t get(location_idx_t const l) {
    while (lb_[l] == kUnknown) {
      run_round();
      ++k_;
      if (!any_marked_ || k_ == end_k_) {
        for (auto& lb : lb_) {
          if (lb == kUnknown) {
            lb = kUnreachable;
          }
        }
      }
    }
    return lb_[l];
  }

  timetable const& tt_;
  rt_timetable const* rtt_;
  query const& q_;
  std::uint8_t k_;
  std::uint8_t end_k_;
  bool any_marked_;
  std::chrono::microseconds total_time_;
  bitvec station_mark_;
  bitvec prev_station_mark_;
  bitvec ride_expanded_;
  bitvec route_mark_;
  bitvec rt_transport_mark_;
  vector_map<location_idx_t, std::uint8_t> lb_;
};

}  // namespace nigiri::routing
