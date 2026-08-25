#pragma once

#include <cassert>
#include <cstddef>

#include "nigiri/routing/limits.h"
#include "nigiri/routing/query.h"
#include "nigiri/types.h"

namespace nigiri::routing {

// Two independent timetable worlds searched in one sweep:
//   slot 0 (kSched): journeys that only use scheduled connections,
//                    transport activity answered by the timetable
//                    (rt-cancelled trips stay usable)
//   slot 1 (kRt):    journeys in the rt world: non-cancelled scheduled
//                    connections + rt transports, activity answered by
//                    the rt_timetable
// The slots never interact: no transitions, no dominance, separate
// time-at-dest bounds. Requires Rt=true (an rt_timetable, may be empty).
template <direction SearchDir>
struct schedrt_criterion {
  static constexpr auto const kFwd = (SearchDir == direction::kForward);
  static constexpr auto const kN = std::size_t{2U};
  static constexpr auto const kSched = std::size_t{0U};
  static constexpr auto const kRt = std::size_t{1U};
  static constexpr auto const kFinalSlot = kRt;

  // both worlds produce journeys, each pruned against its own bound
  static constexpr auto const kNDestBounds = std::size_t{2U};
  static constexpr std::size_t dest_bound_of(std::size_t const v) { return v; }
  static constexpr bool is_dest_slot(std::size_t) { return true; }
  template <typename Fn>
  static void for_each_dest_slot(Fn&& fn) {
    fn(kSched);
    fn(kRt);
  }
  static constexpr std::uint8_t journey_slot(std::size_t const v) {
    return static_cast<std::uint8_t>(v);
  }

  static constexpr auto const kAllSlotsRt = false;
  static constexpr bool slot_uses_rt(std::size_t const v) { return v == kRt; }
  static constexpr bool slot_boards_rt(std::size_t const v) { return v == kRt; }

  // both slots board from the same label times in the common case:
  // fuse the two earliest-transport walks into one
  static constexpr auto const kFusedEt = true;

  // 2-slot interleaved state (see schedrt_cod_criterion for copy-on-diverge)
  static constexpr auto const kCopyOnDiverge = false;

  using bag_t = std::array<delta_t, kN>;

  struct transition {
    std::size_t target_;
    duration_t stay_;
  };

  static bool is_better(auto a, auto b) { return kFwd ? a < b : a > b; }

  schedrt_criterion(std::array<bitvec, kMaxVias> const&,
                    std::vector<via_stop> const& via_stops) {
    assert(via_stops.empty());
    (void)via_stops;
  }

  // both worlds start everywhere a start label exists
  template <typename Fn>
  void for_each_start_slot(std::size_t, Fn&& fn) const {
    fn(kSched);
    fn(kRt);
  }

  transition stop_transition(std::size_t const v, std::size_t) const {
    return {v, 0_minutes};
  }

  template <typename Fn>
  void for_each_final_stop_transition(std::size_t, Fn&&) const {}

  // no cross-slot dominance: incomparable worlds
  static constexpr bool dominates(std::size_t const a, std::size_t const b) {
    return a == b;
  }

  void fold_bounds(bag_t&) const {}

  static constexpr std::size_t bound_slot(std::size_t const v) { return v; }

  int bound_slack(std::size_t) const { return 0; }

  // two independent riders, no slot transitions while riding
  struct route_ride {
    void reset() { et_ = {}; }

    template <typename TimeAt>
    void advance(std::size_t, TimeAt&&) {}

    template <typename Fn>
    void for_each_arrival(Fn&& fn) const {
      for (auto j = 0U; j != kN; ++j) {
        auto const v = kN - 1U - j;
        if (et_[v].is_valid()) {
          fn(v, et_[v]);
        }
      }
    }

    transport& fresh(std::size_t const v) { return et_[v]; }

    schedrt_criterion const& c_;
    std::array<transport, kN> et_{};
  };

  // rt transports only ever serve the rt slot
  struct rt_ride {
    void reset() { et_ = false; }
    void advance(std::size_t) {}

    template <typename Fn>
    void for_each_arrival(Fn&& fn) const {
      if (et_) {
        fn(kRt);
      }
    }

    void board(std::size_t) { et_ = true; }

    schedrt_criterion const& c_;
    bool et_{false};
  };
};

// copy-on-diverge variant: the scheduled world lives in a dense single-slot
// base plane, the rt world is an overlay materialized only where the worlds
// actually differ (measured: ~0.5% of touched cells under full rt load)
template <direction SearchDir>
struct schedrt_cod_criterion : schedrt_criterion<SearchDir> {
  static constexpr auto const kCopyOnDiverge = true;

  schedrt_cod_criterion(std::array<bitvec, kMaxVias> const& is_via,
                        std::vector<via_stop> const& via_stops)
      : schedrt_criterion<SearchDir>{is_via, via_stops} {}
};

}  // namespace nigiri::routing
