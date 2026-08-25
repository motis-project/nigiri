#pragma once

#include <cassert>
#include <cstddef>

#include "nigiri/routing/limits.h"
#include "nigiri/routing/query.h"
#include "nigiri/types.h"

namespace nigiri::routing {

// Slot-static criterion for the RAPTOR product-state search: each stop keeps
// one delta_t per slot, slot count and semantics are fixed at compile time.
//
// Criterion concept:
//   kN                       number of label slots per stop
//   kFinalSlot               accepting slot (journeys are read from here)
//   bag_t                    std::array<delta_t, kN> - the raptor_state layout
//   start_slot(l)            slot a journey starts in at location l
//   stop_transition(v, l)    {target slot, dwell} when crossing stop l in v
//   for_each_final_stop_transition(l, fn)  transitions into kFinalSlot at l
//   route_ride / rt_ride     onboard slot state for route / rt transport scans
//   dominates(a, b)          slot partial order (spec; fold_bounds + pruning)
//   fold_bounds(slots)       fold each slot with its dominating slots
//   bound_slot(v)            ping bounds slot for a label in slot v
//   bound_slack(l)           dwell still owed at l (added to the bound)
template <direction SearchDir, via_offset_t Vias>
struct via_criterion {
  static constexpr auto const kFwd = (SearchDir == direction::kForward);
  static constexpr auto const kN = static_cast<std::size_t>(Vias) + 1U;
  static constexpr auto const kFinalSlot = static_cast<std::size_t>(Vias);

  // one shared time-at-dest bound: only the final slot produces journeys
  static constexpr auto const kNDestBounds = std::size_t{1U};
  static constexpr std::size_t dest_bound_of(std::size_t) { return 0U; }
  static constexpr bool is_dest_slot(std::size_t const v) {
    return v == kFinalSlot;
  }
  template <typename Fn>
  static void for_each_dest_slot(Fn&& fn) {
    fn(kFinalSlot);
  }
  static constexpr std::uint8_t journey_slot(std::size_t) { return 0U; }

  // all slots live in the same timetable world (rt iff the search is rt)
  static constexpr auto const kAllSlotsRt = true;
  static constexpr bool slot_uses_rt(std::size_t) { return true; }
  static constexpr bool slot_boards_rt(std::size_t) { return true; }
  static constexpr auto const kFusedEt = false;
  static constexpr auto const kCopyOnDiverge = false;

  using bag_t = std::array<delta_t, kN>;

  struct transition {
    std::size_t target_;
    duration_t stay_;
  };

  static bool is_better(auto a, auto b) { return kFwd ? a < b : a > b; }

  via_criterion(std::array<bitvec, kMaxVias> const& is_via,
                std::vector<via_stop> const& via_stops)
      : is_via_{is_via}, via_stops_{via_stops} {
    assert(Vias == via_stops_.size());
  }

  std::size_t start_slot(std::size_t const l) const {
    return (Vias != 0 && is_via_[0][l]) ? 1U : 0U;
  }

  template <typename Fn>
  void for_each_start_slot(std::size_t const l, Fn&& fn) const {
    fn(start_slot(l));
  }

  transition stop_transition(std::size_t const v, std::size_t const l) const {
    auto const is_via =
        v != kFinalSlot && is_via_[v][static_cast<bitvec::size_type>(l)];
    return {is_via ? v + 1U : v, is_via ? via_stops_[v].stay_ : 0_minutes};
  }

  template <typename Fn>
  void for_each_final_stop_transition(std::size_t const l, Fn&& fn) const {
    if constexpr (Vias != 0U) {
      constexpr auto const v = static_cast<std::size_t>(Vias) - 1U;
      if (is_via_[v][static_cast<bitvec::size_type>(l)]) {
        fn(v, via_stops_[v].stay_);
      }
    }
  }

  // more progress dominates less at equal time
  static constexpr bool dominates(std::size_t const a, std::size_t const b) {
    return a >= b;
  }

  // chain order -> folding the immediate successor is transitive-complete
  void fold_bounds(bag_t& slots) const {
    if constexpr (Vias != 0U) {
      for (auto v = kN - 1U; v != 0U; --v) {
        slots[v - 1U] = kFwd ? std::min(slots[v - 1U], slots[v])
                             : std::max(slots[v - 1U], slots[v]);
      }
    }
  }

  static constexpr auto const kNBoundsSlots = kN;

  static constexpr std::size_t bound_slot(std::size_t const v) {
    return kFinalSlot - v;
  }

  int bound_slack(std::size_t const l) const {
    auto stays = 0;
    if constexpr (Vias != 0U) {
      for (auto w = std::size_t{0U}; w != Vias; ++w) {
        if (is_via_[w][static_cast<bitvec::size_type>(l)]) {
          stays += static_cast<int>(via_stops_[w].stay_.count());
        }
      }
    }
    return stays;
  }

  // et_[e][o]: a ride that entered in slot e and crossed o zero-dwell
  // transitions while on board, i.e. it arrives in slot e + o
  struct route_ride {
    void reset() { et_ = {}; }

    // zero-dwell transitions move a ride up one slot
    template <typename TimeAt>
    void advance(std::size_t const l, TimeAt&& time_at) {
      if constexpr (Vias != 0U) {
        for (auto e = 0U; e != Vias; ++e) {
          for (auto o = Vias - e; o != 0U; --o) {
            auto& from = et_[e][o - 1U];
            auto const cs = e + o - 1U;  // slot before the crossing
            if (from.is_valid() && c_.is_via_[cs][l] &&
                c_.via_stops_[cs].stay_ == 0_minutes) {
              auto& to = et_[e][o];
              if (!to.is_valid() || is_better(time_at(from), time_at(to))) {
                to = from;
              }
              from = {};
            }
          }
        }
      }
    }

    // all rides arriving in slot cs, highest slot first
    template <typename Fn>
    void for_each_arrival(Fn&& fn) const {
      for (auto j = 0U; j != kN; ++j) {
        auto const cs = kFinalSlot - j;
        for (auto e = 0U; e != cs + 1U; ++e) {
          auto const& r = et_[e][cs - e];
          if (!r.is_valid()) {
            continue;
          }
          fn(cs, r);
        }
      }
    }

    // fresh boardings from a slot-v label enter ride (v, 0)
    transport& fresh(std::size_t const v) { return et_[v][0]; }

    via_criterion const& c_;
    std::array<std::array<transport, kN>, kN> et_{};
  };

  // et_[v] = an entry point exists such that the ride is in slot v here
  struct rt_ride {
    void reset() { et_.fill(false); }

    void advance(std::size_t const l) {
      if constexpr (Vias != 0U) {
        for (auto v = std::size_t{Vias}; v != 0U; --v) {
          if (et_[v - 1U] && c_.is_via_[v - 1U][l] &&
              c_.via_stops_[v - 1U].stay_ == 0_minutes) {
            et_[v] = true;
            et_[v - 1U] = false;
          }
        }
      }
    }

    template <typename Fn>
    void for_each_arrival(Fn&& fn) const {
      for (auto j = 0U; j != kN; ++j) {
        auto const v = kFinalSlot - j;
        if (et_[v]) {
          fn(v);
        }
      }
    }

    void board(std::size_t const v) { et_[v] = true; }

    via_criterion const& c_;
    std::array<bool, kN> et_{};
  };

  std::array<bitvec, kMaxVias> const& is_via_;
  std::vector<via_stop> const& via_stops_;
};

}  // namespace nigiri::routing
