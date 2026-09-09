#pragma once

#include <cassert>
#include <cinttypes>
#include <cstdlib>
#include <algorithm>
#include <type_traits>
#include <array>
#include <limits>
#include <span>
#include <string_view>
#include <vector>

#include "cista/containers/bitvec.h"

#include "date/date.h"

#include "nigiri/common/delta_t.h"
#include "nigiri/routing/clasz_mask.h"
#include "nigiri/routing/journey.h"
#include "nigiri/routing/limits.h"
#include "nigiri/routing/pareto_set.h"
#include "nigiri/routing/raptor/bmrap_bounds.h"
#include "nigiri/routing/raptor/breadcrumb.h"
#include "nigiri/routing/query.h"
#include "nigiri/routing/raptor/raptor_stats.h"
#include "nigiri/routing/transfer_time_settings.h"
#include "nigiri/types.h"

namespace nigiri {
struct timetable;
struct rt_timetable;
}  // namespace nigiri

namespace nigiri::routing {

// McRAPTOR intentionally supports only the plain one-to-one case:
// no via stops and no bike/car transport requirements. Realtime
// (rt transports + time-dependent footpaths) and time-dependent
// first/last-mile offsets ARE supported, like plain raptor.
bool mcraptor_supported(query const&, rt_timetable const*);

// A criteria set: the pareto dimensions of a label. Each combination is a
// type, so every configuration gets its own fully inlined instantiation
// of the algorithm. Requirements:
//   arr_             the primary criterion (drives the scalar
//                    earliest-arrival pruning and journey extraction)
//   dominates<Dir>() pareto dominance over all criteria
//   carried          the criteria carried unchanged while riding a trip
//                    (everything except the arrival), with dominance and
//                    equality; carry()/from_ride(arr, ride duration, carried)
//                    convert between them. May be an empty type (arrival
//                    is the only criterion) - the algorithm does not
//                    distinguish, all configurations share one code path.
//   at_start(arr, ingress) round-0 criteria (ingress = duration between
//                    the query start and the seeded stop time: the
//                    start offset or start footpath)
//   with_transfer(dt) criteria after a same-station transfer of dt
//                    (dt is already signed by the search direction)
//   with_walk(dt, duration) criteria after a footpath/offset of dt
//   projected_to(arr) copy with the arrival replaced by an optimistic
//                    projection (arr + lower bound); all other criteria
//                    stay unchanged (= their trivial lower bound: they
//                    only grow). Used for destination pareto pruning.
//   apply_to(journey&) write the criteria into the journey so the result
//                    pareto set keeps the trade-offs
// What a criterion may know about the trip a label just rode. Passed to
// from_ride() so criteria can score the vehicle itself rather than only
// its timing.
struct ride_attrs {
  clasz clasz_;
};

struct arr_criteria {
  template <direction SearchDir>
  bool dominates(arr_criteria const& o) const {
    return SearchDir == direction::kForward ? arr_ <= o.arr_
                                            : arr_ >= o.arr_;
  }

  template <direction SearchDir>
  bool completed_dominates(arr_criteria const& o) const {
    return dominates<SearchDir>(o);
  }

  struct carried {
    template <direction SearchDir>
    bool dominates(carried const&) const {
      return true;
    }
    bool operator==(carried const&) const = default;
  };
  carried carry() const { return {}; }
  static arr_criteria from_ride(delta_t const arr,
                                std::uint16_t /* ride duration */,
                                ride_attrs const&,
                                carried const&) {
    return {arr};
  }
  static arr_criteria at_start(delta_t const arr, std::uint16_t) {
    return {arr};
  }
  arr_criteria with_transfer(int const dt) const { return {clamp(arr_ + dt)}; }
  arr_criteria with_walk(int const dt, std::uint16_t) const {
    return {clamp(arr_ + dt)};
  }
  arr_criteria projected_to(delta_t const arr) const { return {arr}; }
  void apply_to(journey&) const {}

  // cross-departure rRAPTOR reuse dominance (range-mcRAPTOR): a later-
  // departing label (this, dep) dominates an earlier one (o, o_dep) iff it
  // reaches the stop no later. The departure clause is vacuous for arrival-
  // only search (latest-first processing already guarantees dep>=o_dep).
  template <direction SearchDir>
  bool reuse_dominates(arr_criteria const& o, delta_t const /*dep*/,
                       delta_t const /*o_dep*/) const {
    return SearchDir == direction::kForward ? arr_ <= o.arr_ : arr_ >= o.arr_;
  }

  delta_t arr_;
};

// arrival time + generalized cost in weighted minutes (OTP defaults):
//   cost = elapsed time * 1   (transit/wait reluctance default 1.0:
//                              riding, waiting and transfer buffers)
//        + walking time * 1   (walkReluctance 2.0 minus the elapsed
//                              charge; ingress/footpaths/egress)
//        + boardings * 10     (boardCost default 600s)
// The elapsed part is (arr - start) for every label of one start time,
// so only the extras (walk surcharge + boardings) are stored.
//
// Dominance is the STRICT pareto over (arr, extras): earliness is never
// traded against extras. Comparing "cost so far" (arr + extras) instead
// would credit the earlier arrival at 1 weighted minute per minute -
// exact only if the journey ended at that stop. To stay comparable, the
// earlier label must be padded with the waiting-time penalty of the
// arrival difference (both may have to wait for the same connection);
// waiting is charged at the same rate as elapsed time, so the padding
// cancels the arrival terms and leaves the pure extras comparison. This
// also makes the in-bag rule consistent with the cross-departure
// reuse_dominates (departure-discounted extras), so range reuse is
// result-neutral for this criteria configuration.
struct arr_cost_criteria {
  // walk surcharge on top of the elapsed-time charge; total walk
  // reluctance = 1 (elapsed) + kWalkSurcharge. Default 1 = walkReluctance
  // 2.0 (OTP default). Env-tunable for reluctance-sweep experiments.
  static inline std::uint32_t const kWalkSurcharge = [] {
    auto const* v = std::getenv("NIGIRI_WALK_SURCHARGE");
    return v == nullptr ? std::uint32_t{1U}
                        : static_cast<std::uint32_t>(std::atoi(v));
  }();
  static constexpr auto const kBoardCost = std::uint32_t{10U};  // minutes

  template <direction SearchDir>
  bool dominates(arr_cost_criteria const& o) const {
    constexpr auto const kF = SearchDir == direction::kForward;
    return (kF ? arr_ <= o.arr_ : arr_ >= o.arr_) && cost_ <= o.cost_;
  }

  // dominance for COMPLETED journeys (the destination frontier): at the
  // destination the elapsed part is realized, so (arr, arr + extras) IS
  // the journey-level cost dominance - and it stays valid against the
  // lb-projection of an intermediate label (every completion has
  // arr_f >= projected arr and extras_f >= extras). Label-to-label
  // comparison inside stop bags must NOT use this (see dominates()).
  template <direction SearchDir>
  bool completed_dominates(arr_cost_criteria const& o) const {
    constexpr auto const kF = SearchDir == direction::kForward;
    return (kF ? arr_ <= o.arr_ : arr_ >= o.arr_) &&
           (kF ? arr_ + cost_ <= o.arr_ + o.cost_
               : -arr_ + cost_ <= -o.arr_ + o.cost_);
  }

  struct carried {
    // same boarding: identical future arrivals, totals differ by extras
    template <direction SearchDir>
    bool dominates(carried const& o) const {
      return cost_ <= o.cost_;
    }
    bool operator==(carried const&) const = default;
    std::uint16_t cost_;
  };
  carried carry() const { return {cost_}; }
  static arr_cost_criteria from_ride(delta_t const arr,
                                     std::uint16_t /* ride duration */,
                                     ride_attrs const&,
                                     carried const& c) {
    return {arr, static_cast<std::uint16_t>(c.cost_ + kBoardCost)};
  }
  static arr_cost_criteria at_start(delta_t const arr,
                                    std::uint16_t const ingress) {
    return {arr, static_cast<std::uint16_t>(ingress * kWalkSurcharge)};
  }
  arr_cost_criteria with_transfer(int const dt) const {
    return {clamp(arr_ + dt), cost_};
  }
  arr_cost_criteria with_walk(int const dt,
                              std::uint16_t const duration) const {
    return {clamp(arr_ + dt),
            static_cast<std::uint16_t>(cost_ + duration * kWalkSurcharge)};
  }
  arr_cost_criteria projected_to(delta_t const arr) const {
    return {arr, cost_};
  }

  // cross-departure rRAPTOR reuse dominance (range-mcRAPTOR). A later-
  // departing label (this, dep) dominates an earlier-departing one (o,
  // o_dep) iff it reaches the stop no later (feasibility - it can catch
  // every onward trip) AND its departure-discounted extras are no worse.
  // The cost clause uses (extras - dep), NOT (arr + extras): the elapsed
  // charge is departure-dependent, so an earlier absolute arrival only
  // buys longer downstream waiting, not lower final generalized cost. With
  // the final journey pareto over (dep, arr, transfers, gen_cost), this
  // guarantees the later start's completion dominates the earlier's.
  template <direction SearchDir>
  bool reuse_dominates(arr_cost_criteria const& o, delta_t const dep,
                       delta_t const o_dep) const {
    constexpr auto const kF = SearchDir == direction::kForward;
    return (kF ? arr_ <= o.arr_ : arr_ >= o.arr_) &&
           (kF ? static_cast<int>(cost_) - dep <=
                     static_cast<int>(o.cost_) - o_dep
               : static_cast<int>(cost_) + dep <=
                     static_cast<int>(o.cost_) + o_dep);
  }

  void apply_to(journey& j) const {
    // full generalized cost: the common elapsed part + stored extras
    j.criteria_cost_ = static_cast<std::uint16_t>(
        cost_ + static_cast<std::uint16_t>(j.travel_time().count()));
  }

  delta_t arr_;
  std::uint16_t cost_;  // extras only: walk surcharge + boarding penalties
};

// WALKING minutes: the third criterion of the classic multicriteria RAPTOR
// and of the restricted-pareto paper.
//
// Counted: the ingress offset / start footpath, every footpath relaxation
// and the intermodal egress offset. NOT counted: the same-station transfer
// buffer, which is minimum change time - waiting, not walking - and the
// riding and waiting time the arrival criterion already carries.
// ===== COMPOSABLE CRITERIA =====
//
// A DIMENSION is the criteria protocol minus the arrival time, which
// arr_with<> owns. Adding an optimization axis is then one ~20-line struct
// plus an alias, rather than hand-writing its cross-product with the others.
//
//   dominates(o)            in-bag rule; may price the FUTURE (clasz_dim)
//   completed_dominates(o)  rule at the destination, where nothing follows
//   at_start(ingress)       round-0 value
//   from_ride(dur, ra, prev)  value after boarding a trip
//   with_transfer(dt)       value after a same-stop transfer
//   with_walk(dt, dur)      value after a footpath / offset
//   apply_to(journey&)      write the realized value into the result
//
// Each dimension owns a DISTINCT journey slot, which is what makes them
// freely combinable; that is also why the generalized-cost criterion is
// not a dimension - it writes criteria_cost_, the slot walking uses.

// minutes on foot: offsets + footpaths
struct walk_dim {
  bool dominates(walk_dim const& o) const { return walk_ <= o.walk_; }
  bool completed_dominates(walk_dim const& o) const { return dominates(o); }
  static walk_dim at_start(std::uint16_t const ingress) { return {ingress}; }
  static walk_dim from_ride(std::uint16_t,
                            ride_attrs const&,
                            walk_dim const& prev) {
    return prev;
  }
  walk_dim with_transfer(int) const { return *this; }
  walk_dim with_walk(int, std::uint16_t const duration) const {
    return {static_cast<std::uint16_t>(walk_ + duration)};
  }
  void apply_to(journey& j) const { j.criteria_cost_ = walk_; }
  bool operator==(walk_dim const&) const = default;

  std::uint16_t walk_{0U};
};

// binary "uses an avoided vehicle class" (flights by default). The point is
// that this is a pareto dimension rather than a filter: the result keeps
// BOTH the fast itinerary that flies and the best one that does not.
struct air_dim {
  // NIGIRI_MC_AVOID_CLASZ takes a comma-separated list of clasz names
  // ("AIR", "SUBWAY", ...) so the same dimension can express "prefer to
  // avoid X" for any class.
  static inline clasz_mask_t const kAvoided = [] {
    auto const* const v = std::getenv("NIGIRI_MC_AVOID_CLASZ");
    if (v == nullptr) {
      return to_mask(clasz::kAir);
    }
    auto mask = clasz_mask_t{0U};
    auto const str = std::string_view{v};
    for (auto pos = std::size_t{0U}; pos < str.size();) {
      auto end = str.find(',', pos);
      if (end == std::string_view::npos) {
        end = str.size();
      }
      auto const name = str.substr(pos, end - pos);
      for (auto i = std::uint8_t{0U}; i != kNumClasses; ++i) {
        if (to_str(static_cast<clasz>(i)) == name) {
          mask |= to_mask(static_cast<clasz>(i));
        }
      }
      pos = end + 1U;
    }
    return mask == 0U ? to_mask(clasz::kAir) : mask;
  }();
  static bool is_avoided(clasz const c) { return is_allowed(kAvoided, c); }

  bool dominates(air_dim const& o) const { return air_ <= o.air_; }
  bool completed_dominates(air_dim const& o) const { return dominates(o); }
  static air_dim at_start(std::uint16_t) { return {false}; }
  static air_dim from_ride(std::uint16_t,
                           ride_attrs const& ra,
                           air_dim const& prev) {
    return {prev.air_ || is_avoided(ra.clasz_)};
  }
  air_dim with_transfer(int) const { return *this; }
  air_dim with_walk(int, std::uint16_t) const { return *this; }
  void apply_to(journey& j) const { j.criteria_air_ = air_; }
  bool operator==(air_dim const&) const = default;

  bool air_{false};
};

// number of VEHICLE CLASS SWITCHES between consecutive trips (bus ->
// subway counts, subway -> subway does not), so it rewards journeys that
// stay within one mode.
//
// The carried clasz is not baggage - it prices the FUTURE: a label with the
// same switch count in a different class may still cost one more switch
// downstream, so it may only dominate when a full switch ahead. A label
// that has ridden nothing yet boards anything for free. At the destination
// nothing follows, so completed_dominates drops the penalty.
struct clasz_dim {
  // clasz has no invalid value of its own, so the one-past-the-end
  // enumerator doubles as "no trip ridden yet"
  static constexpr clasz no_clasz() { return clasz::kNumClasses; }

  std::uint8_t switch_penalty(clasz_dim const& o) const {
    return (clasz_ == o.clasz_ || clasz_ == no_clasz()) ? 0U : 1U;
  }
  bool dominates(clasz_dim const& o) const {
    return switches_ + switch_penalty(o) <= o.switches_;
  }
  bool completed_dominates(clasz_dim const& o) const {
    return switches_ <= o.switches_;
  }
  static clasz_dim at_start(std::uint16_t) { return {no_clasz(), 0U}; }
  static clasz_dim from_ride(std::uint16_t,
                             ride_attrs const& ra,
                             clasz_dim const& prev) {
    auto const switched =
        prev.clasz_ != no_clasz() && prev.clasz_ != ra.clasz_;
    return {ra.clasz_,
            static_cast<std::uint8_t>(prev.switches_ + (switched ? 1U : 0U))};
  }
  clasz_dim with_transfer(int) const { return *this; }
  clasz_dim with_walk(int, std::uint16_t) const { return *this; }
  void apply_to(journey& j) const { j.criteria_clasz_ = switches_; }
  bool operator==(clasz_dim const&) const = default;

  clasz clasz_{clasz::kNumClasses};
  std::uint8_t switches_{0U};
};

// Arrival time plus any set of dimensions. Dominance is the strict pareto
// over (arrival, every dimension): earliness is never traded away.
template <typename... Dims>
struct arr_with {
  using dims_t = std::tuple<Dims...>;
  using idx_t = std::index_sequence_for<Dims...>;

  template <typename F>
  static dims_t make(F&& f) {
    return [&]<std::size_t... I>(std::index_sequence<I...>) {
      return dims_t{f.template operator()<I>()...};
    }(idx_t{});
  }
  template <typename F>
  static bool all(dims_t const& a, dims_t const& b, F&& f) {
    return [&]<std::size_t... I>(std::index_sequence<I...>) {
      return (f(std::get<I>(a), std::get<I>(b)) && ...);
    }(idx_t{});
  }

  template <direction SearchDir>
  bool dominates(arr_with const& o) const {
    constexpr auto const kF = SearchDir == direction::kForward;
    return (kF ? arr_ <= o.arr_ : arr_ >= o.arr_) &&
           all(d_, o.d_, [](auto const& x, auto const& y) {
             return x.dominates(y);
           });
  }

  template <direction SearchDir>
  bool completed_dominates(arr_with const& o) const {
    constexpr auto const kF = SearchDir == direction::kForward;
    return (kF ? arr_ <= o.arr_ : arr_ >= o.arr_) &&
           all(d_, o.d_, [](auto const& x, auto const& y) {
             return x.completed_dominates(y);
           });
  }

  // everything except the arrival: what survives a boarding
  struct carried {
    template <direction SearchDir>
    bool dominates(carried const& o) const {
      return all(d_, o.d_,
                 [](auto const& x, auto const& y) { return x.dominates(y); });
    }
    bool operator==(carried const&) const = default;
    dims_t d_;
  };
  carried carry() const { return {d_}; }

  static arr_with from_ride(delta_t const arr,
                            std::uint16_t const ride_duration,
                            ride_attrs const& ra,
                            carried const& c) {
    return {arr, make([&]<std::size_t I>() {
              return std::tuple_element_t<I, dims_t>::from_ride(
                  ride_duration, ra, std::get<I>(c.d_));
            })};
  }
  static arr_with at_start(delta_t const arr, std::uint16_t const ingress) {
    return {arr, make([&]<std::size_t I>() {
              return std::tuple_element_t<I, dims_t>::at_start(ingress);
            })};
  }
  arr_with with_transfer(int const dt) const {
    return {clamp(arr_ + dt), make([&]<std::size_t I>() {
              return std::get<I>(d_).with_transfer(dt);
            })};
  }
  arr_with with_walk(int const dt, std::uint16_t const duration) const {
    return {clamp(arr_ + dt), make([&]<std::size_t I>() {
              return std::get<I>(d_).with_walk(dt, duration);
            })};
  }
  // all dimensions only ever grow, so their trivial lower bound is
  // themselves; only the arrival is projected
  arr_with projected_to(delta_t const arr) const { return {arr, d_}; }

  // every dimension is absolute (independent of when you departed), so the
  // cross-departure rRAPTOR reuse rule is the plain in-bag dominance
  template <direction SearchDir>
  bool reuse_dominates(arr_with const& o,
                       delta_t const /*dep*/,
                       delta_t const /*o_dep*/) const {
    return dominates<SearchDir>(o);
  }

  void apply_to(journey& j) const {
    [&]<std::size_t... I>(std::index_sequence<I...>) {
      (std::get<I>(d_).apply_to(j), ...);
    }(idx_t{});
  }

  // dimension access by type, for tracing and tests
  template <typename D>
  static constexpr bool has() {
    return (std::is_same_v<D, Dims> || ...);
  }
  template <typename D>
    requires(has<D>())
  D const& get() const {
    return std::get<D>(d_);
  }

  delta_t arr_;
  dims_t d_;
};

// Tracing helper: the walking minutes of any criteria that carries a walk
// dimension, 0 for the ones that do not.
template <typename C>
unsigned walk_of(C const& c) {
  if constexpr (requires { c.template get<walk_dim>(); }) {
    return c.template get<walk_dim>().walk_;
  } else {
    return 0U;
  }
}

template <typename Criteria>
struct basic_mcraptor_state {
  // bag entries store the arena breadcrumb index in the lower 31 bits; the
  // top bit marks entries produced by a transit arrival (their footpaths
  // still need to be relaxed - footpath arrivals never chain)
  static constexpr auto const kByRoute = std::uint32_t{1U} << 31U;
  static constexpr auto const kBreadcrumbMask = kByRoute - 1U;
  static constexpr auto const kNoBreadcrumb = kBreadcrumbMask;

  // Breadcrumb: everything needed to emit the transit leg that produced an
  // arrival and to continue the chase at its boarding stop. Uses the gouda
  // GPU raptor's packed encoding (see breadcrumb.h): transport + board +
  // alight fold into one 48-bit payload; the traffic day and the
  // footpath/transfer are recovered at reconstruction time, not stored.
  //   payload_ packed (transport_idx, board stop_idx, alight stop_idx)
  //   parent_  arena index of the boarded breadcrumb (kNoBreadcrumb = start).
  //            Kept explicitly (unlike the GPU raptor) because a pareto set
  //            has several labels per (round, stop), so (k-1, board) is
  //            ambiguous.
  //   arr_     arrival time at the bag location (drives the chase and the
  //            traffic-day recovery)
  struct breadcrumb {
    breadcrumb_t payload_;
    std::uint32_t parent_;
    delta_t arr_;
  };

  // Bag member: one pareto label (a frontier point of a stop's bag).
  // breadcrumb_ links the full breadcrumb in breadcrumbs_ (kNoBreadcrumb = round-0 start).
  // round_ is the transfer round that produced the label: it makes the
  // transfer count an implicit pareto dimension so a single accumulating
  // bag can replace the swapped prev_/curr_ round layers and the best_
  // gate. Boarding round k reads round_ == k-1; footpath/dest collection
  // read round_ == k. A candidate is never rejected/evicted by a
  // lower-round (fewer-transfer) label, so no pareto-optimal label is lost.
  // dep_ is the departure (query start time) that produced the label. With
  // range reuse the bag persists across start times (rRAPTOR); entries of
  // an earlier start are pruned by dominating later-departing entries
  // (departure-aware dominance) and boarding/footpath/dest collection only
  // read entries of the current start. Without reuse the bag is cleared per
  // start, so all entries share one departure and dep_ is inert.
  struct label {
    Criteria crit_;
    std::uint32_t breadcrumb_;
    std::uint8_t round_;
    delta_t dep_;
  };

  // One pareto bag per location (TREX: DataStructures/RAPTOR/Entities/Bags.h),
  // stored as a small inline buffer with an arena overflow. Measured on
  // Germany, a stop's bag holds mean 1.9 / p95 4 / max 18 labels, so kInline
  // covers ~97% of bags without touching the arena.
  static constexpr auto const kInline = std::uint32_t{4U};
  static constexpr auto const kNoOverflow = std::uint32_t{0xFFFFFFFFU};

  // Exactly one cache line (64 B). Metadata first so a small bag's header and
  // its first labels share cache line 0. over_ == kNoOverflow => labels live in
  // inline_; else at arena offset over_, capacity cap_.
  struct alignas(64) small_bag {
    std::uint16_t size_{0U};
    std::uint16_t cap_{static_cast<std::uint16_t>(kInline)};
    std::uint32_t over_{kNoOverflow};
    std::array<label, kInline> inline_;
  };

  // Bump arena for overflow spans with power-of-two size classes and a
  // per-class free list (cf. cista paged.h). small_bag stores an offset (not a
  // pointer) so the backing vector can grow without invalidating live bags;
  // the whole arena is reset when the bags are cleared per start time.
  struct arena {
    static constexpr auto const kMinClass = std::uint32_t{8U};
    static constexpr auto const kNumClasses = std::size_t{10U};  // 8 .. 4096

    static std::uint32_t class_of(std::uint32_t const cap) {
      auto idx = std::uint32_t{0U};
      while ((kMinClass << idx) < cap) {
        ++idx;
      }
      return idx;
    }
    std::uint32_t alloc(std::uint32_t const cap) {  // cap: pow2 >= kMinClass
      auto const idx = class_of(cap);
      assert(idx < kNumClasses);
      if (!free_[idx].empty()) {
        auto const off = free_[idx].back();
        free_[idx].pop_back();
        return off;
      }
      auto const off = bump_;
      bump_ += cap;
      if (bump_ > data_.size()) {
        data_.resize(std::max<std::size_t>(bump_, data_.size() * 2U));
      }
      return off;
    }
    void free_span(std::uint32_t const off, std::uint32_t const cap) {
      free_[class_of(cap)].push_back(off);
    }
    void reset() {
      bump_ = 0U;
      for (auto& f : free_) {
        f.clear();
      }
    }
    label* ptr(std::uint32_t const off) { return data_.data() + off; }
    label const* ptr(std::uint32_t const off) const {
      return data_.data() + off;
    }

    std::vector<label> data_;
    std::uint32_t bump_{0U};
    std::array<std::vector<std::uint32_t>, kNumClasses> free_{};
  };

  // touched_ marks non-empty bags so clearing between start times only visits
  // those (word-skipping) instead of sweeping all locations.
  struct bag_layer {
    void resize(std::size_t const n) {
      bags_.resize(n);
      touched_.resize(static_cast<std::uint32_t>(n));
    }

    label* data(small_bag& b) {
      return b.over_ == kNoOverflow ? b.inline_.data() : arena_.ptr(b.over_);
    }
    label const* data(small_bag const& b) const {
      return b.over_ == kNoOverflow ? b.inline_.data() : arena_.ptr(b.over_);
    }

    std::span<label> span(std::uint32_t const l) {
      auto& b = bags_[l];
      return {data(b), b.size_};
    }
    std::span<label const> span(std::uint32_t const l) const {
      auto const& b = bags_[l];
      return {data(b), b.size_};
    }

    bool empty(std::uint32_t const l) const { return bags_[l].size_ == 0U; }

    // shrink to n labels (n <= current size); capacity unchanged
    void set_size(std::uint32_t const l, std::uint32_t const n) {
      bags_[l].size_ = static_cast<std::uint16_t>(n);
    }

    // append one label, moving to a larger arena span if the buffer is full
    void push_back(std::uint32_t const l, label const& x) {
      auto& b = bags_[l];
      if (b.size_ == b.cap_) {
        grow(b);
      }
      data(b)[b.size_++] = x;
    }

    void grow(small_bag& b) {
      auto const new_cap = std::uint32_t{b.cap_} * 2U;  // 4->8->16->...
      auto const new_off = arena_.alloc(new_cap);  // may resize arena_.data_
      auto* const dst = arena_.ptr(new_off);
      auto const* const src = data(b);
      std::copy(src, src + b.size_, dst);
      if (b.over_ != kNoOverflow) {
        arena_.free_span(b.over_, b.cap_);
      }
      b.over_ = new_off;
      b.cap_ = static_cast<std::uint16_t>(new_cap);
    }

    void clear() {
      touched_.for_each_set_bit([&](std::size_t const l) {
        auto& b = bags_[l];
        b.size_ = 0U;
        b.cap_ = static_cast<std::uint16_t>(kInline);
        b.over_ = kNoOverflow;
      });
      touched_.zero_out();
      arena_.reset();
    }

    std::vector<small_bag> bags_;
    bitvec touched_;
    arena arena_;
  };

  basic_mcraptor_state() = default;
  basic_mcraptor_state(basic_mcraptor_state const&) = delete;
  basic_mcraptor_state& operator=(basic_mcraptor_state const&) = delete;
  basic_mcraptor_state(basic_mcraptor_state&&) = default;
  basic_mcraptor_state& operator=(basic_mcraptor_state&&) = default;
  ~basic_mcraptor_state() = default;

  basic_mcraptor_state& resize(unsigned const n_locations,
                               unsigned const n_routes,
                               unsigned const n_rt_transports) {
    bag_.resize(n_locations);
    station_mark_.resize(n_locations);
    prev_station_mark_.resize(n_locations);
    route_mark_.resize(n_routes);
    rt_transport_mark_.resize(n_rt_transports);
    return *this;
  }

  // One accumulating pareto bag per stop over all rounds of one start time
  // (folds the former prev_/curr_ round layers and the best_ cross-round
  // gate into a single container). Each label carries its transfer round
  // (see label::round_), making the transfer count an implicit pareto
  // dimension; boarding, footpath expansion and destination collection
  // filter by round. Reset per start time.
  bag_layer bag_;
  std::vector<breadcrumb> breadcrumbs_;
  bitvec station_mark_;
  bitvec prev_station_mark_;
  bitvec route_mark_;
  // empty unless the query carries an rt_timetable with rt transports
  bitvec rt_transport_mark_;
};

// RangeReuse: keep the per-stop bags across start times (rRAPTOR reuse,
// raptor_alenex.pdf 4.2) instead of clearing them each departure. Compile-time
// so each driver fixes it: false for the pong forward ping / plain EA (start
// times target different arrivals), true for the pong backward validation and
// the search.h interval search (fixed destination, latest-departure first).
template <direction SearchDir, typename Criteria, bool RangeReuse = false>
struct basic_mcraptor {
  using state_t = basic_mcraptor_state<Criteria>;
  using algo_state_t = state_t;
  using algo_stats_t = raptor_stats;

  static constexpr bool kUseLowerBounds = true;
  static constexpr auto const kFwd = (SearchDir == direction::kForward);
  static constexpr auto const kBwd = (SearchDir == direction::kBackward);
  static constexpr auto const kInvalid = kInvalidDelta<SearchDir>;
  static constexpr auto const kUnreachable =
      std::numeric_limits<std::uint16_t>::max();

  basic_mcraptor(timetable const&,
                 rt_timetable const*,
                 state_t&,
           bitvec& is_dest,
           std::array<bitvec, kMaxVias>& is_via,
           std::vector<std::uint16_t>& dist_to_dest,
           hash_map<location_idx_t, std::vector<td_offset>> const&
               td_dist_to_dest,
           std::vector<std::uint16_t>& lb,
           std::vector<via_stop> const& via_stops,
           day_idx_t base,
           clasz_mask_t allowed_claszes,
           bool require_bike_transport,
           bool require_car_transport,
           bool is_wheelchair,
           transfer_time_settings const&);

  algo_stats_t get_stats() const { return stats_; }

  void reset_arrivals();
  void next_start_time();
  void add_start(location_idx_t, unixtime_t);

  void execute(unixtime_t start_time,
               std::uint8_t max_transfers,
               unixtime_t worst_time_at_dest,
               profile_idx_t prf_idx,
               pareto_set<journey>& results);

  // tight starts (pong ping): the ping runs its whole window as ONE step,
  // so the step start contains waiting until each journey's first boarding
  // and result-pareto decisions in the step frame collapse dep-normalized
  // cost-pareto variants (later-departing-but-cheaper journeys look
  // dominated). With tight starts, collect_dest_journeys re-anchors every
  // journey at its latest feasible departure (first boarding minus the
  // minimum ingress walk) before the result-pareto add - exactly the
  // journey search.h would report, because its steps ARE the concrete
  // departures (zero wait before the first boarding by construction; the
  // arrival side is always tight). The pong re-derives departures anyway,
  // so ping start times are anchor-internal.
  void set_tight_start() { tight_start_ = true; }

  // BM-RAPTOR main search (restricted pareto sets,
  // doi:10.1137/1.9781611975499.5): bounds computed by the backward
  // pruning search discard every arrival that cannot reach the target
  // within the remaining trip budget and the arrival slack. nullptr (the
  // default) = plain unbounded McRAPTOR. The bounds must be built on the
  // same base day as this search.
  void set_bounds(bmrap_bounds const* b) { bounds_ = b; }

  // Core legs are materialized by execute() (breadcrumb chase); this only
  // adds first/last-mile offset legs and the start footpath.
  void reconstruct(query const&, journey&);

private:
  static bool is_better(auto a, auto b) { return kFwd ? a < b : a > b; }
  static bool is_better_or_eq(auto a, auto b) { return kFwd ? a <= b : a >= b; }
  static auto get_best(auto a, auto b) { return is_better(a, b) ? a : b; }
  static auto dir(auto a) { return (kFwd ? 1 : -1) * a; }

  // Label boarded onto a route while scanning it: pareto frontier over
  // the transport order. board_/parent_ are the breadcrumb pieces.
  // key_ = (traffic day << 16 | transport offset in route): lexicographic
  // comparison yields the total trip order within a route.
  struct route_label {
    transport t_;
    std::uint32_t key_;
    delta_t board_dep_;  // departure at the boarding stop (ride duration)
    stop_idx_t board_;
    std::uint32_t parent_;
    [[no_unique_address]] typename Criteria::carried carried_;
  };

  // Label boarded onto an rt transport. Same role as route_label minus the
  // trip identity: an rt transport IS a single trip, so the total trip order
  // route_label::key_ encodes is constant here and the frontier degenerates
  // to a plain pareto set over the carried criteria.
  struct rt_label {
    delta_t board_dep_;
    stop_idx_t board_;
    std::uint32_t parent_;
    [[no_unique_address]] typename Criteria::carried carried_;
  };

  date::sys_days base() const;

  bool loop_routes(unsigned k);
  bool update_route(unsigned k, route_idx_t);
  bool loop_rt_transports(unsigned k);
  bool update_rt_transport(unsigned k, rt_transport_idx_t);
  void update_footpaths(unsigned k, profile_idx_t);
  void collect_dest_journeys(unsigned k,
                             unixtime_t start_time,
                             pareto_set<journey>& results);
  journey materialize(location_idx_t dest,
                      unsigned k,
                      Criteria const&,
                      std::uint32_t breadcrumb_idx,
                      unixtime_t start_time);
  unixtime_t tighten_start(std::uint32_t breadcrumb_idx,
                           unixtime_t step_start);

  transport get_earliest_transport(route_idx_t,
                                   stop_idx_t,
                                   day_idx_t day_at_stop,
                                   minutes_after_midnight_t mam_at_stop,
                                   location_idx_t);
  std::uint32_t trip_order_key(route_idx_t, transport) const;
  static bool is_earlier_trip(std::uint32_t a, std::uint32_t b) {
    return kFwd ? a < b : a > b;
  }

  bool merge_round(std::uint32_t l,
                   Criteria const&,
                   typename state_t::breadcrumb const&,
                   std::uint8_t round);
  delta_t transfer_buffer(std::uint64_t l) const;

  // BM-RAPTOR bound check (the paper's main search): an arrival at stop l
  // in round k with time t is discarded iff t is worse than
  // tau_dep^<-(l, budget - k). `slack` shifts the bound into the looser
  // direction for arrivals that do not have to pay l's transfer buffer
  // before boarding (footpath arrivals): the pruning search stores the
  // post-transfer value, which is exactly the right reference for transit
  // arrivals but transfer_time too tight for footpath arrivals.
  bool bound_prunes(unsigned const k,
                    std::uint32_t const l,
                    delta_t const t,
                    int const slack = 0) const {
    if (bounds_ == nullptr) {
      return false;
    }
    // the trip budget of the CURRENT start time (execute's max_transfers),
    // not the matrix-wide maximum: under BM-RAPTOR every departure has its
    // own budget floor(sigma_tr * K(d)), and the matrix is only sized for
    // the largest one in the window
    auto const budget =
        std::min<unsigned>(cur_budget_, bounds_->budget_);
    if (k > budget) {
      return true;
    }
    return !is_better_or_eq(
        t, clamp(static_cast<int>(bounds_->at(budget - k, l)) + slack));
  }

  // Destination pruning (OTP: HeuristicsProvider.qualify +
  // DestinationArrivalPaths): scalar earliest-arrival tightening is wrong
  // once there is more than one criterion - a label arriving later can
  // still complete a pareto-optimal (cheaper) journey. Instead, all
  // destination arrivals form a pareto frontier over (round, criteria) and
  // candidates are pruned iff their optimistic projection (arrival + lower
  // bound, other criteria unchanged) is dominated by a label with fewer
  // or equal rounds. Persists across start times (starts are processed in
  // dominance order, so entries always stem from dominating start times).
  struct dest_entry {
    unsigned k_;
    Criteria crit_;
  };

  bool dest_dominates(unsigned const k, Criteria const& projected) const {
    // validation switch: disable destination pruning to verify it is
    // non-lossy (results must stay identical, only slower)
    static auto const kDisabled =
        std::getenv("NIGIRI_NO_DEST_PRUNING") != nullptr;
    if (kDisabled) {
      return false;
    }
    for (auto const& e : dest_bag_) {
      if (e.k_ <= k &&
          e.crit_.template completed_dominates<SearchDir>(projected)) {
        return true;
      }
    }
    return false;
  }

  void dest_bag_add(unsigned const k, Criteria const& crit) {
    auto removed = std::size_t{0U};
    for (auto i = std::size_t{0U}; i != dest_bag_.size(); ++i) {
      if (dest_bag_[i].k_ <= k &&
          dest_bag_[i].crit_.template completed_dominates<SearchDir>(crit)) {
        return;
      }
      if (k <= dest_bag_[i].k_ &&
          crit.template completed_dominates<SearchDir>(dest_bag_[i].crit_)) {
        ++removed;
        continue;
      }
      dest_bag_[i - removed] = dest_bag_[i];
    }
    dest_bag_.resize(dest_bag_.size() - removed + 1U);
    dest_bag_.back() = {k, crit};
  }

  delta_t time_at_stop(route_idx_t,
                       transport,
                       stop_idx_t,
                       event_type) const;
  // rt event times are stored absolute on the rt base day, so there is no
  // traffic-day arithmetic (mirrors raptor.h's rt_time_at_stop)
  delta_t rt_time_at_stop(rt_transport_idx_t, stop_idx_t, event_type) const;
  // rt-aware traffic-day test: with an rt_timetable a static transport that
  // got an rt update reads as inactive, so the static route scan skips it and
  // the rt scan below picks the updated run up (raptor.h is_transport_active)
  bool is_transport_active(transport_idx_t, day_idx_t) const;
  delta_t to_delta(day_idx_t day, std::int16_t mam) const;
  unixtime_t to_unix(delta_t) const;
  std::pair<day_idx_t, minutes_after_midnight_t> split(delta_t) const;
  bool is_intermodal_dest() const { return !dist_to_end_.empty(); }
  int as_int(day_idx_t const d) const { return static_cast<int>(d.v_); }

  template <typename T>
  auto get_begin_it(T const& t) {
    if constexpr (kFwd) {
      return t.begin();
    } else {
      return t.rbegin();
    }
  }

  template <typename T>
  auto get_end_it(T const& t) {
    if constexpr (kFwd) {
      return t.end();
    } else {
      return t.rend();
    }
  }

  timetable const& tt_;
  rt_timetable const* rtt_{nullptr};
  // rtt_ != nullptr: realtime is live for this query. Runtime rather than a
  // template parameter (unlike raptor.h's Rt) because mcraptor already
  // instantiates one algorithm per criteria configuration x direction x range
  // reuse - doubling that for a branch this predictable is not worth the
  // compile time. The GPU raptor makes the same call.
  bool has_rt_{false};
  std::uint32_t n_locations_, n_routes_, n_rt_transports_;
  state_t& state_;
  bitvec end_reachable_;
  bitvec const& is_dest_;
  std::vector<std::uint16_t> const& dist_to_end_;
  hash_map<location_idx_t, std::vector<td_offset>> const& td_dist_to_end_;
  std::vector<std::uint16_t> const& lb_;
  // pure search-window bound (never journey-tightened - the dest_bag_
  // pareto frontier owns all destination pruning)
  delta_t worst_at_dest_;
  // restricted-pareto pruning bounds, nullptr = plain McRAPTOR
  bmrap_bounds const* bounds_{nullptr};
  // trip budget of the start time currently being executed (in trips, i.e.
  // max_transfers + 1); indexes the bound rows
  unsigned cur_budget_{0U};
  std::vector<dest_entry> dest_bag_;
  // current start's departure (query start time in delta units); tagged
  // onto every label inserted this execute so range reuse can compare
  // labels across departures (see label::dep_).
  delta_t cur_dep_{};
  bool tight_start_{false};  // see set_tight_start()
  // this execute's round-0 seeds (stop, seeded stop time in delta units),
  // recorded by add_start and inserted with the ingress walking duration at
  // the top of execute (the query start time is only known then). A list -
  // not a re-merge over the bag - because the bag persists across starts
  // under range reuse and must not be cleared at seeded stops.
  std::vector<std::pair<std::uint32_t, delta_t>> seeds_;
  day_idx_t base_;
  raptor_stats stats_;
  clasz_mask_t allowed_claszes_;
  bool is_wheelchair_;
  transfer_time_settings transfer_time_settings_;
  std::vector<route_label> route_bag_;
  std::vector<rt_label> rt_bag_;
  // departure times of route_bag_ labels at the stop currently scanned
  // (avoids repeated event time lookups)
  std::vector<delta_t> route_bag_dep_;
  // scratch: a stop's by-route entries, copied before footpath expansion
  // (expansion inserts into the same layer)
  std::vector<typename state_t::label> fp_labels_;
  // scratch buffer for journey materialization (legs in search order)
  struct rec_leg {
    bool is_footpath_;
    location_idx_t from_, to_;
    delta_t dep_, arr_;
    // exactly one of the two is valid on a transit leg
    transport_idx_t t_;
    rt_transport_idx_t rt_;
    day_idx_t day_;
    stop_idx_t enter_, exit_;
    std::uint16_t fp_duration_;
  };
  std::vector<rec_leg> rec_legs_;
};

// The default configuration: arrival time only (the raptor.h baseline
// comparison). Additional criteria combinations get their own aliases +
// explicit instantiations in mcraptor.cc.
using mcraptor_state = basic_mcraptor_state<arr_criteria>;

template <direction SearchDir>
using mcraptor = basic_mcraptor<SearchDir, arr_criteria>;

// arrival + generalized cost configuration
using mcraptor_cost_state = basic_mcraptor_state<arr_cost_criteria>;

template <direction SearchDir>
using mcraptor_cost = basic_mcraptor<SearchDir, arr_cost_criteria>;

// Every dispatched combination of the dimensions above - one alias each,
// no hand-written types.
using arr_walk_criteria = arr_with<walk_dim>;
using arr_air_criteria = arr_with<air_dim>;
using arr_clasz_criteria = arr_with<clasz_dim>;
using arr_walk_air_criteria = arr_with<walk_dim, air_dim>;
using arr_walk_clasz_criteria = arr_with<walk_dim, clasz_dim>;
using arr_air_clasz_criteria = arr_with<air_dim, clasz_dim>;
using arr_walk_air_clasz_criteria = arr_with<walk_dim, air_dim, clasz_dim>;

using mcraptor_walk_state = basic_mcraptor_state<arr_walk_criteria>;
using mcraptor_air_state = basic_mcraptor_state<arr_air_criteria>;
using mcraptor_walk_air_state = basic_mcraptor_state<arr_walk_air_criteria>;
using mcraptor_clasz_state = basic_mcraptor_state<arr_clasz_criteria>;
using mcraptor_walk_clasz_state =
    basic_mcraptor_state<arr_walk_clasz_criteria>;
using mcraptor_air_clasz_state = basic_mcraptor_state<arr_air_clasz_criteria>;
using mcraptor_walk_air_clasz_state =
    basic_mcraptor_state<arr_walk_air_clasz_criteria>;

template <direction SearchDir>
using mcraptor_walk = basic_mcraptor<SearchDir, arr_walk_criteria>;

template <direction SearchDir>
using mcraptor_air = basic_mcraptor<SearchDir, arr_air_criteria>;

template <direction SearchDir>
using mcraptor_clasz = basic_mcraptor<SearchDir, arr_clasz_criteria>;

}  // namespace nigiri::routing
