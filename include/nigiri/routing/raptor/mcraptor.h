#pragma once

#include <cassert>
#include <cinttypes>
#include <cstdlib>
#include <algorithm>
#include <type_traits>
#include <array>
#include <limits>
#include <span>
#include <vector>

#include "cista/containers/bitvec.h"

#include "date/date.h"

#include "nigiri/common/delta_t.h"
#include "nigiri/routing/clasz_mask.h"
#include "nigiri/routing/journey.h"
#include "nigiri/routing/limits.h"
#include "nigiri/routing/pareto_set.h"
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
// no realtime, no via stops, no time-dependent offsets/footpaths and
// no bike/car transport requirements.
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
      touched_.resize(n);
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
                               unsigned const n_routes) {
    bag_.resize(n_locations);
    station_mark_.resize(n_locations);
    prev_station_mark_.resize(n_locations);
    route_mark_.resize(n_routes);
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

  date::sys_days base() const;

  bool loop_routes(unsigned k);
  bool update_route(unsigned k, route_idx_t);
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
  std::uint32_t n_locations_, n_routes_;
  state_t& state_;
  bitvec end_reachable_;
  bitvec const& is_dest_;
  std::vector<std::uint16_t> const& dist_to_end_;
  std::vector<std::uint16_t> const& lb_;
  // pure search-window bound (never journey-tightened - the dest_bag_
  // pareto frontier owns all destination pruning)
  delta_t worst_at_dest_;
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
    transport_idx_t t_;
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

}  // namespace nigiri::routing
