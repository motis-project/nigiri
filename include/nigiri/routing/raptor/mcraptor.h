#pragma once

#include <cassert>
#include <algorithm>
#include <array>
#include <limits>
#include <optional>
#include <span>
#include <type_traits>
#include <vector>

#include "cista/containers/bitvec.h"

#include "date/date.h"

#include "nigiri/common/delta_t.h"
#include "nigiri/routing/clasz_mask.h"
#include "nigiri/routing/journey.h"
#include "nigiri/routing/limits.h"
#include "nigiri/routing/pareto_set.h"
#include "nigiri/routing/query.h"
#include "nigiri/routing/raptor/bmrap_bounds.h"
#include "nigiri/routing/raptor/raptor_stats.h"
#include "nigiri/routing/transfer_time_settings.h"
#include "nigiri/types.h"

namespace nigiri {
struct timetable;
struct rt_timetable;
}  // namespace nigiri

namespace nigiri::routing {

// No via stops, no bike/car/reservation filters; rt and td offsets work.
bool mcraptor_supported(query const&, rt_timetable const*);

// Adds offset legs and the start footpath to the transit core legs (CPU + GPU).
template <direction SearchDir>
void mc_reconstruct_offsets(timetable const&,
                            rt_timetable const*,
                            query const&,
                            journey&);

// What a criterion may know about the trip a label just rode.
struct ride_attrs {
  clasz clasz_;
};

template <direction SearchDir>
constexpr bool arr_no_worse(delta_t const a, delta_t const b) {
  return SearchDir == direction::kForward ? a <= b : a >= b;
}

// Criteria protocol (the pareto label):
//   arr_                     arrival
//   dominates<Dir>           pareto dominance
//   completed_dominates<Dir> dominance at the destination (nothing follows)
//   carried, carry(), from_ride(arr, duration, ride_attrs, carried)
//                            what survives a boarding, and back
//   at_start(arr, ingress)   round 0 seed
//   with_transfer(dt), with_walk(dt, duration)
//   projected_to(arr)        arrival replaced by arr + lower bound
//   reuse_dominates<Dir>(o, dep, o_dep)  rRAPTOR reuse across departures
//   apply_to(journey&)       writes the criteria into the result
struct arr_criteria {
  template <direction SearchDir>
  bool dominates(arr_criteria const& o) const {
    return arr_no_worse<SearchDir>(arr_, o.arr_);
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
                                std::uint16_t,
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

  template <direction SearchDir>
  bool reuse_dominates(arr_criteria const& o, delta_t, delta_t) const {
    return dominates<SearchDir>(o);
  }

  delta_t arr_;
};

// Composable dimensions: the criteria protocol minus the arrival (arr_with owns
// it). Each writes its own journey slot, so they combine freely.

// Minutes on foot (offsets, footpaths), not the same-stop transfer buffer.
struct non_transit_dim {
  bool dominates(non_transit_dim const& o) const {
    return non_transit_ <= o.non_transit_;
  }
  bool completed_dominates(non_transit_dim const& o) const {
    return dominates(o);
  }
  static non_transit_dim at_start(std::uint16_t const ingress) {
    return {ingress};
  }
  static non_transit_dim from_ride(std::uint16_t,
                                   ride_attrs const&,
                                   non_transit_dim const& prev) {
    return prev;
  }
  non_transit_dim with_transfer(int) const { return *this; }
  non_transit_dim with_walk(int, std::uint16_t const duration) const {
    return {static_cast<std::uint16_t>(non_transit_ + duration)};
  }
  void apply_to(journey& j) const { j.criteria_cost_ = non_transit_; }
  bool operator==(non_transit_dim const&) const = default;

  std::uint16_t non_transit_{0U};
};

// "Uses an avoided vehicle class" as a pareto dimension, not a filter: keeps
// both the fastest itinerary and the best one without that class.
struct mode_filter_dim {
  // Avoided classes (default AIR), per query, hence thread_local (a search runs
  // on one thread). Set before every search; never reset.
  static clasz_mask_t& avoided_mask() {
    thread_local clasz_mask_t mask = to_mask(clasz::kAir);
    return mask;
  }
  static void set_avoided(clasz_mask_t const m) { avoided_mask() = m; }
  static bool is_avoided(clasz const c) {
    return is_allowed(avoided_mask(), c);
  }

  bool dominates(mode_filter_dim const& o) const {
    return mode_filter_ <= o.mode_filter_;
  }
  bool completed_dominates(mode_filter_dim const& o) const {
    return dominates(o);
  }
  static mode_filter_dim at_start(std::uint16_t) { return {false}; }
  static mode_filter_dim from_ride(std::uint16_t,
                                   ride_attrs const& ra,
                                   mode_filter_dim const& prev) {
    return {prev.mode_filter_ || is_avoided(ra.clasz_)};
  }
  mode_filter_dim with_transfer(int) const { return *this; }
  mode_filter_dim with_walk(int, std::uint16_t) const { return *this; }
  void apply_to(journey& j) const { j.criteria_mode_filter_ = mode_filter_; }
  bool operator==(mode_filter_dim const&) const = default;

  bool mode_filter_{false};
};

// Vehicle-class switches between trips; suburban, subway, tram and bus are one
// group. Labels in different classes only dominate with a switch to spare (the
// next switch may differ); completed_dominates drops that margin.
struct mode_switches_dim {
  // one past the last clasz doubles as "no trip ridden yet"
  static constexpr clasz no_clasz() { return clasz::kNumClasses; }

  static constexpr clasz group_of(clasz const c) {
    switch (c) {
      case clasz::kSuburban:
      case clasz::kSubway:
      case clasz::kTram: return clasz::kBus;
      default: return c;
    }
  }

  std::uint8_t switch_penalty(mode_switches_dim const& o) const {
    return (clasz_ == o.clasz_ || clasz_ == no_clasz()) ? 0U : 1U;
  }
  bool dominates(mode_switches_dim const& o) const {
    return switches_ + switch_penalty(o) <= o.switches_;
  }
  bool completed_dominates(mode_switches_dim const& o) const {
    return switches_ <= o.switches_;
  }
  static mode_switches_dim at_start(std::uint16_t) { return {no_clasz(), 0U}; }
  static mode_switches_dim from_ride(std::uint16_t,
                                     ride_attrs const& ra,
                                     mode_switches_dim const& prev) {
    auto const group = group_of(ra.clasz_);
    auto const switched = prev.clasz_ != no_clasz() && prev.clasz_ != group;
    return {group,
            static_cast<std::uint8_t>(prev.switches_ + (switched ? 1U : 0U))};
  }
  mode_switches_dim with_transfer(int) const { return *this; }
  mode_switches_dim with_walk(int, std::uint16_t) const { return *this; }
  void apply_to(journey& j) const { j.criteria_mode_switches_ = switches_; }
  bool operator==(mode_switches_dim const&) const = default;

  clasz clasz_{clasz::kNumClasses};
  std::uint8_t switches_{0U};
};

// Arrival plus any set of dimensions; strict pareto over all of them.
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
    return arr_no_worse<SearchDir>(arr_, o.arr_) &&
           all(d_, o.d_,
               [](auto const& x, auto const& y) { return x.dominates(y); });
  }

  template <direction SearchDir>
  bool completed_dominates(arr_with const& o) const {
    return arr_no_worse<SearchDir>(arr_, o.arr_) &&
           all(d_, o.d_, [](auto const& x, auto const& y) {
             return x.completed_dominates(y);
           });
  }

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
  // dimensions only grow, so they are their own lower bound
  arr_with projected_to(delta_t const arr) const { return {arr, d_}; }

  // dimensions do not depend on the departure: plain dominance
  template <direction SearchDir>
  bool reuse_dominates(arr_with const& o, delta_t, delta_t) const {
    return dominates<SearchDir>(o);
  }

  void apply_to(journey& j) const {
    [&]<std::size_t... I>(std::index_sequence<I...>) {
      (std::get<I>(d_).apply_to(j), ...);
    }(idx_t{});
  }

  delta_t arr_;
  dims_t d_;
};

template <typename Criteria>
struct basic_mcraptor_state {
  // breadcrumb index in the low 31 bits; the top bit marks transit arrivals
  // (only those get footpaths relaxed)
  static constexpr auto const kByRoute = std::uint32_t{1U} << 31U;
  static constexpr auto const kBreadcrumbMask = kByRoute - 1U;
  static constexpr auto const kNoBreadcrumb = kBreadcrumbMask;

  // The transit leg behind an arrival: the trip (a transport_idx_t on traffic
  // day day_, or an rt_transport_idx_t if day_ is invalid), its board and
  // alight stops, the arrival at the label's stop and the boarded label's
  // breadcrumb (kNoBreadcrumb = start).
  struct breadcrumb {
    std::uint32_t transport_;
    std::uint32_t parent_;
    day_idx_t day_;
    stop_idx_t board_;
    stop_idx_t alight_;
    delta_t arr_;
  };
  static_assert(sizeof(breadcrumb) == 16U);

  // One label of a stop's bag. round_ is an implicit pareto dimension (never
  // dominated by a higher round); dep_ tags the start time under range reuse,
  // where the bag outlives it.
  struct label {
    Criteria crit_;
    std::uint32_t breadcrumb_;
    std::uint8_t round_;
    delta_t dep_;
  };

  // per-location bag: inline buffer, arena overflow
  static constexpr auto const kInline = std::uint32_t{4U};
  static constexpr auto const kNoOverflow = std::uint32_t{0xFFFFFFFFU};

  // one cache line; over_ != kNoOverflow: labels at arena offset over_
  struct alignas(64) small_bag {
    std::uint16_t size_{0U};
    std::uint16_t cap_{static_cast<std::uint16_t>(kInline)};
    std::uint32_t over_{kNoOverflow};
    std::array<label, kInline> inline_;
  };

  // overflow spans in power-of-two classes with free lists; offsets, not
  // pointers, so the vector may grow
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

  // touched_ marks non-empty bags so clearing only visits those.
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

    // shrink to n labels; capacity unchanged
    void set_size(std::uint32_t const l, std::uint32_t const n) {
      bags_[l].size_ = static_cast<std::uint16_t>(n);
    }

    // append one label, moving to a larger arena span when full
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

  // one bag per stop for all rounds of a start time
  bag_layer bag_;
  std::vector<breadcrumb> breadcrumbs_;
  bitvec station_mark_;
  bitvec prev_station_mark_;
  bitvec route_mark_;
  // empty unless the query carries an rt_timetable with rt transports
  bitvec rt_transport_mark_;
};

// RangeReuse: keep the bags across start times (rRAPTOR, latest departure
// first) instead of clearing them per departure.
template <direction SearchDir, typename Criteria, bool RangeReuse = false>
struct basic_mcraptor {
  using state_t = basic_mcraptor_state<Criteria>;
  using algo_state_t = state_t;
  using algo_stats_t = raptor_stats;

  static constexpr bool kUseLowerBounds = true;
  // Whether a caller that always sets bounds_ (bmrap_profile.cc) still needs a
  // real lb array: no, effective_lb() ignores lb_ once bounded.
  static constexpr bool kNeedsLbWhenBounded = false;
  static constexpr auto const kFwd = (SearchDir == direction::kForward);
  static constexpr auto const kBwd = (SearchDir == direction::kBackward);
  static constexpr auto const kInvalid = kInvalidDelta<SearchDir>;
  static constexpr auto const kUnreachable =
      std::numeric_limits<std::uint16_t>::max();

  basic_mcraptor(
      timetable const&,
      rt_timetable const*,
      state_t&,
      bitvec& is_dest,
      std::array<bitvec, kMaxVias>& is_via,
      std::vector<std::uint16_t>& dist_to_dest,
      hash_map<location_idx_t, std::vector<td_offset>> const& td_dist_to_dest,
      std::vector<std::uint16_t>& lb,
      std::vector<via_stop> const& via_stops,
      day_idx_t base,
      clasz_mask_t allowed_claszes,
      bool require_bike_transport,
      bool require_car_transport,
      bool is_wheelchair,
      bool no_compulsory_reservation,
      transfer_time_settings const&,
      profile_idx_t prf_idx);

  algo_stats_t get_stats() const { return stats_; }

  void reset_arrivals();
  void next_start_time();
  void add_start(location_idx_t, unixtime_t, via_offset_t = 0U);

  void execute(unixtime_t start_time,
               std::uint8_t max_transfers,
               unixtime_t worst_time_at_dest,
               pareto_set<journey>& results);

  // BM-RAPTOR main search: bounds from the backward pruning search discard
  // arrivals that cannot reach the target within the remaining trip budget and
  // arrival slack. nullptr = plain McRAPTOR. Must share this search's base day.
  void set_bounds(bmrap_bounds const* b) { bounds_ = b; }

  // adds offset legs and the start footpath to execute()'s core legs
  void reconstruct(query const&, journey&);

private:
  static bool is_better(auto a, auto b) { return kFwd ? a < b : a > b; }
  static bool is_better_or_eq(auto a, auto b) { return kFwd ? a <= b : a >= b; }
  static auto get_best(auto a, auto b) { return is_better(a, b) ? a : b; }
  static auto dir(auto a) { return (kFwd ? 1 : -1) * a; }

  // route-scan label; key_ = traffic day << 16 | trip offset orders the trips
  struct route_label {
    transport t_;
    std::uint32_t key_;
    delta_t board_dep_;  // departure at the boarding stop (ride duration)
    stop_idx_t board_;
    std::uint32_t parent_;
    [[no_unique_address]] typename Criteria::carried carried_;
  };

  // rt transports are single trips: no trip order
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
  void update_footpaths(unsigned k);
  void collect_dest_journeys(unsigned k,
                             unixtime_t start_time,
                             pareto_set<journey>& results);
  journey materialize(location_idx_t dest,
                      unsigned k,
                      Criteria const&,
                      std::uint32_t breadcrumb_idx,
                      unixtime_t start_time);

  transport get_earliest_transport(route_idx_t,
                                   stop_idx_t,
                                   day_idx_t day_at_stop,
                                   minutes_after_midnight_t mam_at_stop,
                                   location_idx_t);
  std::uint32_t trip_order_key(route_idx_t, transport) const;
  static bool is_earlier_trip(std::uint32_t a, std::uint32_t b) {
    return kFwd ? a < b : a > b;
  }

  template <typename Label, typename Sections>
  bool alight(unsigned k,
              std::uint32_t l_idx,
              stop_idx_t,
              delta_t by_transport,
              Label const&,
              Sections const&,
              clasz fallback,
              std::uint32_t transport,
              day_idx_t day);

  bool merge_round(std::uint32_t l,
                   Criteria const&,
                   typename state_t::breadcrumb const&,
                   std::uint8_t round);
  delta_t transfer_buffer(std::uint64_t l) const;

  // An arrival at stop l in round k at time t is discarded iff t is worse than
  // tau_dep^<-(l, budget - k). `slack` loosens the bound for footpath arrivals,
  // which pay no transfer buffer at l: the stored bound is post-transfer, right
  // for transit arrivals but one buffer too tight for footpaths.
  bool bound_prunes(unsigned const k,
                    std::uint32_t const l,
                    delta_t const t,
                    int const slack = 0) const {
    if (bounds_ == nullptr) {
      return false;
    }
    // The current start time's budget limits the rounds, but the row is
    // relative to the MATRIX budget: compute_bounds() staggers an anchor run
    // with budget b into rows (matrix budget - b) + t, so row
    // matrix budget - k checks every anchor for its own b - k trips left.
    // Indexing with a smaller per-departure budget (range BMRAP's
    // max_transfers_fn_) shifted every row by the difference and pruned
    // anchors of later departures from their own bounds.
    if (k > std::min<unsigned>(cur_budget_, bounds_->budget_)) {
      return true;
    }
    return !is_better_or_eq(
        t,
        clamp(static_cast<int>(bounds_->at(bounds_->budget_ - k, l)) + slack));
  }

  // 0 is always a valid lower bound. Under BM-RAPTOR bound_prunes() is already
  // a tighter feasibility cutoff, so lb_ is only needed when unbounded.
  std::uint16_t effective_lb(std::uint32_t const l_idx) const {
    return bounds_ == nullptr ? lb_[l_idx] : std::uint16_t{0U};
  }

  // Destination pruning: a pareto frontier over (round, criteria); a candidate
  // is pruned iff its lb projection is dominated by an entry of no more rounds.
  // Persists across start times.
  struct dest_entry {
    unsigned k_;
    Criteria crit_;
  };

  bool dest_dominates(unsigned const k, Criteria const& projected) const {
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

  delta_t time_at_stop(route_idx_t, transport, stop_idx_t, event_type) const;
  // rt event times are absolute on the rt base day: no traffic-day arithmetic
  delta_t rt_time_at_stop(rt_transport_idx_t, stop_idx_t, event_type) const;
  // inactive if rt-updated: the rt scan covers it
  bool is_transport_active(transport_idx_t, day_idx_t) const;
  delta_t to_delta(day_idx_t day, std::int16_t mam) const;
  unixtime_t to_unix(delta_t) const;
  std::pair<day_idx_t, minutes_after_midnight_t> split(delta_t) const;
  bool is_intermodal_dest() const { return !dist_to_end_.empty(); }
  int as_int(day_idx_t const d) const { return static_cast<int>(d.v_); }

  timetable const& tt_;
  rt_timetable const* rtt_{nullptr};
  // runtime, not a template parameter: enough instantiations already
  bool has_rt_{false};
  std::uint32_t n_rt_transports_;
  state_t& state_;
  bitvec const& is_dest_;
  std::vector<std::uint16_t> const& dist_to_end_;
  hash_map<location_idx_t, std::vector<td_offset>> const& td_dist_to_end_;
  std::vector<std::uint16_t> const& lb_;
  // search-window bound only; dest_bag_ owns destination pruning
  delta_t worst_at_dest_;
  bmrap_bounds const* bounds_{nullptr};  // nullptr = plain McRAPTOR
  // trips (max_transfers + 1) of the start time being executed
  unsigned cur_budget_{0U};
  std::vector<dest_entry> dest_bag_;
  // current start's departure, tagged onto inserted labels (see label::dep_)
  delta_t cur_dep_{};
  // round-0 seeds, inserted in execute() once the ingress is known
  std::vector<std::pair<std::uint32_t, delta_t>> seeds_;
  day_idx_t base_;
  raptor_stats stats_;
  clasz_mask_t allowed_claszes_;
  bool is_wheelchair_;
  transfer_time_settings transfer_time_settings_;
  profile_idx_t prf_idx_;
  std::vector<route_label> route_bag_;
  std::vector<rt_label> rt_bag_;
  // departures of route_bag_ labels at the scanned stop (saves lookups)
  std::vector<delta_t> route_bag_dep_;
  // a stop's by-route entries, copied before footpath expansion inserts
  std::vector<typename state_t::label> fp_labels_;
  // journey materialization scratch (legs in search order)
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

using mcraptor_state = basic_mcraptor_state<arr_criteria>;

// Every dispatched combination of dimensions; instantiated in mcraptor.cc.
using arr_non_transit_criteria = arr_with<non_transit_dim>;
using arr_mode_filter_criteria = arr_with<mode_filter_dim>;
using arr_mode_switches_criteria = arr_with<mode_switches_dim>;
using arr_non_transit_mode_filter_criteria =
    arr_with<non_transit_dim, mode_filter_dim>;
using arr_non_transit_mode_switches_criteria =
    arr_with<non_transit_dim, mode_switches_dim>;
using arr_mode_filter_mode_switches_criteria =
    arr_with<mode_filter_dim, mode_switches_dim>;
using arr_non_transit_mode_filter_mode_switches_criteria =
    arr_with<non_transit_dim, mode_filter_dim, mode_switches_dim>;

using mcraptor_non_transit_state =
    basic_mcraptor_state<arr_non_transit_criteria>;
using mcraptor_mode_filter_state =
    basic_mcraptor_state<arr_mode_filter_criteria>;
using mcraptor_non_transit_mode_filter_state =
    basic_mcraptor_state<arr_non_transit_mode_filter_criteria>;
using mcraptor_mode_switches_state =
    basic_mcraptor_state<arr_mode_switches_criteria>;
using mcraptor_non_transit_mode_switches_state =
    basic_mcraptor_state<arr_non_transit_mode_switches_criteria>;
using mcraptor_mode_filter_mode_switches_state =
    basic_mcraptor_state<arr_mode_filter_mode_switches_criteria>;
using mcraptor_non_transit_mode_filter_mode_switches_state =
    basic_mcraptor_state<arr_non_transit_mode_filter_mode_switches_criteria>;

}  // namespace nigiri::routing
