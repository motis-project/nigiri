#pragma once

#include <cassert>
#include <algorithm>
#include <array>
#include <limits>
#include <optional>
#include <span>
#include <tuple>
#include <vector>

#include "cista/containers/bitvec.h"

#include "date/date.h"

#include "nigiri/common/delta_t.h"
#include "nigiri/routing/clasz_mask.h"
#include "nigiri/routing/journey.h"
#include "nigiri/routing/limits.h"
#include "nigiri/routing/mc_criteria.h"
#include "nigiri/routing/pareto_set.h"
#include "nigiri/routing/query.h"
#include "nigiri/routing/raptor/raptor_stats.h"
#include "nigiri/routing/transfer_time_settings.h"
#include "nigiri/td_footpath.h"
#include "nigiri/types.h"

namespace nigiri {
struct timetable;
struct rt_timetable;
}  // namespace nigiri

namespace nigiri::routing {

// No via stops, no bike/car/reservation filters; rt and td offsets work.
bool mcraptor_supported(query const&, rt_timetable const*);

// Minutes walked to a seed reached `elapsed` minutes after the query start: a
// td start offset may also wait, which is not walking.
template <direction SearchDir>
std::uint16_t seed_walk(td_offsets_t const* td_start,
                        location_idx_t const l,
                        unixtime_t const start_time,
                        std::uint16_t const elapsed) {
  if (td_start != nullptr) {
    if (auto const it = td_start->find(l); it != end(*td_start)) {
      auto const fp = get_td_duration<SearchDir>(it->second, start_time);
      if (fp.has_value() && fp->first.count() <= elapsed) {
        return static_cast<std::uint16_t>(fp->second.duration_.count());
      }
    }
  }
  return elapsed;
}

template <direction SearchDir>
constexpr bool arr_no_worse(delta_t const a, delta_t const b) {
  return SearchDir == direction::kForward ? a <= b : a >= b;
}

// Pareto dimensions besides the arrival, combined by arr_with. Each provides
// dominates, completed_dominates (at the destination: nothing follows),
// at_start, from_ride, with_walk and apply_to (its own journey field).

// Walking (offsets, footpaths; not the same-stop transfer buffer) in units of
// kNonTransitBin minutes, see non_transit_add().
struct non_transit_dim {
  bool dominates(non_transit_dim const& o) const {
    return non_transit_ <= o.non_transit_;
  }
  bool completed_dominates(non_transit_dim const& o) const {
    return dominates(o);
  }
  static non_transit_dim at_start(std::uint16_t const ingress) {
    return {static_cast<std::uint8_t>(non_transit_add(0U, ingress))};
  }
  static non_transit_dim from_ride(clasz, non_transit_dim const& prev) {
    return prev;
  }
  non_transit_dim with_walk(std::uint16_t const duration) const {
    return {static_cast<std::uint8_t>(non_transit_add(non_transit_, duration))};
  }
  void apply_to(journey& j) const {
    j.criteria_cost_ =
        static_cast<std::uint16_t>(non_transit_ * kNonTransitBin);
  }
  bool operator==(non_transit_dim const&) const = default;

  std::uint8_t non_transit_{0U};  // units
};

// "Uses an avoided vehicle class" as a pareto dimension, not a filter: keeps
// both the fastest itinerary and the best one without that class.
struct mode_filter_dim {
  // Avoided classes (default AIR), set per query; thread_local, as a search
  // runs on one thread.
  static clasz_mask_t& avoided_mask() {
    thread_local clasz_mask_t mask = to_mask(clasz::kAir);
    return mask;
  }
  static void set_avoided(clasz_mask_t const m) { avoided_mask() = m; }

  bool dominates(mode_filter_dim const& o) const {
    return mode_filter_ <= o.mode_filter_;
  }
  bool completed_dominates(mode_filter_dim const& o) const {
    return dominates(o);
  }
  static mode_filter_dim at_start(std::uint16_t) { return {false}; }
  static mode_filter_dim from_ride(clasz const c, mode_filter_dim const& prev) {
    return {prev.mode_filter_ || is_allowed(avoided_mask(), c)};
  }
  mode_filter_dim with_walk(std::uint16_t) const { return *this; }
  void apply_to(journey& j) const { j.criteria_mode_filter_ = mode_filter_; }
  bool operator==(mode_filter_dim const&) const = default;

  bool mode_filter_{false};
};

// Switches between vehicle class groups (mode_group()) from trip to trip.
// Labels in different groups only dominate with a switch to spare (the next
// switch may differ); completed_dominates drops that margin.
struct mode_switches_dim {
  bool dominates(mode_switches_dim const& o) const {
    return switches_dominate(group_, switches_, o.group_, o.switches_);
  }
  bool completed_dominates(mode_switches_dim const& o) const {
    return switches_ <= o.switches_;
  }
  static mode_switches_dim at_start(std::uint16_t) {
    return {static_cast<std::uint8_t>(kNoModeGroup), 0U};
  }
  static mode_switches_dim from_ride(clasz const c,
                                     mode_switches_dim const& prev) {
    auto const group = mode_group(c);
    return {static_cast<std::uint8_t>(group),
            static_cast<std::uint8_t>(
                switches_after_ride(prev.group_, prev.switches_, group))};
  }
  mode_switches_dim with_walk(std::uint16_t) const { return *this; }
  void apply_to(journey& j) const { j.criteria_mode_switches_ = switches_; }
  bool operator==(mode_switches_dim const&) const = default;

  std::uint8_t group_{static_cast<std::uint8_t>(kNoModeGroup)};
  std::uint8_t switches_{0U};
};

// The label: arrival plus any set of dimensions.
template <typename... Dims>
struct arr_with {
  using carried = std::tuple<Dims...>;  // what survives a boarding

  static bool carried_dominates([[maybe_unused]] carried const& a,
                                [[maybe_unused]] carried const& b) {
    return (std::get<Dims>(a).dominates(std::get<Dims>(b)) && ...);
  }
  template <direction SearchDir>
  bool dominates(arr_with const& o) const {
    return arr_no_worse<SearchDir>(arr_, o.arr_) && carried_dominates(d_, o.d_);
  }
  template <direction SearchDir>
  bool completed_dominates(arr_with const& o) const {
    return arr_no_worse<SearchDir>(arr_, o.arr_) &&
           (std::get<Dims>(d_).completed_dominates(std::get<Dims>(o.d_)) &&
            ...);
  }
  static arr_with from_ride(delta_t const arr,
                            [[maybe_unused]] clasz const c,
                            [[maybe_unused]] carried const& prev) {
    return {arr, carried{Dims::from_ride(c, std::get<Dims>(prev))...}};
  }
  static arr_with at_start(delta_t const arr,
                           [[maybe_unused]] std::uint16_t const ingress) {
    return {arr, carried{Dims::at_start(ingress)...}};
  }
  arr_with with_transfer(int const dt) const { return {clamp(arr_ + dt), d_}; }
  arr_with with_walk(int const dt,
                     [[maybe_unused]] std::uint16_t const duration) const {
    return {clamp(arr_ + dt),
            carried{std::get<Dims>(d_).with_walk(duration)...}};
  }
  // dimensions only grow, so they are their own lower bound
  arr_with projected_to(delta_t const arr) const { return {arr, d_}; }
  void apply_to([[maybe_unused]] journey& j) const {
    (std::get<Dims>(d_).apply_to(j), ...);
  }

  delta_t arr_;
  [[no_unique_address]] carried d_;
};

// Every dispatched combination of dimensions; instantiated in mcraptor.cc.
using arr_criteria = arr_with<>;
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

template <typename Criteria>
struct basic_mcraptor_state {
  // breadcrumb index in the low 31 bits; the top bit marks transit arrivals
  // (only those get footpaths relaxed)
  static constexpr auto const kByRoute = std::uint32_t{1U} << 31U;
  static constexpr auto const kBreadcrumbMask = kByRoute - 1U;
  static constexpr auto const kNoBreadcrumb = kBreadcrumbMask;

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

  // touched_ marks non-empty bags so clearing only visits those.
  struct bag_layer {
    void resize(std::size_t const n) {
      bags_.resize(n);
      touched_.resize(static_cast<std::uint32_t>(n));
    }

    label* data(small_bag& b) {
      return b.over_ == kNoOverflow ? b.inline_.data()
                                    : arena_.data() + b.over_;
    }
    label const* data(small_bag const& b) const {
      return b.over_ == kNoOverflow ? b.inline_.data()
                                    : arena_.data() + b.over_;
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

    // doubles: outgrown spans are only reclaimed by clear()
    void grow(small_bag& b) {
      auto const new_cap = std::uint32_t{b.cap_} * 2U;
      assert(new_cap <= std::numeric_limits<std::uint16_t>::max());
      auto const new_off = arena_size_;
      arena_size_ += new_cap;
      if (arena_size_ > arena_.size()) {
        arena_.resize(std::max<std::size_t>(arena_size_, arena_.size() * 2U));
      }
      auto const* const src = data(b);
      std::copy(src, src + b.size_, arena_.data() + new_off);
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
      arena_size_ = 0U;
    }

    std::vector<small_bag> bags_;
    bitvec touched_;
    std::vector<label> arena_;  // offsets, not pointers: it may grow
    std::uint32_t arena_size_{0U};
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
  // parent_ == kNoBreadcrumb: boarded at the start
  std::vector<mc_breadcrumb> breadcrumbs_;
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
  static constexpr auto const kFwd = (SearchDir == direction::kForward);
  static constexpr auto const kInvalid = kInvalidDelta<SearchDir>;
  static constexpr auto const kUnreachable =
      std::numeric_limits<std::uint16_t>::max();
  static constexpr auto const kArrEv =
      kFwd ? event_type::kArr : event_type::kDep;
  static constexpr auto const kDepEv =
      kFwd ? event_type::kDep : event_type::kArr;

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
  void set_td_start(td_offsets_t const& td) { td_start_ = &td; }

  void execute(unixtime_t start_time,
               std::uint8_t max_transfers,
               unixtime_t worst_time_at_dest,
               pareto_set<journey>& results);

  void reconstruct(query const&, journey&);

private:
  static bool is_better(auto a, auto b) { return kFwd ? a < b : a > b; }
  static bool is_better_or_eq(auto a, auto b) { return kFwd ? a <= b : a >= b; }
  static auto get_best(auto a, auto b) { return is_better(a, b) ? a : b; }
  static auto dir(auto a) { return (kFwd ? 1 : -1) * a; }

  // route-scan label; key_ orders the trips (trip_order_key, rt: a single trip)
  struct route_label {
    transport t_;
    std::uint32_t key_;
    stop_idx_t board_;
    std::uint32_t parent_;
    [[no_unique_address]] typename Criteria::carried carried_;
  };

  date::sys_days base() const;

  bool loop_routes(unsigned k);
  bool update_route(unsigned k, route_idx_t);
  bool loop_rt_transports(unsigned k);
  bool update_rt_transport(unsigned k, rt_transport_idx_t);
  template <typename Stops, typename Ridable, typename Alight, typename Board>
  bool scan_stops(unsigned k, Stops const&, Ridable&&, Alight&&, Board&&);
  void board(std::uint32_t key,
             transport,
             stop_idx_t,
             std::uint32_t parent,
             typename Criteria::carried const&);
  void update_footpaths(unsigned k);
  void collect_dest_journeys(unsigned k,
                             unixtime_t start_time,
                             pareto_set<journey>& results);

  transport get_earliest_transport(route_idx_t,
                                   stop_idx_t,
                                   day_idx_t day_at_stop,
                                   minutes_after_midnight_t mam_at_stop,
                                   location_idx_t);
  std::uint32_t trip_order_key(route_idx_t, transport) const;
  static bool is_earlier_trip(std::uint32_t a, std::uint32_t b) {
    return kFwd ? a < b : a > b;
  }

  bool alight(unsigned k,
              std::uint32_t l_idx,
              stop_idx_t,
              delta_t by_transport,
              route_label const&,
              clasz,
              std::uint32_t transport,
              day_idx_t day);

  template <typename Fn>
  void for_each_label(std::uint32_t l, unsigned round, Fn&&) const;
  bool bag_insert(std::uint32_t l,
                  Criteria const&,
                  std::uint32_t flagged_breadcrumb,
                  unsigned round,
                  bool by_route);
  bool merge_round(std::uint32_t l,
                   Criteria const&,
                   mc_breadcrumb const&,
                   unsigned round,
                   bool by_route);
  delta_t transfer_buffer(std::uint64_t l) const;

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
  state_t& state_;
  bitvec const& is_dest_;
  std::vector<std::uint16_t> const& dist_to_end_;
  hash_map<location_idx_t, std::vector<td_offset>> const& td_dist_to_end_;
  td_offsets_t const* td_start_{nullptr};
  std::vector<std::uint16_t> const& lb_;
  // search-window bound only; dest_bag_ owns destination pruning
  delta_t worst_at_dest_;
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
  // a stop's by-route entries, copied before footpath expansion inserts
  std::vector<typename state_t::label> fp_labels_;
  std::vector<mc_breadcrumb> chain_;
};

using mcraptor_state = basic_mcraptor_state<arr_criteria>;

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
