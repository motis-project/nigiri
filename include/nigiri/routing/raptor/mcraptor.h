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

#include "utl/helpers/algorithm.h"

#include "nigiri/common/delta_t.h"
#include "nigiri/routing/clasz_mask.h"
#include "nigiri/routing/journey.h"
#include "nigiri/routing/limits.h"
#include "nigiri/routing/mc_criteria.h"
#include "nigiri/routing/pareto_set.h"
#include "nigiri/routing/query.h"
#include "nigiri/routing/raptor/bmrap_bounds.h"
#include "nigiri/routing/raptor/raptor_stats.h"
#include "nigiri/routing/transfer_time_settings.h"
#include "nigiri/routing/transfers.h"
#include "nigiri/rt/rt_timetable.h"
#include "nigiri/td_footpath.h"
#include "nigiri/types.h"

namespace nigiri {
struct timetable;
struct rt_timetable;
}  // namespace nigiri

namespace nigiri::routing {

// No via stops, no bike/car/reservation filters; rt, td offsets and transfer
// rules (virtual locations, rt virts) work.
bool mcraptor_supported(query const&, rt_timetable const*);

// The query's start, for seed_walk() (set_start_stops / set_td_start).
struct mc_start {
  std::vector<offset> const* offsets_{nullptr};
  location_match_mode mode_{location_match_mode::kExact};
  td_offsets_t const* td_{nullptr};
};

// Minutes walked to a seed reached `elapsed` minutes after the query start: a
// td start offset may also wait, which is not walking, and a location of a
// start stop (its base or a virtual location) is reached by a transfer.
template <direction SearchDir>
std::uint16_t seed_walk(timetable const& tt,
                        rt_timetable const* rtt,
                        mc_start const& start,
                        location_idx_t const l,
                        unixtime_t const start_time,
                        std::uint16_t const elapsed) {
  if (start.offsets_ != nullptr &&
      start.mode_ != location_match_mode::kIntermodal) {
    auto const b = rt::base(tt, rtt, l);
    if (utl::any_of(*start.offsets_, [&](offset const& o) {
          return rt::base(tt, rtt, o.target()) == b;
        })) {
      return 0U;
    }
  }
  auto const* const td_start = start.td_;
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

// The arrival and the criteria Dims (mc_criteria.h), packed as on the GPU.
template <typename... Dims>
struct arr_with {
  using extras_t = mc_extras<Dims...>;
  using carried = std::uint16_t;  // what survives a boarding: the extras

  static bool carried_dominates(carried const a, carried const b) {
    return extras_t::dominates(a, b);
  }
  template <direction SearchDir>
  bool dominates(arr_with const& o) const {
    return arr_no_worse<SearchDir>(arr_, o.arr_) &&
           extras_t::dominates(extras_, o.extras_);
  }
  template <direction SearchDir>
  bool completed_dominates(arr_with const& o) const {
    return arr_no_worse<SearchDir>(arr_, o.arr_) &&
           extras_t::completed_dominates(extras_, o.extras_);
  }
  static arr_with from_ride(delta_t const arr,
                            clasz const c,
                            carried const prev,
                            mc_context const& ctx) {
    return {arr, static_cast<carried>(extras_t::after_ride(prev, c, ctx))};
  }
  static arr_with at_start(delta_t const arr, std::uint16_t const ingress) {
    return {arr, static_cast<carried>(extras_t::at_start(ingress))};
  }
  arr_with with_transfer(int const dt) const {
    return {clamp(arr_ + dt), extras_};
  }
  arr_with with_walk(int const dt, std::uint16_t const duration) const {
    return {clamp(arr_ + dt),
            static_cast<carried>(extras_t::after_walk(extras_, duration))};
  }
  // this prefix joined with a suffix label at the same stop fits j
  // the criteria only grow, so they are their own lower bound
  arr_with projected_to(delta_t const arr) const { return {arr, extras_}; }
  void apply_to(journey& j) const { extras_t::to_journey(extras_, j); }
  bool operator==(arr_with const&) const = default;
  bool join_fits(arr_with const& suffix, journey const& j) const {
    return extras_t::join_fits(extras_, suffix.extras_,
                               extras_t::of_journey(j));
  }

  delta_t arr_;
  carried extras_{0U};
};

using arr_criteria = arr_with<>;
using arr_non_transit_criteria = arr_with<non_transit_dim>;
using arr_avoided_modes_criteria = arr_with<avoided_modes_dim>;
using arr_mode_switches_criteria = arr_with<mode_switches_dim>;
using arr_non_transit_avoided_modes_criteria =
    arr_with<non_transit_dim, avoided_modes_dim>;
using arr_non_transit_mode_switches_criteria =
    arr_with<non_transit_dim, mode_switches_dim>;
using arr_avoided_modes_mode_switches_criteria =
    arr_with<avoided_modes_dim, mode_switches_dim>;
using arr_non_transit_avoided_modes_mode_switches_criteria =
    arr_with<non_transit_dim, avoided_modes_dim, mode_switches_dim>;

template <typename Criteria>
struct basic_mcraptor_state {
  static constexpr auto const kNoBreadcrumb = (std::uint32_t{1U} << 31U) - 1U;

  // round_ is an implicit Pareto dimension; dep_ tags the start time (the
  // bags outlive next_start_time())
  struct label {
    Criteria crit_;
    std::uint32_t breadcrumb_ : 31;  // kNoBreadcrumb: a start label
    std::uint32_t by_route_ : 1;  // transit arrival: gets footpaths relaxed
    std::uint8_t round_;
    delta_t dep_;
  };

  // per-location bag: inline buffer, arena overflow
  static constexpr auto const kInline = std::uint32_t{4U};
  static constexpr auto const kNoOverflow = std::uint32_t{0xFFFFFFFFU};

  // one cache line; labels at arena offset over_ once grown
  struct alignas(64) small_bag {
    std::uint16_t size_{0U};
    std::uint16_t cap_{static_cast<std::uint16_t>(kInline)};
    std::uint32_t over_{kNoOverflow};
    std::array<label, kInline> inline_;
  };

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

    void set_size(std::uint32_t const l, std::uint32_t const n) {
      bags_[l].size_ = static_cast<std::uint16_t>(n);
    }

    void push_back(std::uint32_t const l, label const& x) {
      auto& b = bags_[l];
      if (b.size_ == b.cap_) {
        grow(b);
      }
      data(b)[b.size_++] = x;
    }

    // outgrown spans are only reclaimed by clear()
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
    std::vector<label> arena_;
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

  bag_layer bag_;  // all rounds of a start time
  std::vector<mc_breadcrumb> breadcrumbs_;
  bitvec station_mark_;
  bitvec prev_station_mark_;
  bitvec route_mark_;
  bitvec rt_transport_mark_;
};

// a search's labels per stop, copied before the next search reuses the state
template <typename Criteria>
struct mc_label_snapshot {
  struct entry {
    Criteria crit_;
    std::uint8_t round_;
  };
  static constexpr auto kNone = std::numeric_limits<std::uint32_t>::max();

  void take(basic_mcraptor_state<Criteria> const& s) {
    for (auto const l : stops_) {
      slot_[l] = kNone;
    }
    stops_.clear();
    ranges_.clear();
    labels_.clear();
    slot_.resize(s.bag_.bags_.size(), kNone);
    s.bag_.touched_.for_each_set_bit([&](auto const i) {
      auto const l = static_cast<std::uint32_t>(i);
      auto const bag = s.bag_.span(l);
      if (bag.empty()) {
        return;
      }
      slot_[l] = static_cast<std::uint32_t>(ranges_.size());
      stops_.push_back(l);
      ranges_.emplace_back(static_cast<std::uint32_t>(labels_.size()),
                           static_cast<std::uint32_t>(bag.size()));
      for (auto const& e : bag) {
        labels_.push_back({e.crit_, e.round_});
      }
    });
  }

  std::span<entry const> at(std::uint32_t const l) const {
    if (l >= slot_.size() || slot_[l] == kNone) {
      return {};
    }
    auto const [from, n] = ranges_[slot_[l]];
    return {labels_.data() + from, n};
  }

  std::vector<std::uint32_t> slot_, stops_;
  std::vector<std::pair<std::uint32_t, std::uint32_t>> ranges_;  // from, n
  std::vector<entry> labels_;
};

// A search keeps only the labels the oracle calls feasible; `t` is the raw
// arrival at l after k trips.
template <typename Criteria>
struct mc_oracle {
  virtual ~mc_oracle() = default;
  virtual bool feasible(std::uint32_t l,
                        unsigned k,
                        delta_t t,
                        Criteria const& prefix) const = 0;
};

// As raptor's round times, the bags outlive next_start_time() (rRAPTOR, latest
// departure first; labels of earlier start times only prune) and
// reset_arrivals() clears them.
template <direction SearchDir, typename Criteria>
struct basic_mcraptor {
  using state_t = basic_mcraptor_state<Criteria>;
  using algo_state_t = state_t;
  using algo_stats_t = raptor_stats;

  static constexpr bool kUseLowerBounds = true;
  static constexpr bool kNeedsLbWhenBounded = false;  // see effective_lb()
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
      profile_idx_t prf_idx,
      mc_context const& = {});

  algo_stats_t get_stats() const { return stats_; }

  void reset_arrivals();
  void next_start_time();
  void add_start(location_idx_t, unixtime_t, via_offset_t = 0U);
  void set_td_start(td_offsets_t const& td) { start_.td_ = &td; }
  void set_start_stops(std::vector<offset> const& offsets,
                       location_match_mode const mode) {
    start_.offsets_ = &offsets;
    start_.mode_ = mode;
  }

  void execute(
      unixtime_t start_time,
      std::uint8_t max_transfers,
      std::array<unixtime_t, kMaxTransfers + 2U> const& worst_time_at_dest,
      pareto_set<journey>& results);

  // BM-RAPTOR bounds on this search's base day; nullptr: plain McRAPTOR
  void set_bounds(bmrap_bounds const* b) { bounds_ = b; }

  void set_oracle(mc_oracle<Criteria> const* o) { oracle_ = o; }

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
    typename Criteria::carried carried_;
  };

  date::sys_days base() const;

  bool loop_routes(unsigned k);
  bool update_route(unsigned k, route_idx_t);
  bool loop_rt_transports(unsigned k);
  bool update_rt_transport(unsigned k, rt_transport_idx_t);
  // virts: the rt virts of the stops (rt_stop_virts), empty: the stops
  template <typename Stops, typename Ridable, typename Alight, typename Board>
  bool scan_stops(unsigned k,
                  Stops const&,
                  std::span<location_idx_t const> virts,
                  Ridable&&,
                  Alight&&,
                  Board&&);
  void board(std::uint32_t key,
             transport,
             stop_idx_t,
             std::uint32_t parent,
             typename Criteria::carried);
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
  location_idx_t base_of(std::uint32_t const l) const {
    return rt::base(tt_, rtt_, location_idx_t{l});
  }
  // transfer rules ban the same-stop transfer: arrivals by transport only walk
  bool no_transfer(std::uint32_t const l) const {
    return get_transfer_time(tt_, rtt_, location_idx_t{l}) ==
           kNoTransferAllowed;
  }

  // t worse than tau_dep^<-(l, trips left)? The bound is post-transfer:
  // footpath arrivals pay no buffer and pass it as `slack`.
  bool bound_prunes(unsigned const k,
                    std::uint32_t const l,
                    delta_t const t,
                    int const slack = 0) const {
    if (bounds_ == nullptr) {
      return false;
    }
    // row: see bmrap_bounds::staggered_
    auto const budget = std::min<unsigned>(cur_budget_, bounds_->budget_);
    if (k > budget) {
      return true;
    }
    auto const row = bounds_->staggered_ ? bounds_->budget_ - k : budget - k;
    return !is_better_or_eq(
        t, clamp(static_cast<int>(bounds_->at(row, l)) + slack));
  }

  // bounded, bound_prunes() is the tighter cutoff
  std::uint16_t effective_lb(std::uint32_t const l_idx) const {
    return bounds_ == nullptr ? lb_[l_idx] : std::uint16_t{0U};
  }

  // destination frontier over (round, criteria), across start times
  struct dest_entry {
    unsigned k_;
    Criteria crit_;
  };

  // Strict: a tie survives, so the bags keep every label that can complete to
  // one of the results (what guided searches join against).
  bool dest_dominates(unsigned const k, Criteria const& projected) const {
    for (auto const& e : dest_bag_) {
      if (e.k_ <= k &&
          e.crit_.template completed_dominates<SearchDir>(projected) &&
          !(k <= e.k_ &&
            projected.template completed_dominates<SearchDir>(e.crit_))) {
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
  mc_start start_;
  std::vector<std::uint16_t> const& lb_;
  delta_t worst_at_dest_;  // search window only, see dest_bag_
  bmrap_bounds const* bounds_{nullptr};
  mc_oracle<Criteria> const* oracle_{nullptr};
  unsigned cur_budget_{0U};  // trips of the current start time
  std::vector<dest_entry> dest_bag_;
  delta_t cur_dep_{};
  // inserted in execute() once the ingress is known
  std::vector<std::pair<std::uint32_t, delta_t>> seeds_;
  day_idx_t base_;
  raptor_stats stats_;
  clasz_mask_t allowed_claszes_;
  mc_context ctx_;
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
using mcraptor_avoided_modes_state =
    basic_mcraptor_state<arr_avoided_modes_criteria>;
using mcraptor_non_transit_avoided_modes_state =
    basic_mcraptor_state<arr_non_transit_avoided_modes_criteria>;
using mcraptor_mode_switches_state =
    basic_mcraptor_state<arr_mode_switches_criteria>;
using mcraptor_non_transit_mode_switches_state =
    basic_mcraptor_state<arr_non_transit_mode_switches_criteria>;
using mcraptor_avoided_modes_mode_switches_state =
    basic_mcraptor_state<arr_avoided_modes_mode_switches_criteria>;
using mcraptor_non_transit_avoided_modes_mode_switches_state =
    basic_mcraptor_state<arr_non_transit_avoided_modes_mode_switches_criteria>;

}  // namespace nigiri::routing
