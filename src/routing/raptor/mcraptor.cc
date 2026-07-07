#include "nigiri/routing/raptor/mcraptor.h"

#include <algorithm>

#include "utl/erase_if.h"
#include "utl/helpers/algorithm.h"
#include "utl/verify.h"

#include "nigiri/common/it_range.h"
#include "nigiri/common/linear_lower_bound.h"
#include "nigiri/for_each_meta.h"
#include "nigiri/routing/raptor/reconstruct.h"
#include "nigiri/rt/rt_timetable.h"
#include "nigiri/special_stations.h"
#include "nigiri/timetable.h"

namespace nigiri::routing {

constexpr auto const kIntermodalTarget =
    get_special_station(special_station::kEnd);

bool mcraptor_supported(query const& q, rt_timetable const* rtt) {
  auto const rt_ok =
      rtt == nullptr ||
      (rtt->n_rt_transports() == 0U &&
       (q.prf_idx_ == 0U || (!rtt->has_td_footpaths_out_[q.prf_idx_].any() &&
                             !rtt->has_td_footpaths_in_[q.prf_idx_].any())));
  return rt_ok && q.td_start_.empty() && q.td_dest_.empty() &&
         !q.require_bike_transport_ && !q.require_car_transport_ &&
         q.via_stops_.empty();
}

template <direction SearchDir, typename Criteria>
basic_mcraptor<SearchDir, Criteria>::basic_mcraptor(
    timetable const& tt,
    rt_timetable const* rtt,
    state_t& state,
    bitvec& is_dest,
    std::array<bitvec, kMaxVias>& is_via,
    std::vector<std::uint16_t>& dist_to_dest,
    hash_map<location_idx_t, std::vector<td_offset>> const& td_dist_to_dest,
    std::vector<std::uint16_t>& lb,
    std::vector<via_stop> const& via_stops,
    day_idx_t const base,
    clasz_mask_t const allowed_claszes,
    bool const require_bike_transport,
    bool const require_car_transport,
    bool const is_wheelchair,
    transfer_time_settings const& tts)
    : tt_{tt},
      rtt_{rtt},
      n_locations_{tt_.n_locations()},
      n_routes_{tt_.n_routes()},
      state_{state.resize(n_locations_, n_routes_)},
      is_dest_{is_dest},
      dist_to_end_{dist_to_dest},
      lb_{lb},
      base_{base},
      allowed_claszes_{allowed_claszes},
      is_wheelchair_{is_wheelchair},
      transfer_time_settings_{tts} {
  static_cast<void>(is_via);
  utl::verify(via_stops.empty(), "mcraptor: via stops not supported");
  utl::verify(td_dist_to_dest.empty(),
              "mcraptor: time-dependent offsets not supported");
  utl::verify(!require_bike_transport && !require_car_transport,
              "mcraptor: bike/car transport not supported");
  utl::verify(rtt == nullptr || rtt->n_rt_transports() == 0U,
              "mcraptor: realtime not supported");
  reset_arrivals();
  if (!dist_to_end_.empty()) {
    end_reachable_.resize(n_locations_);
    for (auto i = 0U; i != dist_to_end_.size(); ++i) {
      if (dist_to_end_[i] != kUnreachable) {
        end_reachable_.set(i, true);
      }
    }
  }
}

template <direction SearchDir, typename Criteria>
date::sys_days basic_mcraptor<SearchDir, Criteria>::base() const {
  return tt_.internal_interval_days().from_ + as_int(base_) * date::days{1};
}

// per-slice / per-query reset: the bag and its arena start empty (with
// range reuse the bag persists across start times *within* a slice, so it
// is cleared only here, not in next_start_time).
template <direction SearchDir, typename Criteria>
void basic_mcraptor<SearchDir, Criteria>::reset_arrivals() {
  worst_at_dest_ = kInvalid;
  dest_bag_.clear();
  state_.bag_.clear();
  state_.breadcrumbs_.clear();
  seeds_.clear();
}

template <direction SearchDir, typename Criteria>
void basic_mcraptor<SearchDir, Criteria>::next_start_time() {
  // With range reuse the accumulating bag persists across start times so a
  // later-departing start can prune an earlier one; without it the bag is
  // wiped every start to stay lean (pong).
  if (!range_reuse_) {
    state_.bag_.clear();
    state_.breadcrumbs_.clear();
  }
  seeds_.clear();
  std::fill(begin(state_.prev_station_mark_.blocks_),
            end(state_.prev_station_mark_.blocks_), 0U);
  std::fill(begin(state_.station_mark_.blocks_),
            end(state_.station_mark_.blocks_), 0U);
  std::fill(begin(state_.route_mark_.blocks_),  //
            end(state_.route_mark_.blocks_), 0U);
}

namespace {

// visit the frontier members of a location's bag produced in one round of
// the current start (dep). Under range reuse the bag also holds labels of
// other departures (kept only for cross-departure pruning); those are
// skipped here so boarding / footpath expansion / dest collection see only
// this execute's labels. Without reuse every label has dep == cur_dep.
template <typename Criteria, typename Fn>
void for_each_label_in_round(
    typename basic_mcraptor_state<Criteria>::bag_layer const& layer,
    std::uint32_t const l,
    std::uint8_t const round,
    delta_t const dep,
    Fn&& fn) {
  for (auto const& e : layer.bags_[l]) {
    if (e.round_ == round && e.dep_ == dep) {
      fn(e.crit_, e.breadcrumb_);
    }
  }
}

// unified pareto insert into the single accumulating per-stop bag. The
// transfer round is an implicit pareto dimension, which lets one bag hold
// every round at once; the departure (dep) makes cross-start reuse a third
// dimension, so one persistent bag also subsumes the rRAPTOR frontier
// (OTP's McStopArrivals across departure iterations). A candidate produced
// in `round` at departure `dep` is
//   - REJECTED only by a label of a lower-or-equal round (round_ <= round)
//     - a by-route candidate only by another by-route label (footpaths are
//       not transitive: a transit arrival must survive to relax its own
//       footpaths even if a footpath arrival dominates it here);
//     - a by-transfer candidate by any label.
//   - EVICTS only labels of a higher-or-equal round (round_ >= round),
//     never a fewer-transfer label it happens to dominate on the other
//     criteria (that label is still a boarding source / result with fewer
//     transfers):
//     - a by-route candidate evicts any dominated label (a dominated
//       by-transfer label is terminal, so nothing is lost);
//     - a by-transfer candidate evicts only dominated by-transfer labels.
// Same-departure labels compare with the ordinary criteria dominance;
// labels of a different (necessarily later, processed-first) departure use
// the departure-aware reuse dominance, so a later start prunes an earlier
// one only when it dominates the final journey (dep, arr, transfers, cost).
// No pareto-optimal label is ever dropped. Returns true iff inserted.
template <direction SearchDir, typename Criteria>
bool bag_insert(typename basic_mcraptor_state<Criteria>::bag_layer& layer,
                std::uint32_t const l,
                Criteria const& crit,
                std::uint32_t const flagged_breadcrumb,
                std::uint8_t const round,
                delta_t const dep,
                bool const by_route) {
  using state_t = basic_mcraptor_state<Criteria>;
  auto& bag = layer.bags_[l];
  if (bag.empty()) {
    layer.touched_.set(l);
  }
  for (auto const& e : bag) {
    if (e.round_ <= round &&
        (!by_route || (e.breadcrumb_ & state_t::kByRoute) != 0U) &&
        (e.dep_ == dep
             ? e.crit_.template dominates<SearchDir>(crit)
             : e.crit_.template reuse_dominates<SearchDir>(crit, e.dep_, dep))) {
      return false;
    }
  }
  // boarding watermark: mark where this round's block begins. The first
  // label of a new round is appended after the (frozen) earlier-round
  // labels, so its position is the block start; eviction below only touches
  // round >= round, which cannot exist yet for a first-of-round insert.
  if (layer.blk_round_[l] != round) {
    layer.blk_round_[l] = round;
    layer.blk_start_[l] = static_cast<std::uint32_t>(bag.size());
  }
  auto removed = std::size_t{0U};
  for (auto i = std::size_t{0U}; i != bag.size(); ++i) {
    if (bag[i].round_ >= round &&
        (by_route || (bag[i].breadcrumb_ & state_t::kByRoute) == 0U) &&
        (bag[i].dep_ == dep
             ? crit.template dominates<SearchDir>(bag[i].crit_)
             : crit.template reuse_dominates<SearchDir>(bag[i].crit_, dep,
                                                        bag[i].dep_))) {
      ++removed;
      continue;
    }
    bag[i - removed] = bag[i];
  }
  bag.resize(bag.size() - removed + 1U);
  bag.back() = {crit, flagged_breadcrumb, round, dep};
  return true;
}

}  // namespace

// dest-aware same-station transfer buffer (0 at a non-intermodal
// destination, like raptor.h's update_transfers)
template <direction SearchDir, typename Criteria>
delta_t basic_mcraptor<SearchDir, Criteria>::transfer_buffer(
    std::uint64_t const l) const {
  return static_cast<delta_t>(
      (!is_intermodal_dest() && is_dest_[l])
          ? 0
          : dir(adjusted_transfer_time(
                transfer_time_settings_,
                tt_.locations_.transfer_time_[location_idx_t{l}].count())));
}

template <direction SearchDir, typename Criteria>
bool basic_mcraptor<SearchDir, Criteria>::merge_round(
    std::uint32_t const l,
    Criteria const& crit,
    typename state_t::breadcrumb const& bc,
    std::uint8_t const round) {
  if (bag_insert<SearchDir, Criteria>(
          state_.bag_, l, crit,
          static_cast<std::uint32_t>(state_.breadcrumbs_.size()), round,
          cur_dep_, /*by_route=*/false)) {
    state_.breadcrumbs_.push_back(bc);
    return true;
  }
  return false;
}

template <direction SearchDir, typename Criteria>
void basic_mcraptor<SearchDir, Criteria>::add_start(location_idx_t const l,
                                                    unixtime_t const t) {
  auto const i = to_idx(l);
  // record the raw seed; execute() inserts it once the query start time is
  // known (needed for the ingress walking cost and the departure tag).
  seeds_.emplace_back(static_cast<std::uint32_t>(i), unix_to_delta(base(), t));
  state_.station_mark_.set(i, true);
}

template <direction SearchDir, typename Criteria>
void basic_mcraptor<SearchDir, Criteria>::execute(
    unixtime_t const start_time,
    std::uint8_t const max_transfers,
    unixtime_t const worst_time_at_dest,
    profile_idx_t const prf_idx,
    pareto_set<journey>& results) {
  auto const end_k = std::min(max_transfers, kMaxTransfers) + 2U;

  worst_at_dest_ =
      get_best(unix_to_delta(base(), worst_time_at_dest), worst_at_dest_);

  // seed the round-0 labels now that the query start time is known: assign
  // the ingress walking duration (start offsets/footpaths are applied by
  // get_starts, so the walking part is the seeded stop time minus the query
  // start) and tag the current departure. Inserted straight into the bag -
  // which may already hold other departures' entries under range reuse.
  auto const d_start = unix_to_delta(base(), start_time);
  cur_dep_ = d_start;
  for (auto const& [l, arr] : seeds_) {
    bag_insert<SearchDir, Criteria>(
        state_.bag_, l,
        Criteria::at_start(arr,
                           static_cast<std::uint16_t>(dir(arr - d_start))),
        state_t::kNoBreadcrumb, /*round=*/std::uint8_t{0U}, cur_dep_,
        /*by_route=*/false);
  }
  seeds_.clear();

  for (auto k = 1U; k != end_k; ++k) {
    auto any_marked = false;
    state_.station_mark_.for_each_set_bit([&](std::uint64_t const i) {
      // snapshot each boarding stop's round-(k-1) block now, before any
      // round-k insert moves the boundary. station_mark_ here = stops touched
      // in round k-1, whose last (and only current) block is round k-1. Only
      // the single-departure path uses this: with reuse, cross-departure
      // eviction can shift the block, so boarding keeps the filtered scan.
      if (!range_reuse_ && use_watermark_) {
        state_.board_[i] = {state_.bag_.blk_start_[i],
                            static_cast<std::uint32_t>(state_.bag_.bags_[i].size())};
      }
      for (auto const& r : tt_.location_routes_[location_idx_t{i}]) {
        any_marked = true;
        state_.route_mark_.set(to_idx(r), true);
      }
    });

    if (!any_marked) {
      break;
    }

    // the round layers are one accumulating bag now (entries carry their
    // round); only the station marks alternate per round.
    std::swap(state_.prev_station_mark_, state_.station_mark_);
    std::fill(begin(state_.station_mark_.blocks_),
              end(state_.station_mark_.blocks_), 0U);

    any_marked = loop_routes(k);

    std::fill(begin(state_.route_mark_.blocks_),
              end(state_.route_mark_.blocks_), 0U);
    if (!any_marked) {
      break;
    }

    // footpath expansion iterates the marks the route scan just set;
    // fp targets are added to the same mark set for the next round's
    // route collection but are flag-skipped by the expansion itself. One
    // fused pass does intermodal egress + footpaths.
    update_footpaths(k, prf_idx);

    collect_dest_journeys(k, start_time, results);
  }
}

template <direction SearchDir, typename Criteria>
bool basic_mcraptor<SearchDir, Criteria>::loop_routes(unsigned const k) {
  auto const clasz_filter = allowed_claszes_ != all_clasz_allowed();
  auto any_marked = false;
  state_.route_mark_.for_each_set_bit([&](auto const r_idx) {
    auto const r = route_idx_t{r_idx};
    if (clasz_filter && !is_allowed(allowed_claszes_, tt_.route_clasz_[r])) {
      return;
    }
    ++stats_.n_routes_visited_;
    any_marked |= update_route(k, r);
  });
  return any_marked;
}

template <direction SearchDir, typename Criteria>
bool basic_mcraptor<SearchDir, Criteria>::update_route(unsigned const k,
                                                       route_idx_t const r) {
  auto const stop_seq = tt_.route_location_seq_[r];
  auto const n = stop_seq.size();
  auto any_marked = false;

  route_bag_.clear();
  route_bag_dep_.clear();

  auto const arr_ev = kFwd ? event_type::kArr : event_type::kDep;
  auto const dep_ev = kFwd ? event_type::kDep : event_type::kArr;

  for (auto i = 0U; i != n; ++i) {
    auto const stop_idx = static_cast<stop_idx_t>(kFwd ? i : n - i - 1U);
    auto const stp = stop{stop_seq[stop_idx]};
    auto const l_idx = cista::to_idx(stp.location_idx());
    auto const is_last = i == n - 1U;

    if (i != 0U && !route_bag_.empty() &&
        stp.can_finish<SearchDir>(is_wheelchair_)) {
      for (auto const& rl : route_bag_) {
        auto const by_transport = time_at_stop(r, rl.t_, stop_idx, arr_ev);
        if (!is_better(by_transport, worst_at_dest_) ||
            lb_[l_idx] == kUnreachable ||
            !is_better(by_transport + dir(lb_[l_idx]), worst_at_dest_)) {
          ++stats_.route_update_prevented_by_lower_bound_;
          continue;
        }
        auto const ride_duration =
            static_cast<std::uint16_t>(dir(by_transport - rl.board_dep_));
        auto const ride_crit =
            Criteria::from_ride(by_transport, ride_duration, rl.carried_);
        // destination pareto pruning: optimistic projection to the
        // destination checked against the (round, criteria) frontier
        if (dest_dominates(k, ride_crit.projected_to(
                                  clamp(by_transport + dir(lb_[l_idx]))))) {
          ++stats_.route_update_prevented_by_lower_bound_;
          continue;
        }
        // the same-station transfer buffer is added immediately; the label
        // goes straight into the accumulating bag flagged kByRoute so its
        // footpaths are relaxed after the route scan. bag_insert is the
        // by-transit gate (rejects a route arrival dominated by an earlier-
        // or-equal-round by-route label; post-buffer dominance == pre-buffer,
        // the buffer being per-stop constant) and, under range reuse, the
        // cross-departure gate. Footpath arrivals never gate it (footpaths
        // are not transitive).
        auto const post_crit = ride_crit.with_transfer(transfer_buffer(l_idx));
        auto const breadcrumb_idx =
            static_cast<std::uint32_t>(state_.breadcrumbs_.size());
        if (!bag_insert<SearchDir, Criteria>(
                state_.bag_, static_cast<std::uint32_t>(l_idx), post_crit,
                breadcrumb_idx | state_t::kByRoute, static_cast<std::uint8_t>(k),
                cur_dep_, /*by_route=*/true)) {
          continue;
        }
        // traffic day is not stored: reconstruction recovers it from arr_
        state_.breadcrumbs_.push_back(
            {.payload_ = make_transport_payload(to_idx(rl.t_.t_idx_), rl.board_,
                                                stop_idx),
             .parent_ = rl.parent_,
             .arr_ = post_crit.arr_});
        ++stats_.n_earliest_arrival_updated_by_route_;
        state_.station_mark_.set(l_idx, true);
        any_marked = true;

        if (is_dest_[l_idx]) {
          // transfer_buffer() is 0 at a station destination, so the ride
          // criteria are the journey's arrival criteria
          dest_bag_add(k, ride_crit);
        }
      }
    }

    if (is_last || !stp.can_start<SearchDir>(is_wheelchair_) ||
        !state_.prev_station_mark_[l_idx]) {
      continue;
    }

    if (lb_[l_idx] == kUnreachable) {
      break;
    }

    if (state_.bag_.empty(static_cast<std::uint32_t>(l_idx))) {
      continue;
    }

    // cache the route labels' departure times at this stop
    // (used by the coverage check and the bag merge below)
    route_bag_dep_.clear();
    for (auto const& rl : route_bag_) {
      route_bag_dep_.push_back(time_at_stop(r, rl.t_, stop_idx, dep_ev));
    }

    // board from the previous round's arrivals at this stop (round k-1) of
    // the current start
    auto const board_from = [&](Criteria const& pe_crit,
                                std::uint32_t const pe_breadcrumb) {
          auto const pe_arr = pe_crit.arr_;
          auto const pe_carried = pe_crit.carry();
          // destination pareto pruning before the boarding search: even
          // the optimistic completion of this breadcrumb is dominated
          if (dest_dominates(
                  k, pe_crit.projected_to(clamp(pe_arr + dir(lb_[l_idx]))))) {
            return;
          }
          // Skip the earliest-transport lookup if a route breadcrumb already
          // boards a trip departing before this label's arrival (with
          // carried criteria dominating): the lookup result would be
          // dominated.
          for (auto j = std::size_t{0U}; j != route_bag_.size(); ++j) {
            if (route_bag_[j].carried_.template dominates<SearchDir>(
                    pe_carried) &&
                is_better(route_bag_dep_[j], pe_arr)) {
              return;
            }
          }

          auto const [day, mam] = split(pe_arr);
          auto const et =
              get_earliest_transport(r, stop_idx, day, mam, stp.location_idx());
          if (!et.is_valid()) {
            return;
          }

          // merge into the route bag: pareto over
          // (total trip order, carried criteria)
          auto const key_new = trip_order_key(r, et);
          for (auto& rl : route_bag_) {
            if (rl.key_ == key_new && rl.carried_ == pe_carried) {
              // same trip, same carried criteria: prefer boarding closest
              // to the exit (matches reconstruct.cc's label point)
              rl.board_ = stop_idx;
              rl.board_dep_ = time_at_stop(r, rl.t_, stop_idx, dep_ev);
              rl.parent_ = pe_breadcrumb & state_t::kBreadcrumbMask;
              return;
            }
            if (!is_earlier_trip(key_new, rl.key_) &&
                rl.carried_.template dominates<SearchDir>(pe_carried)) {
              return;
            }
          }
          auto w = 0U;
          for (auto j = 0U; j != route_bag_.size(); ++j) {
            if (!(is_earlier_trip(key_new, route_bag_[j].key_) ||
                  (key_new == route_bag_[j].key_)) ||
                !pe_carried.template dominates<SearchDir>(
                    route_bag_[j].carried_)) {
              route_bag_[w] = route_bag_[j];
              route_bag_dep_[w] = route_bag_dep_[j];
              ++w;
            }
          }
          route_bag_.resize(w);
          route_bag_dep_.resize(w);
          auto const dep_new = time_at_stop(r, et, stop_idx, dep_ev);
          route_bag_.push_back({et, key_new, dep_new, stop_idx,
                                pe_breadcrumb & state_t::kBreadcrumbMask, pe_carried});
          route_bag_dep_.push_back(dep_new);
    };
    if (range_reuse_ || !use_watermark_) {
      // persistent bag (reuse) or watermark disabled: filter round + departure
      for_each_label_in_round<Criteria>(
          state_.bag_, static_cast<std::uint32_t>(l_idx),
          static_cast<std::uint8_t>(k - 1U), cur_dep_, board_from);
    } else {
      // watermark: iterate exactly this start's round-(k-1) block, no filter
      auto const& bag = state_.bag_.bags_[static_cast<std::uint32_t>(l_idx)];
      auto const range = state_.board_[l_idx];
      for (auto idx = range.lo_; idx != range.hi_; ++idx) {
        board_from(bag[idx].crit_, bag[idx].breadcrumb_);
      }
    }
  }
  return any_marked;
}

// (traffic day << 16 | transport offset in route): lexicographic order =
// total trip order within a route (trips in a route do not overtake)
template <direction SearchDir, typename Criteria>
std::uint32_t basic_mcraptor<SearchDir, Criteria>::trip_order_key(
    route_idx_t const r, transport const t) const {
  auto const t_offset = static_cast<std::uint32_t>(
      to_idx(t.t_idx_) - to_idx(tt_.route_transport_ranges_[r].from_));
  assert(t_offset < (1U << 16U));
  return (static_cast<std::uint32_t>(as_int(t.day_)) << 16U) | t_offset;
}

// One pass over the marked stops does same-round intermodal egress AND
// footpath relaxation (gouda raptor_impl update_transfers_and_footpaths; the
// same-station transfer is already folded into update_route on our side).
// Fusing collects each stop's by-route arrivals ONCE instead of once per
// former loop.
template <direction SearchDir, typename Criteria>
void basic_mcraptor<SearchDir, Criteria>::update_footpaths(
    unsigned const k, profile_idx_t const prf_idx) {
  auto const intermodal = is_intermodal_dest();
  state_.station_mark_.for_each_set_bit([&](std::uint64_t const i) {
    if (state_.bag_.empty(static_cast<std::uint32_t>(i))) {
      return;
    }
    auto const l = location_idx_t{i};
    auto const& fps = kFwd ? tt_.locations_.footpaths_out_[prf_idx][l]
                           : tt_.locations_.footpaths_in_[prf_idx][l];
    auto const egress_ok =
        intermodal && end_reachable_.test(i) &&
        dist_to_end_[i] != std::numeric_limits<std::uint16_t>::max();
    if (fps.empty() && !egress_ok) {
      return;
    }

    // copy this round's by-route entries once (expansion inserts into the
    // same bag and marks stops in the bitvec being iterated - inserted
    // entries are by-transfer and would be flag-skipped anyway); the
    // pre-transfer criteria are recovered so footpaths/egress relax from the
    // transit arrival itself.
    auto const buf = transfer_buffer(i);
    fp_labels_.clear();
    for_each_label_in_round<Criteria>(
        state_.bag_, static_cast<std::uint32_t>(i),
        static_cast<std::uint8_t>(k), cur_dep_,
        [&](Criteria const& e_crit, std::uint32_t const e_breadcrumb) {
          if ((e_breadcrumb & state_t::kByRoute) != 0U) {
            fp_labels_.push_back({e_crit.with_transfer(-buf),
                                  e_breadcrumb & state_t::kBreadcrumbMask,
                                  std::uint8_t{0U}, cur_dep_});
          }
        });
    if (fp_labels_.empty()) {
      return;
    }

    // intermodal egress (former update_intermodal_footpaths)
    if (egress_ok) {
      for (auto const& te : fp_labels_) {
        auto const end_crit =
            te.crit_.with_walk(dir(dist_to_end_[i]), dist_to_end_[i]);
        // window bound: without this, pong's reverse searches write
        // journeys departing beyond the ping's initial start time into the
        // results and poison the destination frontier (gouda fix)
        if (!is_better(end_crit.arr_, worst_at_dest_)) {
          continue;
        }
        auto bc = state_.breadcrumbs_[te.breadcrumb_];
        bc.arr_ = end_crit.arr_;
        if (!merge_round(to_idx(kIntermodalTarget), end_crit, bc,
                         static_cast<std::uint8_t>(k))) {
          continue;
        }
        dest_bag_add(k, end_crit);
      }
    }

    // footpaths (former update_footpaths)
    for (auto const& fp : fps) {
      ++stats_.n_footpaths_visited_;
      auto const target = to_idx(fp.target());
      if (target == i) {
        continue;
      }
      auto const fp_duration = adjusted_transfer_time(transfer_time_settings_,
                                                      fp.duration().count());
      for (auto const& te : fp_labels_) {
        auto const fp_crit = te.crit_.with_walk(
            dir(fp_duration), static_cast<std::uint16_t>(fp_duration));
        auto const fp_target_time = fp_crit.arr_;
        if (!is_better(fp_target_time, worst_at_dest_)) {
          continue;
        }
        auto const lower_bound = lb_[target];
        if (lower_bound == kUnreachable ||
            !is_better(fp_target_time + dir(lower_bound), worst_at_dest_)) {
          ++stats_.fp_update_prevented_by_lower_bound_;
          continue;
        }
        if (dest_dominates(k, fp_crit.projected_to(
                                  clamp(fp_target_time + dir(lower_bound))))) {
          ++stats_.fp_update_prevented_by_lower_bound_;
          continue;
        }
        // the bag insert is the cross-round + this-round dominance gate (and
        // the cross-departure reuse gate); the footpath breadcrumb keeps the
        // transit ride's payload + parent (the footpath is derived at
        // reconstruction from alight vs. target), only the arrival moves.
        auto bc = state_.breadcrumbs_[te.breadcrumb_];
        bc.arr_ = fp_crit.arr_;
        if (!merge_round(static_cast<std::uint32_t>(target), fp_crit, bc,
                         static_cast<std::uint8_t>(k))) {
          continue;
        }
        ++stats_.n_earliest_arrival_updated_by_footpath_;
        state_.station_mark_.set(target, true);
        if (is_dest_[target]) {
          dest_bag_add(k, fp_crit);
        }
      }
    }
  });
}

template <direction SearchDir, typename Criteria>
transport basic_mcraptor<SearchDir, Criteria>::get_earliest_transport(
    route_idx_t const r,
    stop_idx_t const stop_idx,
    day_idx_t const day_at_stop,
    minutes_after_midnight_t const mam_at_stop,
    location_idx_t const l) {
  ++stats_.n_earliest_trip_calls_;

  auto const event_times = tt_.event_times_at_stop(
      r, stop_idx, kFwd ? event_type::kDep : event_type::kArr);

  auto const seek_first_day = [&]() {
    return linear_lb(get_begin_it(event_times), get_end_it(event_times),
                     mam_at_stop,
                     [&](delta const a, minutes_after_midnight_t const b) {
                       return is_better(a.mam(), b.count());
                     });
  };

  constexpr auto const kNDaysToIterate =
      kMaxTravelTime / std::chrono::days{1} + 1U;
  for (auto i = day_idx_t::value_t{0U}; i != kNDaysToIterate; ++i) {
    auto const day = kFwd ? day_at_stop + i : day_at_stop - i;

    if (!tt_.is_route_active(r, day)) {
      continue;
    }

    auto const ev_time_range =
        it_range{i == 0U ? seek_first_day() : get_begin_it(event_times),
                 get_end_it(event_times)};
    if (ev_time_range.empty()) {
      continue;
    }
    for (auto it = begin(ev_time_range); it != end(ev_time_range); ++it) {
      auto const t_offset = static_cast<std::size_t>(&*it - event_times.data());
      auto const ev = *it;
      auto const ev_mam = ev.mam();

      if (is_better_or_eq(worst_at_dest_,
                          to_delta(day, ev_mam) + dir(lb_[to_idx(l)]))) {
        return {transport_idx_t::invalid(), day_idx_t::invalid()};
      }

      auto const t = tt_.route_transport_ranges_[r][t_offset];
      if (i == 0U && !is_better_or_eq(mam_at_stop.count(), ev_mam)) {
        continue;
      }

      auto const ev_day_offset = ev.days();
      auto const start_day =
          static_cast<day_idx_t>(as_int(day) - ev_day_offset);
      if (!tt_.is_transport_active(t, start_day)) {
        continue;
      }

      return {t, start_day};
    }
  }
  return {};
}

template <direction SearchDir, typename Criteria>
void basic_mcraptor<SearchDir, Criteria>::collect_dest_journeys(
    unsigned const k,
    unixtime_t const start_time,
    pareto_set<journey>& results) {
  is_dest_.for_each_set_bit([&](std::uint64_t const i) {
    for_each_label_in_round<Criteria>(
        state_.bag_, static_cast<std::uint32_t>(i),
        static_cast<std::uint8_t>(k), cur_dep_,
        [&](Criteria const& e_crit, std::uint32_t const e_breadcrumb) {
          // check dominance before materializing the legs: most candidates
          // are rejected and materialization allocates (journey legs)
          auto probe = journey{};
          probe.start_time_ = start_time;
          probe.dest_time_ = to_unix(e_crit.arr_);
          probe.dest_ = location_idx_t{i};
          probe.transfers_ = static_cast<std::uint8_t>(k - 1U);
          e_crit.apply_to(probe);
          if (results.is_dominated(probe)) {
            return;
          }
          results.add(materialize(location_idx_t{i}, k, e_crit,
                                  e_breadcrumb & state_t::kBreadcrumbMask, start_time));
        });
  });
}

template <direction SearchDir, typename Criteria>
journey basic_mcraptor<SearchDir, Criteria>::materialize(
    location_idx_t const dest,
    unsigned const k,
    Criteria const& crit,
    std::uint32_t const breadcrumb_idx,
    unixtime_t const start_time) {
  auto j = journey{};
  j.start_time_ = start_time;
  j.dest_time_ = to_unix(crit.arr_);
  j.dest_ = dest;
  j.transfers_ = static_cast<std::uint8_t>(k - 1U);
  crit.apply_to(j);

  // Chase breadcrumbs from the destination to the start. Legs are
  // collected in search order (reverse chronological for forward search).
  auto& legs = rec_legs_;
  legs.clear();

  auto const arr_ev = kFwd ? event_type::kArr : event_type::kDep;
  auto const dep_ev = kFwd ? event_type::kDep : event_type::kArr;

  auto cur_l = dest;
  auto li = breadcrumb_idx;
  while (li != state_t::kNoBreadcrumb) {
    auto const& bc = state_.breadcrumbs_[li];
    auto const cur_arr = bc.arr_;  // arrival at cur_l (== crit.arr_ first hop)
    auto const t_idx = transport_idx_t{bc_transport(bc.payload_)};
    auto const board = static_cast<stop_idx_t>(bc_board(bc.payload_));
    auto const alight = static_cast<stop_idx_t>(bc_alight(bc.payload_));
    auto const r = tt_.transport_route_[t_idx];

    // recover the traffic day: the ride's stored arrival is cur_arr, so find
    // the day whose train arrives at alight no later than cur_arr. A single
    // footpath/transfer crosses midnight at most once, so the day is
    // arr_day - event_day_offset - {0,1} (gouda raptor_impl reconstruction).
    auto const event_day_offset = tt_.event_mam(r, t_idx, alight, arr_ev).count() / 1440;
    auto const arr_day = as_int(split(cur_arr).first);
    auto day = day_idx_t::invalid();
    auto train_arr = kInvalid;
    for (auto off = 0; off != 2; ++off) {
      auto const cand = arr_day - event_day_offset - (kFwd ? off : -off);
      if (cand < 0) {
        continue;
      }
      auto const cand_day = day_idx_t{static_cast<day_idx_t::value_t>(cand)};
      if (!tt_.is_transport_active(t_idx, cand_day)) {
        continue;
      }
      auto const ev = time_at_stop(r, transport{t_idx, cand_day}, alight, arr_ev);
      if (is_better_or_eq(ev, cur_arr)) {
        day = cand_day;
        train_arr = ev;
        break;
      }
    }
    utl::verify(day != day_idx_t::invalid(),
                "mcraptor reconstruct: traffic day recovery failed");

    auto const tr = transport{t_idx, day};
    auto const dep_at_board = time_at_stop(r, tr, board, dep_ev);
    auto const stop_seq = tt_.route_location_seq_[r];
    auto const board_loc = stop{stop_seq[board]}.location_idx();
    auto const alight_loc = stop{stop_seq[alight]}.location_idx();

    if (is_intermodal_dest() && cur_l == kIntermodalTarget) {
      // no footpath leg: the last mile is added by reconstruct(); the
      // journey's terminal is the ride's alighting stop
      j.dest_ = alight_loc;
    } else if (!legs.empty() || alight_loc != cur_l || train_arr != cur_arr) {
      // Footpaths are always emitted (even zero minute reflexive
      // transfers). Journey structure: [WALK]? TRANSIT [TRANSIT WALK]*
      legs.push_back(
          {.is_footpath_ = true,
           .from_ = alight_loc,
           .to_ = cur_l,
           .dep_ = train_arr,
           .arr_ = cur_arr,
           .t_ = transport_idx_t::invalid(),
           .day_ = day_idx_t::invalid(),
           .enter_ = 0U,
           .exit_ = 0U,
           .fp_duration_ = static_cast<std::uint16_t>(
               kFwd ? (cur_arr - train_arr) : (train_arr - cur_arr))});
    }

    legs.push_back({.is_footpath_ = false,
                    .from_ = board_loc,
                    .to_ = alight_loc,
                    .dep_ = dep_at_board,
                    .arr_ = train_arr,
                    .t_ = t_idx,
                    .day_ = day,
                    .enter_ = board,
                    .exit_ = alight,
                    .fp_duration_ = 0U});

    cur_l = board_loc;
    li = bc.parent_;
  }

  // materialize journey legs in chronological order
  for (auto z = 0U; z != legs.size(); ++z) {
    auto const& gl = kFwd ? legs[legs.size() - 1U - z] : legs[z];
    auto const dep = to_unix(gl.dep_);
    auto const arr = to_unix(gl.arr_);
    if (gl.is_footpath_) {
      j.legs_.emplace_back(journey::leg{
          SearchDir, gl.from_, gl.to_, dep, arr,
          footpath{gl.to_,
                   duration_t{static_cast<duration_t::rep>(gl.fp_duration_)}}});
    } else {
      auto const route = tt_.transport_route_[gl.t_];
      auto const route_len =
          static_cast<stop_idx_t>(tt_.route_location_seq_[route].size());
      auto const run = rt::run{
          .t_ = transport{gl.t_, gl.day_},
          .stop_range_ = interval<stop_idx_t>{stop_idx_t{0U}, route_len}};
      j.legs_.emplace_back(
          journey::leg{SearchDir, gl.from_, gl.to_, dep, arr,
                       journey::run_enter_exit{run, gl.enter_, gl.exit_}});
    }
  }

  // Backward search requires to re-anchor footpath durations to the
  // arrival of the previous trip instead of the departure of the next
  // trip. No-op for forward search.
  for (auto z = std::size_t{1U}; z < j.legs_.size(); ++z) {
    if (std::holds_alternative<footpath>(j.legs_[z].uses_)) {
      auto const dur = std::get<footpath>(j.legs_[z].uses_).duration();
      j.legs_[z].dep_time_ = j.legs_[z - 1U].arr_time_;
      j.legs_[z].arr_time_ = j.legs_[z].dep_time_ + dur;
    }
  }

  return j;
}

// First/last mile mumo offset and start footpath legs are added here,
// where the query offsets live.
template <direction SearchDir, typename Criteria>
void basic_mcraptor<SearchDir, Criteria>::reconstruct(query const& q,
                                                      journey& j) {
  utl::verify(!j.legs_.empty(), "mcraptor reconstruct: journey without legs");

  // The legs are chronological (materialization reverses forward order).
  constexpr auto const is_fwd = SearchDir == direction::kForward;

  // Front-side mumo leg: special_station -> first transit stop.
  auto const from = j.legs_.front().from_;
  auto const dep_time = j.legs_.front().dep_time_;
  auto const front_match_mode =
      is_fwd ? q.start_match_mode_ : q.dest_match_mode_;
  if (front_match_mode == location_match_mode::kIntermodal) {
    auto const& offsets = is_fwd ? q.start_ : q.destination_;
    auto const special = get_special_station(is_fwd ? special_station::kStart
                                                    : special_station::kEnd);
    auto const o = utl::find_if(offsets, [&](offset const& x) {
      return matches(tt_, front_match_mode, x.target(), from) &&
             (is_fwd
                  // fwd: query start, check feasibility (allows ontrip start)
                  ? dep_time - x.duration() >= j.start_time_
                  // bwd: destination, anchored exactly at j.dest_time_
                  : dep_time - x.duration() == j.dest_time_);
    });
    utl::verify(o != end(offsets), "mcraptor reconstruct: no front offset");
    // start side: anchored at the journey start (like reconstruct.cc),
    // dest side: anchored at the first transit event
    auto const dep = is_fwd ? j.start_time_ : dep_time - o->duration();
    auto const arr = is_fwd ? j.start_time_ + o->duration() : dep_time;
    j.legs_.insert(begin(j.legs_), journey::leg{direction::kForward, special,
                                                from, dep, arr, *o});
  }

  // Back-side mumo leg: last transit stop -> special_station.
  auto const to = j.legs_.back().to_;
  auto const arr_time = j.legs_.back().arr_time_;
  auto const back_match_mode =
      is_fwd ? q.dest_match_mode_ : q.start_match_mode_;
  if (back_match_mode == location_match_mode::kIntermodal) {
    auto const& offsets = is_fwd ? q.destination_ : q.start_;
    auto const special = get_special_station(is_fwd ? special_station::kEnd
                                                    : special_station::kStart);
    auto const o = utl::find_if(offsets, [&](offset const& x) {
      return matches(tt_, back_match_mode, x.target(), to) &&
             (is_fwd
                  // fwd: destination, anchored exactly at j.dest_time_
                  ? arr_time + x.duration() == j.dest_time_
                  // bwd: query start, anchored by feasibility
                  : arr_time + x.duration() <= j.start_time_);
    });
    utl::verify(o != end(offsets), "mcraptor reconstruct: no back offset");
    // dest side: anchored at the last transit event, start side (bwd):
    // anchored at the journey start (like reconstruct.cc)
    auto const dep = is_fwd ? arr_time : j.start_time_ - o->duration();
    auto const arr = is_fwd ? arr_time + o->duration() : j.start_time_;
    j.legs_.push_back(
        journey::leg{direction::kForward, to, special, dep, arr, *o});
    j.dest_ = special;
  }

  // Reconstruct the start footpath that seeded round k=0 at the first stop.
  if (q.start_match_mode_ != location_match_mode::kIntermodal) {
    auto const is_journey_start = [&](location_idx_t const l) {
      return utl::any_of(q.start_, [&](offset const& o) {
        return matches(tt_, q.start_match_mode_, o.target(), l);
      });
    };
    auto const start_l = is_fwd ? j.legs_.front().from_ : j.legs_.back().to_;
    auto const start_t =
        is_fwd ? j.legs_.front().dep_time_ : j.legs_.back().arr_time_;
    auto const direct_start_ok =
        is_fwd ? j.start_time_ <= start_t : j.start_time_ >= start_t;
    if (!is_journey_start(start_l) || !direct_start_ok) {
      auto const fps = is_fwd
                           ? tt_.locations_.footpaths_in_[q.prf_idx_][start_l]
                           : tt_.locations_.footpaths_out_[q.prf_idx_][start_l];
      auto best = std::optional<footpath>{};
      for (auto const fp : fps) {
        if ((!best.has_value() || fp.duration() < best->duration()) &&
            is_journey_start(fp.target())) {
          best = fp;
        }
      }
      if (best.has_value()) {
        auto const dur = duration_t{adjusted_transfer_time(
            q.transfer_time_settings_, best->duration().count())};
        auto const fp_arr = j.start_time_ + (is_fwd ? dur : -dur);
        if (is_fwd ? fp_arr <= start_t : fp_arr >= start_t) {
          auto const lg = journey::leg{
              SearchDir,     best->target(), start_l,
              j.start_time_, fp_arr,         footpath{best->target(), dur}};
          if (is_fwd) {
            j.legs_.insert(begin(j.legs_), lg);
          } else {
            j.legs_.push_back(lg);
          }
        }
      }
    }
  }

  if constexpr (is_fwd) {
    optimize_footpaths(tt_, rtt_, q, j);
  } else {
    // The journey's legs are chronological, but q is in search direction.
    auto journey_q = q;
    journey_q.flip_dir();
    optimize_footpaths(tt_, rtt_, journey_q, j);
  }

  j.is_reconstructed_ = true;
}

template <direction SearchDir, typename Criteria>
delta_t basic_mcraptor<SearchDir, Criteria>::time_at_stop(
    route_idx_t const r,
    transport const t,
    stop_idx_t const stop_idx,
    event_type const ev_type) const {
  return to_delta(t.day_,
                  tt_.event_mam(r, t.t_idx_, stop_idx, ev_type).count());
}

template <direction SearchDir, typename Criteria>
delta_t basic_mcraptor<SearchDir, Criteria>::to_delta(
    day_idx_t const day, std::int16_t const mam) const {
  return clamp((as_int(day) - as_int(base_)) * 1440 + mam);
}

template <direction SearchDir, typename Criteria>
unixtime_t basic_mcraptor<SearchDir, Criteria>::to_unix(delta_t const t) const {
  return delta_to_unix(base(), t);
}

template <direction SearchDir, typename Criteria>
std::pair<day_idx_t, minutes_after_midnight_t>
basic_mcraptor<SearchDir, Criteria>::split(delta_t const x) const {
  return split_day_mam(base_, x);
}

template struct basic_mcraptor<direction::kForward, arr_criteria>;
template struct basic_mcraptor<direction::kBackward, arr_criteria>;
template struct basic_mcraptor<direction::kForward, arr_cost_criteria>;
template struct basic_mcraptor<direction::kBackward, arr_cost_criteria>;

}  // namespace nigiri::routing
