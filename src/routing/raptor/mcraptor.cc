#include "nigiri/routing/raptor/mcraptor.h"

#include <algorithm>
#include <optional>

#include "utl/erase_if.h"
#include "utl/helpers/algorithm.h"
#include "utl/verify.h"

#include "nigiri/common/it_range.h"
#include "nigiri/common/linear_lower_bound.h"
#include "nigiri/for_each_meta.h"
#include "nigiri/routing/raptor/reconstruct.h"
#include "nigiri/rt/rt_timetable.h"
#include "nigiri/special_stations.h"
#include "nigiri/td_footpath.h"
#include "nigiri/timetable.h"

namespace nigiri::routing {

constexpr auto const kIntermodalTarget =
    get_special_station(special_station::kEnd);

bool mcraptor_supported(query const& q, rt_timetable const*) {
  return !q.require_bike_transport_ && !q.require_car_transport_ &&
         !q.no_compulsory_reservation_ && q.via_stops_.empty();
}

template <direction SearchDir, typename Criteria, bool RangeReuse>
basic_mcraptor<SearchDir, Criteria, RangeReuse>::basic_mcraptor(
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
    bool const no_compulsory_reservation,
    transfer_time_settings const& tts,
    profile_idx_t const prf_idx)
    : tt_{tt},
      rtt_{rtt},
      has_rt_{rtt != nullptr},
      n_locations_{tt_.n_locations()},
      n_routes_{tt_.n_routes()},
      n_rt_transports_{has_rt_ ? rtt->n_rt_transports() : 0U},
      state_{state.resize(n_locations_, n_routes_, n_rt_transports_)},
      is_dest_{is_dest},
      dist_to_end_{dist_to_dest},
      td_dist_to_end_{td_dist_to_dest},
      lb_{lb},
      base_{base},
      allowed_claszes_{allowed_claszes},
      is_wheelchair_{is_wheelchair},
      transfer_time_settings_{tts},
      prf_idx_{prf_idx} {
  static_cast<void>(is_via);
  utl::verify(via_stops.empty(), "mcraptor: via stops not supported");
  utl::verify(!require_bike_transport && !require_car_transport &&
                  !no_compulsory_reservation,
              "mcraptor: bike/car transport, reservation filter not supported");
  // overlapping index ranges would silently mis-decode legs (see breadcrumb.h)
  utl::verify(
      bc_transport_space_fits(tt_.transport_route_.size(), n_rt_transports_),
      "mcraptor: transport index space exceeds the breadcrumb field");
  reset_arrivals();
  if (!dist_to_end_.empty()) {
    end_reachable_.resize(n_locations_);
    for (auto i = 0U; i != dist_to_end_.size(); ++i) {
      if (dist_to_end_[i] != kUnreachable) {
        end_reachable_.set(i, true);
      }
    }
    // a td egress location need not have a static offset
    for (auto const& [l, _] : td_dist_to_end_) {
      end_reachable_.set(to_idx(l), true);
    }
  }
}

template <direction SearchDir, typename Criteria, bool RangeReuse>
date::sys_days basic_mcraptor<SearchDir, Criteria, RangeReuse>::base() const {
  return tt_.internal_interval_days().from_ + as_int(base_) * date::days{1};
}

// Per-query reset. Under range reuse the bag survives next_start_time().
template <direction SearchDir, typename Criteria, bool RangeReuse>
void basic_mcraptor<SearchDir, Criteria, RangeReuse>::reset_arrivals() {
  worst_at_dest_ = kInvalid;
  dest_bag_.clear();
  state_.bag_.clear();
  state_.breadcrumbs_.clear();
  seeds_.clear();
}

template <direction SearchDir, typename Criteria, bool RangeReuse>
void basic_mcraptor<SearchDir, Criteria, RangeReuse>::next_start_time() {
  if constexpr (!RangeReuse) {
    state_.bag_.clear();
    state_.breadcrumbs_.clear();
  }
  seeds_.clear();
  state_.prev_station_mark_.zero_out();
  state_.station_mark_.zero_out();
  state_.route_mark_.zero_out();
  state_.rt_transport_mark_.zero_out();
}

namespace {

// Visits a bag's labels of one round and departure. Under range reuse the bag
// also holds other departures' labels, kept only for pruning.
template <typename Criteria, typename Fn>
void for_each_label_in_round(
    typename basic_mcraptor_state<Criteria>::bag_layer const& layer,
    std::uint32_t const l,
    std::uint8_t const round,
    delta_t const dep,
    Fn&& fn) {
  for (auto const& e : layer.span(l)) {
    if (e.round_ == round && e.dep_ == dep) {
      fn(e.crit_, e.breadcrumb_);
    }
  }
}

// Pareto insert into a stop's bag, which holds every round (and, under range
// reuse, every departure) at once. A candidate of `round` is
//   - rejected only by a label of a lower-or-equal round; a by-route candidate
//     only by another by-route one (footpaths are not transitive: a transit
//     arrival must survive to relax its own footpaths);
//   - evicting only labels of a higher-or-equal round, so a fewer-transfer
//     label is never lost; a by-route candidate evicts any dominated label, a
//     by-transfer one only dominated by-transfer labels (those are terminal).
// Same-departure labels use ordinary dominance, others (necessarily later,
// processed first) the departure-aware reuse dominance. Returns true iff
// inserted.
template <direction SearchDir, typename Criteria>
bool bag_insert(typename basic_mcraptor_state<Criteria>::bag_layer& layer,
                std::uint32_t const l,
                Criteria const& crit,
                std::uint32_t const flagged_breadcrumb,
                std::uint8_t const round,
                delta_t const dep,
                bool const by_route) {
  using state_t = basic_mcraptor_state<Criteria>;
  if (layer.empty(l)) {
    layer.touched_.set(l);
  }
  auto const bag = layer.span(l);
  for (auto const& e : bag) {
    if (e.round_ <= round &&
        (!by_route || (e.breadcrumb_ & state_t::kByRoute) != 0U) &&
        (e.dep_ == dep ? e.crit_.template dominates<SearchDir>(crit)
                       : e.crit_.template reuse_dominates<SearchDir>(
                             crit, e.dep_, dep))) {
      return false;
    }
  }
  auto removed = std::size_t{0U};
  for (auto i = std::size_t{0U}; i != bag.size(); ++i) {
    if (bag[i].round_ >= round &&
        (by_route || (bag[i].breadcrumb_ & state_t::kByRoute) == 0U) &&
        (bag[i].dep_ == dep ? crit.template dominates<SearchDir>(bag[i].crit_)
                            : crit.template reuse_dominates<SearchDir>(
                                  bag[i].crit_, dep, bag[i].dep_))) {
      ++removed;
      continue;
    }
    bag[i - removed] = bag[i];
  }
  layer.set_size(l, static_cast<std::uint32_t>(bag.size() - removed));
  layer.push_back(l, {crit, flagged_breadcrumb, round, dep});
  return true;
}

}  // namespace

// same-stop transfer buffer; 0 at a non-intermodal destination
template <direction SearchDir, typename Criteria, bool RangeReuse>
delta_t basic_mcraptor<SearchDir, Criteria, RangeReuse>::transfer_buffer(
    std::uint64_t const l) const {
  return static_cast<delta_t>(
      (!is_intermodal_dest() && is_dest_[static_cast<std::uint32_t>(l)])
          ? 0
          : dir(adjusted_transfer_time(
                transfer_time_settings_,
                tt_.locations_.transfer_time_[location_idx_t{l}].count())));
}

template <direction SearchDir, typename Criteria, bool RangeReuse>
bool basic_mcraptor<SearchDir, Criteria, RangeReuse>::merge_round(
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

template <direction SearchDir, typename Criteria, bool RangeReuse>
void basic_mcraptor<SearchDir, Criteria, RangeReuse>::add_start(
    location_idx_t const l, unixtime_t const t) {
  auto const i = to_idx(l);
  seeds_.emplace_back(static_cast<std::uint32_t>(i), unix_to_delta(base(), t));
  state_.station_mark_.set(i, true);
}

template <direction SearchDir, typename Criteria, bool RangeReuse>
void basic_mcraptor<SearchDir, Criteria, RangeReuse>::execute(
    unixtime_t const start_time,
    std::uint8_t const max_transfers,
    unixtime_t const worst_time_at_dest,
    pareto_set<journey>& results) {
  auto const end_k = std::min(max_transfers, kMaxTransfers) + 2U;
  cur_budget_ = end_k - 1U;  // trips allowed for this start time

  worst_at_dest_ =
      get_best(unix_to_delta(base(), worst_time_at_dest), worst_at_dest_);

  // Round-0 labels; the ingress walk is the seeded time minus the query start.
  auto const d_start = unix_to_delta(base(), start_time);
  cur_dep_ = d_start;
  for (auto const& [l, arr] : seeds_) {
    bag_insert<SearchDir, Criteria>(
        state_.bag_, l,
        Criteria::at_start(arr, static_cast<std::uint16_t>(dir(arr - d_start))),
        state_t::kNoBreadcrumb, /*round=*/std::uint8_t{0U}, cur_dep_,
        /*by_route=*/false);
  }
  seeds_.clear();

  for (auto k = 1U; k != end_k; ++k) {
    auto any_marked = false;
    state_.station_mark_.for_each_set_bit([&](std::uint64_t const i) {
      for (auto const& r : tt_.location_routes_[location_idx_t{i}]) {
        any_marked = true;
        state_.route_mark_.set(to_idx(r), true);
      }
      if (has_rt_) {
        for (auto const& rt_t :
             rtt_->location_rt_transports_[location_idx_t{i}]) {
          any_marked = true;
          state_.rt_transport_mark_.set(to_idx(rt_t), true);
        }
      }
    });

    if (!any_marked) {
      break;
    }

    std::swap(state_.prev_station_mark_, state_.station_mark_);
    state_.station_mark_.zero_out();

    // Both scans board from round k-1 and insert at round k, which never
    // evicts a round-(k-1) label, so their order does not matter.
    any_marked = loop_routes(k);
    if (has_rt_) {
      any_marked = loop_rt_transports(k) || any_marked;
    }

    state_.route_mark_.zero_out();
    state_.rt_transport_mark_.zero_out();
    if (!any_marked) {
      break;
    }

    // one fused pass for intermodal egress and footpaths
    update_footpaths(k);

    collect_dest_journeys(k, start_time, results);
  }
}

// Alights the boarded label `rl` at stop l_idx at `by_transport`: prunes, then
// inserts the arrival (plus transfer buffer) flagged by-route so its footpaths
// get relaxed. Returns true iff a label was inserted.
template <direction SearchDir, typename Criteria, bool RangeReuse>
template <typename Label, typename Sections>
bool basic_mcraptor<SearchDir, Criteria, RangeReuse>::alight(
    unsigned const k,
    std::uint32_t const l_idx,
    stop_idx_t const stop_idx,
    delta_t const by_transport,
    Label const& rl,
    Sections const& sections,
    clasz const fallback,
    std::uint32_t const transport_field) {
  auto const lb = effective_lb(l_idx);
  if (!is_better(by_transport, worst_at_dest_) || lb == kUnreachable ||
      !is_better(by_transport + dir(lb), worst_at_dest_) ||
      // the pruning search stores the post-transfer value, so the raw transit
      // arrival is the matching reference
      bound_prunes(k, l_idx, by_transport)) {
    ++stats_.route_update_prevented_by_lower_bound_;
    return false;
  }
  // clasz of the section alighted from: route_clasz_ is that of the first
  // section, which a run changing category midway never leaves
  auto const n_sec = static_cast<std::uint32_t>(sections.size());
  auto const sec = static_cast<std::uint32_t>(kFwd ? stop_idx - 1 : stop_idx);
  auto const ride = ride_attrs{
      .clasz_ = n_sec == 0U ? fallback : sections[std::min(sec, n_sec - 1U)]};
  auto const ride_crit = Criteria::from_ride(
      by_transport,
      static_cast<std::uint16_t>(dir(by_transport - rl.board_dep_)), ride,
      rl.carried_);
  if (dest_dominates(k,
                     ride_crit.projected_to(clamp(by_transport + dir(lb))))) {
    ++stats_.route_update_prevented_by_lower_bound_;
    return false;
  }
  // Post-buffer dominance equals pre-buffer (the buffer is per-stop constant).
  auto const post_crit = ride_crit.with_transfer(transfer_buffer(l_idx));
  auto const breadcrumb_idx =
      static_cast<std::uint32_t>(state_.breadcrumbs_.size());
  if (!bag_insert<SearchDir, Criteria>(
          state_.bag_, l_idx, post_crit, breadcrumb_idx | state_t::kByRoute,
          static_cast<std::uint8_t>(k), cur_dep_, /*by_route=*/true)) {
    return false;
  }
  state_.breadcrumbs_.push_back(
      {.payload_ = make_transport_payload(transport_field, rl.board_, stop_idx),
       .parent_ = rl.parent_,
       .arr_ = post_crit.arr_});
  ++stats_.n_earliest_arrival_updated_by_route_;
  state_.station_mark_.set(l_idx, true);
  // the buffer is 0 at a station destination: ride_crit is the arrival
  if (is_dest_[l_idx]) {
    dest_bag_add(k, ride_crit);
  }
  return true;
}

template <direction SearchDir, typename Criteria, bool RangeReuse>
bool basic_mcraptor<SearchDir, Criteria, RangeReuse>::loop_routes(
    unsigned const k) {
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

template <direction SearchDir, typename Criteria, bool RangeReuse>
bool basic_mcraptor<SearchDir, Criteria, RangeReuse>::update_route(
    unsigned const k, route_idx_t const r) {
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
        any_marked |=
            alight(k, l_idx, stop_idx, time_at_stop(r, rl.t_, stop_idx, arr_ev),
                   rl, tt_.route_section_clasz_[r], tt_.route_clasz_[r],
                   to_idx(rl.t_.t_idx_));
      }
    }

    if (is_last || !stp.can_start<SearchDir>(is_wheelchair_) ||
        !state_.prev_station_mark_[l_idx]) {
      continue;
    }

    if (effective_lb(l_idx) == kUnreachable) {
      break;
    }

    if (state_.bag_.empty(static_cast<std::uint32_t>(l_idx))) {
      continue;
    }

    route_bag_dep_.clear();
    for (auto const& rl : route_bag_) {
      route_bag_dep_.push_back(time_at_stop(r, rl.t_, stop_idx, dep_ev));
    }

    auto const board_from = [&](Criteria const& pe_crit,
                                std::uint32_t const pe_breadcrumb) {
      auto const pe_arr = pe_crit.arr_;
      auto const pe_carried = pe_crit.carry();
      // even the optimistic completion is dominated
      if (dest_dominates(k, pe_crit.projected_to(
                                clamp(pe_arr + dir(effective_lb(l_idx)))))) {
        return;
      }
      // skip the lookup if a boarded trip already departs before this arrival
      // with dominating carried criteria
      for (auto j = std::size_t{0U}; j != route_bag_.size(); ++j) {
        if (route_bag_[j].carried_.template dominates<SearchDir>(pe_carried) &&
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

      // pareto over (trip order, carried criteria)
      auto const key_new = trip_order_key(r, et);
      for (auto& rl : route_bag_) {
        if (rl.key_ == key_new && rl.carried_ == pe_carried) {
          // same trip and criteria: board closest to the exit
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
            !pe_carried.template dominates<SearchDir>(route_bag_[j].carried_)) {
          route_bag_[w] = route_bag_[j];
          route_bag_dep_[w] = route_bag_dep_[j];
          ++w;
        }
      }
      route_bag_.resize(w);
      route_bag_dep_.resize(w);
      auto const dep_new = time_at_stop(r, et, stop_idx, dep_ev);
      route_bag_.push_back({et, key_new, dep_new, stop_idx,
                            pe_breadcrumb & state_t::kBreadcrumbMask,
                            pe_carried});
      route_bag_dep_.push_back(dep_new);
    };
    for_each_label_in_round<Criteria>(
        state_.bag_, static_cast<std::uint32_t>(l_idx),
        static_cast<std::uint8_t>(k - 1U), cur_dep_, board_from);
  }
  return any_marked;
}

template <direction SearchDir, typename Criteria, bool RangeReuse>
bool basic_mcraptor<SearchDir, Criteria, RangeReuse>::loop_rt_transports(
    unsigned const k) {
  auto const clasz_filter = allowed_claszes_ != all_clasz_allowed();
  auto any_marked = false;
  state_.rt_transport_mark_.for_each_set_bit([&](auto const rt_t_idx) {
    auto const rt_t = rt_transport_idx_t{rt_t_idx};
    if (clasz_filter &&
        !is_allowed(allowed_claszes_,
                    rtt_->rt_transport_section_clasz_[rt_t][0])) {
      return;
    }
    ++stats_.n_routes_visited_;
    any_marked |= update_rt_transport(k, rt_t);
  });
  return any_marked;
}

// update_route for a realtime run: one trip with absolute event times, so no
// earliest-transport lookup, traffic day or trip order.
template <direction SearchDir, typename Criteria, bool RangeReuse>
bool basic_mcraptor<SearchDir, Criteria, RangeReuse>::update_rt_transport(
    unsigned const k, rt_transport_idx_t const rt_t) {
  auto const stop_seq = rtt_->rt_transport_location_seq_[rt_t];
  auto const n = stop_seq.size();
  auto any_marked = false;

  rt_bag_.clear();

  auto const arr_ev = kFwd ? event_type::kArr : event_type::kDep;
  auto const dep_ev = kFwd ? event_type::kDep : event_type::kArr;

  for (auto i = 0U; i != n; ++i) {
    auto const stop_idx = static_cast<stop_idx_t>(kFwd ? i : n - i - 1U);
    auto const stp = stop{stop_seq[stop_idx]};
    auto const l_idx = cista::to_idx(stp.location_idx());
    auto const is_last = i == n - 1U;

    if (i != 0U && !rt_bag_.empty() &&
        stp.can_finish<SearchDir>(is_wheelchair_)) {
      auto const by_transport = rt_time_at_stop(rt_t, stop_idx, arr_ev);
      for (auto const& rl : rt_bag_) {
        any_marked |=
            alight(k, l_idx, stop_idx, by_transport, rl,
                   rtt_->rt_transport_section_clasz_[rt_t], clasz::kOther,
                   encode_rt_bc_transport(to_idx(rt_t)));
      }
    }

    if (is_last || !stp.can_start<SearchDir>(is_wheelchair_) ||
        !state_.prev_station_mark_[l_idx]) {
      continue;
    }

    if (effective_lb(l_idx) == kUnreachable) {
      break;
    }

    if (state_.bag_.empty(static_cast<std::uint32_t>(l_idx))) {
      continue;
    }

    auto const dep = rt_time_at_stop(rt_t, stop_idx, dep_ev);

    // update_route's rules reduce to carried-criteria dominance here
    for_each_label_in_round<Criteria>(
        state_.bag_, static_cast<std::uint32_t>(l_idx),
        static_cast<std::uint8_t>(k - 1U), cur_dep_,
        [&](Criteria const& pe_crit, std::uint32_t const pe_breadcrumb) {
          if (!is_better_or_eq(pe_crit.arr_, dep)) {
            return;  // cannot make this run
          }
          if (dest_dominates(
                  k, pe_crit.projected_to(
                         clamp(pe_crit.arr_ + dir(effective_lb(l_idx)))))) {
            return;
          }
          auto const pe_carried = pe_crit.carry();
          for (auto& rl : rt_bag_) {
            if (rl.carried_ == pe_carried) {
              rl.board_ = stop_idx;
              rl.board_dep_ = dep;
              rl.parent_ = pe_breadcrumb & state_t::kBreadcrumbMask;
              return;
            }
            if (rl.carried_.template dominates<SearchDir>(pe_carried)) {
              return;
            }
          }
          auto w = 0U;
          for (auto j = 0U; j != rt_bag_.size(); ++j) {
            if (!pe_carried.template dominates<SearchDir>(
                    rt_bag_[j].carried_)) {
              rt_bag_[w] = rt_bag_[j];
              ++w;
            }
          }
          rt_bag_.resize(w);
          rt_bag_.push_back({dep, stop_idx,
                             pe_breadcrumb & state_t::kBreadcrumbMask,
                             pe_carried});
        });
  }
  return any_marked;
}

// (traffic day << 16 | transport offset in route); trips of a route do not
// overtake, so this orders them
template <direction SearchDir, typename Criteria, bool RangeReuse>
std::uint32_t basic_mcraptor<SearchDir, Criteria, RangeReuse>::trip_order_key(
    route_idx_t const r, transport const t) const {
  auto const t_offset = static_cast<std::uint32_t>(
      to_idx(t.t_idx_) - to_idx(tt_.route_transport_ranges_[r].from_));
  assert(t_offset < (1U << 16U));
  return (static_cast<std::uint32_t>(as_int(t.day_)) << 16U) | t_offset;
}

// Relaxes intermodal egress and footpaths from the marked stops' by-route
// arrivals of round k (same-stop transfers are folded into alight()).
template <direction SearchDir, typename Criteria, bool RangeReuse>
void basic_mcraptor<SearchDir, Criteria, RangeReuse>::update_footpaths(
    unsigned const k) {
  auto const intermodal = is_intermodal_dest();
  state_.station_mark_.for_each_set_bit([&](std::uint64_t const i) {
    if (state_.bag_.empty(static_cast<std::uint32_t>(i))) {
      return;
    }
    auto const l = location_idx_t{i};
    // td footpaths replace the static ones
    auto const use_td_fps = has_rt_ && prf_idx_ != 0U &&
                            (kFwd ? rtt_->has_td_footpaths_out_
                                  : rtt_->has_td_footpaths_in_)[prf_idx_]
                                .test(l);
    auto const& fps = kFwd ? tt_.locations_.footpaths_out_[prf_idx_][l]
                           : tt_.locations_.footpaths_in_[prf_idx_][l];
    auto const egress_ok =
        intermodal && end_reachable_.test(static_cast<std::uint32_t>(i)) &&
        dist_to_end_[i] != std::numeric_limits<std::uint16_t>::max();
    // empty for non-td queries: keep the lookup off the hot path
    auto const td_egress = (intermodal && !td_dist_to_end_.empty())
                               ? td_dist_to_end_.find(l)
                               : end(td_dist_to_end_);
    auto const td_egress_ok = td_egress != end(td_dist_to_end_);
    if ((use_td_fps ? false : fps.empty()) && !egress_ok && !td_egress_ok) {
      return;
    }

    // Copy the by-route labels first (relaxing inserts into the same bag) and
    // undo the transfer buffer so footpaths start from the transit arrival.
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

    // egress to the intermodal target, plain and/or time-dependent
    auto const relax_egress = [&](typename state_t::label const& te,
                                  std::uint16_t const duration) {
      auto const end_crit = te.crit_.with_walk(dir(duration), duration);
      // window bound: keeps pong's reverse searches from writing journeys
      // beyond the ping's start into the destination frontier
      if (!is_better(end_crit.arr_, worst_at_dest_)) {
        return;
      }
      auto bc = state_.breadcrumbs_[te.breadcrumb_];
      bc.arr_ = end_crit.arr_;
      if (!merge_round(to_idx(kIntermodalTarget), end_crit, bc,
                       static_cast<std::uint8_t>(k))) {
        return;
      }
      dest_bag_add(k, end_crit);
    };

    // `duration` is the effective walking time: static footpaths are already
    // adjusted, td ones carry their own (waiting included) duration.
    auto const relax_fp = [&](typename state_t::label const& te,
                              std::uint32_t const target,
                              std::uint16_t const duration) {
      auto const fp_crit = te.crit_.with_walk(dir(duration), duration);
      auto const fp_target_time = fp_crit.arr_;
      if (!is_better(fp_target_time, worst_at_dest_)) {
        return;
      }
      auto const lower_bound = effective_lb(target);
      if (lower_bound == kUnreachable ||
          !is_better(fp_target_time + dir(lower_bound), worst_at_dest_)) {
        ++stats_.fp_update_prevented_by_lower_bound_;
        return;
      }
      if (dest_dominates(k, fp_crit.projected_to(
                                clamp(fp_target_time + dir(lower_bound))))) {
        ++stats_.fp_update_prevented_by_lower_bound_;
        return;
      }
      // a footpath arrival pays no transfer buffer: relax the bound by it
      if (bound_prunes(k, target, fp_target_time, transfer_buffer(target))) {
        ++stats_.fp_update_prevented_by_lower_bound_;
        return;
      }
      // the breadcrumb keeps the ride's payload; the footpath is derived at
      // reconstruction
      auto bc = state_.breadcrumbs_[te.breadcrumb_];
      bc.arr_ = fp_crit.arr_;
      if (!merge_round(target, fp_crit, bc, static_cast<std::uint8_t>(k))) {
        return;
      }
      ++stats_.n_earliest_arrival_updated_by_footpath_;
      state_.station_mark_.set(target, true);
      if (is_dest_[target]) {
        dest_bag_add(k, fp_crit);
      }
    };

    if (egress_ok) {
      for (auto const& te : fp_labels_) {
        relax_egress(te, dist_to_end_[i]);
      }
    }

    // td durations depend on the arrival, so they are evaluated per label
    if (td_egress_ok) {
      for (auto const& te : fp_labels_) {
        auto const fp = get_td_duration<SearchDir>(td_egress->second,
                                                   to_unix(te.crit_.arr_));
        if (fp.has_value()) {
          relax_egress(te, static_cast<std::uint16_t>(fp->first.count()));
        }
      }
    }

    if (use_td_fps) {
      auto const& td_fps = kFwd ? rtt_->td_footpaths_out_[prf_idx_][l]
                                : rtt_->td_footpaths_in_[prf_idx_][l];
      for (auto const& te : fp_labels_) {
        for_each_footpath<SearchDir>(
            td_fps, to_unix(te.crit_.arr_), [&](footpath const fp) {
              ++stats_.n_footpaths_visited_;
              auto const target = to_idx(fp.target());
              if (target == i) {
                return;
              }
              relax_fp(te, static_cast<std::uint32_t>(target),
                       static_cast<std::uint16_t>(fp.duration().count()));
            });
      }
    } else {
      for (auto const& fp : fps) {
        ++stats_.n_footpaths_visited_;
        auto const target = to_idx(fp.target());
        if (target == i) {
          continue;
        }
        auto const fp_duration = adjusted_transfer_time(transfer_time_settings_,
                                                        fp.duration().count());
        for (auto const& te : fp_labels_) {
          relax_fp(te, static_cast<std::uint32_t>(target),
                   static_cast<std::uint16_t>(fp_duration));
        }
      }
    }
  });
}

template <direction SearchDir, typename Criteria, bool RangeReuse>
transport
basic_mcraptor<SearchDir, Criteria, RangeReuse>::get_earliest_transport(
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

      if (is_better_or_eq(worst_at_dest_, to_delta(day, ev_mam) +
                                              dir(effective_lb(to_idx(l))))) {
        return {transport_idx_t::invalid(), day_idx_t::invalid()};
      }

      auto const t = tt_.route_transport_ranges_[r][t_offset];
      if (i == 0U && !is_better_or_eq(mam_at_stop.count(), ev_mam)) {
        continue;
      }

      auto const ev_day_offset = ev.days();
      auto const start_day =
          static_cast<day_idx_t>(as_int(day) - ev_day_offset);
      if (!is_transport_active(t, start_day)) {
        continue;
      }

      return {t, start_day};
    }
  }
  return {};
}

namespace {

// Label dominance lifted to journeys (see set_intermediate_results): the
// generalized cost is priced from the departure, so it is compared with the
// departure discounted (cf. arr_cost_criteria::reuse_dominates), which for a
// journey is criteria_cost_ -/+ dest_time_ (extras = cost - travel time).
template <direction SearchDir, typename Criteria>
bool label_dominates(journey const& a, journey const& b) {
  constexpr auto const kFwd = SearchDir == direction::kForward;
  auto const cost = [](journey const& j) {
    if constexpr (std::is_same_v<Criteria, arr_cost_criteria>) {
      auto const dest =
          static_cast<int>(j.dest_time_.time_since_epoch().count());
      return static_cast<int>(j.criteria_cost_) + (kFwd ? -dest : dest);
    } else {
      return static_cast<int>(j.criteria_cost_);
    }
  };
  return a.transfers_ <= b.transfers_ &&
         (kFwd ? a.dest_time_ <= b.dest_time_ : a.dest_time_ >= b.dest_time_) &&
         cost(a) <= cost(b) &&
         a.criteria_mode_filter_ <= b.criteria_mode_filter_ &&
         a.criteria_mode_switches_ <= b.criteria_mode_switches_;
}

}  // namespace

template <direction SearchDir, typename Criteria, bool RangeReuse>
void basic_mcraptor<SearchDir, Criteria, RangeReuse>::collect_dest_journeys(
    unsigned const k,
    unixtime_t const start_time,
    pareto_set<journey>& results) {
  is_dest_.for_each_set_bit([&](std::uint64_t const i) {
    for_each_label_in_round<Criteria>(
        state_.bag_, static_cast<std::uint32_t>(i),
        static_cast<std::uint8_t>(k), cur_dep_,
        [&](Criteria const& e_crit, std::uint32_t const e_breadcrumb) {
          auto const j_start =
              tight_start_
                  ? tighten_start(e_breadcrumb & state_t::kBreadcrumbMask,
                                  start_time)
                  : start_time;
          // most candidates are dominated; materializing allocates
          auto probe = journey{};
          probe.start_time_ = j_start;
          probe.dest_time_ = to_unix(e_crit.arr_);
          probe.dest_ = location_idx_t{i};
          probe.transfers_ = static_cast<std::uint8_t>(k - 1U);
          e_crit.apply_to(probe);
          if (intermediate_results_) {
            if (utl::any_of(results.els_, [&](journey const& x) {
                  return label_dominates<SearchDir, Criteria>(x, probe);
                })) {
              return;
            }
            utl::erase_if(results.els_, [&](journey const& x) {
              return label_dominates<SearchDir, Criteria>(probe, x);
            });
            results.add_not_optimal(
                materialize(location_idx_t{i}, k, e_crit,
                            e_breadcrumb & state_t::kBreadcrumbMask, j_start));
            return;
          }
          if (results.is_dominated(probe)) {
            return;
          }
          auto j =
              materialize(location_idx_t{i}, k, e_crit,
                          e_breadcrumb & state_t::kBreadcrumbMask, j_start);
          results.add(std::move(j));
        });
  });
}

// The traffic day of the trip behind a breadcrumb. Its arrival at `alight` is
// stored, and a single footpath/transfer crosses midnight at most once, so the
// day is arr_day - event_day_offset - {0, 1}. Returns the day and the trip's
// event time at `alight`.
template <direction SearchDir, typename Criteria, bool RangeReuse>
std::optional<std::pair<day_idx_t, delta_t>>
basic_mcraptor<SearchDir, Criteria, RangeReuse>::recover_day(
    route_idx_t const r,
    transport_idx_t const t_idx,
    stop_idx_t const alight,
    delta_t const arr) const {
  constexpr auto const arr_ev = kFwd ? event_type::kArr : event_type::kDep;
  auto const event_day_offset =
      tt_.event_mam(r, t_idx, alight, arr_ev).count() / 1440;
  auto const arr_day = as_int(split(arr).first);
  for (auto off = 0; off != 2; ++off) {
    auto const cand = arr_day - event_day_offset - (kFwd ? off : -off);
    if (cand < 0) {
      continue;
    }
    auto const cand_day = day_idx_t{static_cast<day_idx_t::value_t>(cand)};
    if (!is_transport_active(t_idx, cand_day)) {
      continue;
    }
    auto const ev = time_at_stop(r, transport{t_idx, cand_day}, alight, arr_ev);
    if (is_better_or_eq(ev, arr)) {
      return std::pair{cand_day, ev};
    }
  }
  return std::nullopt;
}

// Latest feasible departure of the journey behind a destination label (see
// set_tight_start): chase the breadcrumbs to the first ride and re-anchor at
// the minimum-walk round-0 label of its boarding stop that still makes the
// boarding departure - the label the route scan boarded. Falls back to the
// step start defensively.
template <direction SearchDir, typename Criteria, bool RangeReuse>
unixtime_t basic_mcraptor<SearchDir, Criteria, RangeReuse>::tighten_start(
    std::uint32_t const breadcrumb_idx, unixtime_t const step_start) {
  auto li = breadcrumb_idx;
  if (li == state_t::kNoBreadcrumb) {
    return step_start;
  }
  while (state_.breadcrumbs_[li].parent_ != state_t::kNoBreadcrumb) {
    li = state_.breadcrumbs_[li].parent_;
  }
  auto const& bc = state_.breadcrumbs_[li];
  auto const bc_t = bc_transport(bc.payload_);
  auto const is_rt = is_rt_bc_transport(bc_t, n_rt_transports_);
  auto const board = static_cast<stop_idx_t>(bc_board(bc.payload_));
  auto const alight = static_cast<stop_idx_t>(bc_alight(bc.payload_));
  auto const dep_ev = kFwd ? event_type::kDep : event_type::kArr;

  auto dep_at_board = kInvalid;
  auto board_loc = location_idx_t::invalid();

  if (is_rt) {
    auto const rt_t = rt_transport_idx_t{decode_rt_bc_transport(bc_t)};
    dep_at_board = rt_time_at_stop(rt_t, board, dep_ev);
    board_loc =
        stop{rtt_->rt_transport_location_seq_[rt_t][board]}.location_idx();
  } else {
    auto const t_idx = transport_idx_t{bc_t};
    auto const r = tt_.transport_route_[t_idx];

    if (auto const rec = recover_day(r, t_idx, alight, bc.arr_)) {
      dep_at_board =
          time_at_stop(r, transport{t_idx, rec->first}, board, dep_ev);
    }
    board_loc = stop{tt_.route_location_seq_[r][board]}.location_idx();
  }
  if (dep_at_board == kInvalid) {
    return step_start;
  }

  auto best = kInvalid;
  for_each_label_in_round<Criteria>(
      state_.bag_, static_cast<std::uint32_t>(to_idx(board_loc)),
      std::uint8_t{0U}, cur_dep_, [&](Criteria const& c, std::uint32_t) {
        if (is_better_or_eq(c.arr_, dep_at_board) && is_better(c.arr_, best)) {
          best = c.arr_;
        }
      });
  if (best == kInvalid) {
    return step_start;
  }
  return step_start +
         duration_t{static_cast<duration_t::rep>(dep_at_board - best)};
}

template <direction SearchDir, typename Criteria, bool RangeReuse>
journey basic_mcraptor<SearchDir, Criteria, RangeReuse>::materialize(
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

  // Chase the breadcrumbs from the destination to the start; legs are
  // collected in search order.
  auto& legs = rec_legs_;
  legs.clear();

  auto const arr_ev = kFwd ? event_type::kArr : event_type::kDep;
  auto const dep_ev = kFwd ? event_type::kDep : event_type::kArr;

  auto cur_l = dest;
  auto li = breadcrumb_idx;
  while (li != state_t::kNoBreadcrumb) {
    auto const& bc = state_.breadcrumbs_[li];
    auto const cur_arr = bc.arr_;  // arrival at cur_l
    auto const bc_t = bc_transport(bc.payload_);
    auto const is_rt = is_rt_bc_transport(bc_t, n_rt_transports_);
    auto const t_idx =
        is_rt ? transport_idx_t::invalid() : transport_idx_t{bc_t};
    auto const rt_t = is_rt ? rt_transport_idx_t{decode_rt_bc_transport(bc_t)}
                            : rt_transport_idx_t::invalid();
    auto const board = static_cast<stop_idx_t>(bc_board(bc.payload_));
    auto const alight = static_cast<stop_idx_t>(bc_alight(bc.payload_));

    auto day = day_idx_t::invalid();
    auto train_arr = kInvalid;
    auto dep_at_board = kInvalid;
    auto board_loc = location_idx_t::invalid();
    auto alight_loc = location_idx_t::invalid();

    if (is_rt) {
      auto const stop_seq = rtt_->rt_transport_location_seq_[rt_t];
      board_loc = stop{stop_seq[board]}.location_idx();
      alight_loc = stop{stop_seq[alight]}.location_idx();
      train_arr = rt_time_at_stop(rt_t, alight, arr_ev);
      dep_at_board = rt_time_at_stop(rt_t, board, dep_ev);
    } else {
      auto const r = tt_.transport_route_[t_idx];

      auto const rec = recover_day(r, t_idx, alight, cur_arr);
      utl::verify(rec.has_value(),
                  "mcraptor reconstruct: traffic day recovery failed");
      std::tie(day, train_arr) = *rec;

      dep_at_board = time_at_stop(r, transport{t_idx, day}, board, dep_ev);
      auto const stop_seq = tt_.route_location_seq_[r];
      board_loc = stop{stop_seq[board]}.location_idx();
      alight_loc = stop{stop_seq[alight]}.location_idx();
    }

    if (is_intermodal_dest() && cur_l == kIntermodalTarget) {
      // the last mile is added by reconstruct()
      j.dest_ = alight_loc;
    } else if (!legs.empty() || alight_loc != cur_l || train_arr != cur_arr) {
      // footpaths are always emitted, even zero-minute transfers
      legs.push_back(
          {.is_footpath_ = true,
           .from_ = alight_loc,
           .to_ = cur_l,
           .dep_ = train_arr,
           .arr_ = cur_arr,
           .t_ = transport_idx_t::invalid(),
           .rt_ = rt_transport_idx_t::invalid(),
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
                    .rt_ = rt_t,
                    .day_ = day,
                    .enter_ = board,
                    .exit_ = alight,
                    .fp_duration_ = 0U});

    cur_l = board_loc;
    li = bc.parent_;
  }

  // legs in chronological order
  for (auto z = 0U; z != legs.size(); ++z) {
    auto const& gl = kFwd ? legs[legs.size() - 1U - z] : legs[z];
    auto const dep = to_unix(gl.dep_);
    auto const arr = to_unix(gl.arr_);
    if (gl.is_footpath_) {
      j.legs_.emplace_back(journey::leg{
          SearchDir, gl.from_, gl.to_, dep, arr,
          footpath{gl.to_,
                   duration_t{static_cast<duration_t::rep>(gl.fp_duration_)}}});
    } else if (gl.rt_ != rt_transport_idx_t::invalid()) {
      auto const run =
          rt::run{.t_ = rtt_->resolve_static(gl.rt_),
                  .stop_range_ =
                      interval<stop_idx_t>{
                          stop_idx_t{0U},
                          static_cast<stop_idx_t>(
                              rtt_->rt_transport_location_seq_[gl.rt_].size())},
                  .rt_ = gl.rt_};
      j.legs_.emplace_back(
          journey::leg{SearchDir, gl.from_, gl.to_, dep, arr,
                       journey::run_enter_exit{run, gl.enter_, gl.exit_}});
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

  // a backward search re-anchors footpaths to the previous trip's arrival
  for (auto z = std::size_t{1U}; z < j.legs_.size(); ++z) {
    if (std::holds_alternative<footpath>(j.legs_[z].uses_)) {
      auto const dur = std::get<footpath>(j.legs_[z].uses_).duration();
      j.legs_[z].dep_time_ = j.legs_[z - 1U].arr_time_;
      j.legs_[z].arr_time_ = j.legs_[z].dep_time_ + dur;
    }
  }

  return j;
}

// Adds the first/last-mile offset legs and the start footpath.
template <direction SearchDir, typename Criteria, bool RangeReuse>
void basic_mcraptor<SearchDir, Criteria, RangeReuse>::reconstruct(
    query const& q, journey& j) {
  utl::verify(!j.legs_.empty(), "mcraptor reconstruct: journey without legs");

  constexpr auto const is_fwd = SearchDir == direction::kForward;

  // front: special_station -> first transit stop
  auto const from = j.legs_.front().from_;
  auto const dep_time = j.legs_.front().dep_time_;
  auto const front_match_mode =
      is_fwd ? q.start_match_mode_ : q.dest_match_mode_;
  if (front_match_mode == location_match_mode::kIntermodal) {
    auto const& offsets = is_fwd ? q.start_ : q.destination_;
    auto const special = get_special_station(is_fwd ? special_station::kStart
                                                    : special_station::kEnd);
    // the front leg ends at the first transit event
    auto const front_ok = [&](duration_t const d) {
      return is_fwd
                 // fwd: query start, check feasibility (allows ontrip start)
                 ? dep_time - d >= j.start_time_
                 // bwd: destination, anchored exactly at j.dest_time_
                 : dep_time - d == j.dest_time_;
    };
    auto const o = utl::find_if(offsets, [&](offset const& x) {
      return matches(tt_, front_match_mode, x.target(), from) &&
             front_ok(x.duration());
    });
    auto front = std::optional<offset>{};
    auto front_dep = dep_time;  // set with the offset, see the td case
    if (o != end(offsets)) {
      front = *o;
      front_dep = dep_time - o->duration();
    } else if (auto const& td = is_fwd ? q.td_start_ : q.td_dest_;
               td.contains(from)) {
      // td offsets are valid at fixed times and the traveller then waits for
      // the first transit event: the backward query gives the latest departure
      // that makes the boarding, the forward query from there strips the wait.
      auto const& offs = td.at(from);
      auto const back = get_td_duration<direction::kBackward>(offs, dep_time);
      if (back.has_value() && front_ok(back->first)) {
        auto const start = dep_time - back->first;
        auto const fwd = get_td_duration<direction::kForward>(offs, start);
        if (fwd.has_value()) {
          front = offset{from, fwd->first, fwd->second.mode()};
          front_dep = start;
        }
      }
    }
    utl::verify(front.has_value(), "mcraptor reconstruct: no front offset");
    auto const dep = front_dep;
    auto const arr = front_dep + front->duration();
    j.legs_.insert(begin(j.legs_), journey::leg{direction::kForward, special,
                                                from, dep, arr, *front});
  }

  // back: last transit stop -> special_station
  auto const to = j.legs_.back().to_;
  auto const arr_time = j.legs_.back().arr_time_;
  auto const back_match_mode =
      is_fwd ? q.dest_match_mode_ : q.start_match_mode_;
  if (back_match_mode == location_match_mode::kIntermodal) {
    auto const& offsets = is_fwd ? q.destination_ : q.start_;
    auto const special = get_special_station(is_fwd ? special_station::kEnd
                                                    : special_station::kStart);
    // the back leg starts at the last transit event
    auto const back_ok = [&](duration_t const d) {
      return is_fwd
                 // fwd: destination, anchored exactly at j.dest_time_
                 ? arr_time + d == j.dest_time_
                 // bwd: query start, anchored by feasibility
                 : arr_time + d <= j.start_time_;
    };
    auto const o = utl::find_if(offsets, [&](offset const& x) {
      return matches(tt_, back_match_mode, x.target(), to) &&
             back_ok(x.duration());
    });
    auto back = std::optional<offset>{};
    auto back_dep = arr_time;  // set with the offset, see the td case
    if (o != end(offsets)) {
      back = *o;
    } else if (auto const& td = is_fwd ? q.td_dest_ : q.td_start_;
               td.contains(to)) {
      // mirrors the td first mile above
      auto const& offs = td.at(to);
      auto const fwd = get_td_duration<direction::kForward>(offs, arr_time);
      if (fwd.has_value() && back_ok(fwd->first)) {
        auto const journey_end = arr_time + fwd->first;
        auto const bck =
            get_td_duration<direction::kBackward>(offs, journey_end);
        if (bck.has_value() && journey_end - bck->first >= arr_time) {
          back = offset{to, bck->first, bck->second.mode()};
          back_dep = journey_end - bck->first;
        }
      }
    }
    utl::verify(back.has_value(), "mcraptor reconstruct: no back offset");
    auto const dep = back_dep;
    auto const arr = back_dep + back->duration();
    j.legs_.push_back(
        journey::leg{direction::kForward, to, special, dep, arr, *back});
    j.dest_ = special;
  }

  // the start footpath that seeded round 0
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

  // Shorten td footpath legs to their walking time: the search stores the wait
  // in the arrival, but showing it as walking is wrong and misprices the
  // transfer for optimize_footpaths.
  if (has_rt_ && q.prf_idx_ != 0U) {
    auto const& has_td = kFwd ? rtt_->has_td_footpaths_in_[q.prf_idx_]
                              : rtt_->has_td_footpaths_out_[q.prf_idx_];
    auto const& td_fps = kFwd ? rtt_->td_footpaths_in_[q.prf_idx_]
                              : rtt_->td_footpaths_out_[q.prf_idx_];
    if (!td_fps.empty()) {
      for (auto& lg : j.legs_) {
        if (!std::holds_alternative<footpath>(lg.uses_)) {
          continue;
        }
        auto const key_l = kFwd ? lg.to_ : lg.from_;
        auto const target_l = kFwd ? lg.from_ : lg.to_;
        if (!has_td.test(key_l)) {
          continue;
        }
        auto const t = lg.arr_time_;
        for_each_footpath<SearchDir>(td_fps[key_l], t, [&](footpath const fp) {
          if (fp.target() != target_l) {
            return utl::cflow::kContinue;
          }
          lg.dep_time_ = t - fp.duration();
          lg.arr_time_ = t;
          lg.uses_ = footpath{lg.to_, fp.duration()};
          return utl::cflow::kBreak;
        });
      }
    }
  }

  if constexpr (is_fwd) {
    optimize_footpaths(tt_, rtt_, q, j);
  } else {
    // legs are chronological, q is in search direction
    auto journey_q = q;
    journey_q.flip_dir();
    optimize_footpaths(tt_, rtt_, journey_q, j);
  }

  j.is_reconstructed_ = true;
}

template <direction SearchDir, typename Criteria, bool RangeReuse>
delta_t basic_mcraptor<SearchDir, Criteria, RangeReuse>::time_at_stop(
    route_idx_t const r,
    transport const t,
    stop_idx_t const stop_idx,
    event_type const ev_type) const {
  return to_delta(t.day_,
                  tt_.event_mam(r, t.t_idx_, stop_idx, ev_type).count());
}

template <direction SearchDir, typename Criteria, bool RangeReuse>
delta_t basic_mcraptor<SearchDir, Criteria, RangeReuse>::rt_time_at_stop(
    rt_transport_idx_t const rt_t,
    stop_idx_t const stop_idx,
    event_type const ev_type) const {
  return to_delta(rtt_->base_day_idx_,
                  rtt_->event_time(rt_t, stop_idx, ev_type));
}

template <direction SearchDir, typename Criteria, bool RangeReuse>
bool basic_mcraptor<SearchDir, Criteria, RangeReuse>::is_transport_active(
    transport_idx_t const t, day_idx_t const day) const {
  return has_rt_ ? rtt_->is_transport_active(t, day)
                 : tt_.is_transport_active(t, day);
}

template <direction SearchDir, typename Criteria, bool RangeReuse>
delta_t basic_mcraptor<SearchDir, Criteria, RangeReuse>::to_delta(
    day_idx_t const day, std::int16_t const mam) const {
  return clamp((as_int(day) - as_int(base_)) * 1440 + mam);
}

template <direction SearchDir, typename Criteria, bool RangeReuse>
unixtime_t basic_mcraptor<SearchDir, Criteria, RangeReuse>::to_unix(
    delta_t const t) const {
  return delta_to_unix(base(), t);
}

template <direction SearchDir, typename Criteria, bool RangeReuse>
std::pair<day_idx_t, minutes_after_midnight_t>
basic_mcraptor<SearchDir, Criteria, RangeReuse>::split(delta_t const x) const {
  return split_day_mam(base_, x);
}

// One line per criteria configuration: both directions, range reuse off/on.
#define NIGIRI_MC_INSTANTIATE(C)                                  \
  template struct basic_mcraptor<direction::kForward, C, false>;  \
  template struct basic_mcraptor<direction::kBackward, C, false>; \
  template struct basic_mcraptor<direction::kForward, C, true>;   \
  template struct basic_mcraptor<direction::kBackward, C, true>;

NIGIRI_MC_INSTANTIATE(arr_criteria)
NIGIRI_MC_INSTANTIATE(arr_cost_criteria)
NIGIRI_MC_INSTANTIATE(arr_non_transit_criteria)
NIGIRI_MC_INSTANTIATE(arr_mode_filter_criteria)
NIGIRI_MC_INSTANTIATE(arr_mode_switches_criteria)
NIGIRI_MC_INSTANTIATE(arr_non_transit_mode_filter_criteria)
NIGIRI_MC_INSTANTIATE(arr_non_transit_mode_switches_criteria)
NIGIRI_MC_INSTANTIATE(arr_mode_filter_mode_switches_criteria)
NIGIRI_MC_INSTANTIATE(arr_non_transit_mode_filter_mode_switches_criteria)

#undef NIGIRI_MC_INSTANTIATE

}  // namespace nigiri::routing
