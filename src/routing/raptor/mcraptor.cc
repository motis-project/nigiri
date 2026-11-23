#include "nigiri/routing/raptor/mcraptor.h"

#include <cassert>
#include <algorithm>
#include <span>

#include "utl/verify.h"

#include "nigiri/location_routes.h"
#include "nigiri/routing/raptor/mc_reconstruct.h"
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

template <direction SearchDir, typename Criteria>
basic_mcraptor<SearchDir, Criteria>::basic_mcraptor(
    timetable const& tt,
    rt_timetable const* rtt,
    state_t& state,
    bitvec const& is_dest,
    std::array<bitvec, kMaxVias> const&,
    std::vector<std::uint16_t> const& dist_to_dest,
    hash_map<location_idx_t, std::vector<td_offset>> const& td_dist_to_dest,
    std::vector<std::uint16_t> const& lb,
    std::vector<via_stop> const& via_stops,
    day_idx_t const base,
    clasz_mask_t const allowed_claszes,
    bool const require_bike_transport,
    bool const require_car_transport,
    bool const is_wheelchair,
    bool const no_compulsory_reservation,
    transfer_time_settings const& tts,
    profile_idx_t const prf_idx,
    mc_context const& ctx)
    : tt_{tt},
      rtt_{rtt},
      state_{
          state.resize(rtt != nullptr ? rtt->n_locations() : tt.n_locations(),
                       tt.n_routes(),
                       rtt != nullptr ? rtt->n_rt_transports() : 0U)},
      is_dest_{is_dest},
      dist_to_end_{dist_to_dest},
      td_dist_to_end_{td_dist_to_dest},
      lb_{lb},
      base_{base},
      allowed_claszes_{allowed_claszes},
      ctx_{ctx},
      clock_{tt, rtt},
      is_wheelchair_{is_wheelchair},
      transfer_time_settings_{tts},
      prf_idx_{prf_idx} {
  utl::verify(via_stops.empty(), "mcraptor: via stops not supported");
  utl::verify(!require_bike_transport && !require_car_transport &&
                  !no_compulsory_reservation,
              "mcraptor: bike/car transport, reservation filter not supported");
  reset_arrivals();
}

template <direction SearchDir, typename Criteria>
std::pair<int, int> basic_mcraptor<SearchDir, Criteria>::local_mams(
    std::span<stop::value_type const> const seq,
    stop_idx_t const board,
    stop_idx_t const alight,
    delta_t const board_time,
    delta_t const alight_time,
    timezone_idx_t const trip_tz) {
  return {
      clock_.mam(to_unix(board_time), stop{seq[board]}.location_idx(), trip_tz),
      clock_.mam(to_unix(alight_time), stop{seq[alight]}.location_idx(),
                 trip_tz)};
}

template <direction SearchDir, typename Criteria>
date::sys_days basic_mcraptor<SearchDir, Criteria>::base() const {
  return tt_.internal_interval_days().from_ + as_int(base_) * date::days{1};
}

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
  seeds_.clear();
  state_.prev_station_mark_.zero_out();
  state_.station_mark_.zero_out();
  state_.route_mark_.zero_out();
  state_.rt_transport_mark_.zero_out();
}

template <direction SearchDir, typename Criteria>
template <typename Fn>
void basic_mcraptor<SearchDir, Criteria>::for_each_label(std::uint32_t const l,
                                                         unsigned const round,
                                                         Fn&& fn) const {
  for (auto const& e : state_.bag_.span(l)) {
    if (e.round_ == round && e.dep_ == cur_dep_) {
      fn(e);
    }
  }
}

// A candidate is rejected by labels of no higher round, evicts labels of no
// lower round. By-route labels are only rejected by by-route ones (their
// footpaths are still to relax). Where the same-stop transfer is banned they
// cannot board, so the two kinds do not compare at all.
template <direction SearchDir, typename Criteria>
bool basic_mcraptor<SearchDir, Criteria>::bag_insert(
    std::uint32_t const l,
    Criteria const& crit,
    std::uint32_t const breadcrumb,
    unsigned const round,
    bool const by_route) {
  auto& layer = state_.bag_;
  if (layer.empty(l)) {
    layer.touched_.set(l);
  }
  auto const bag = layer.span(l);
  auto const separate = no_transfer(l);
  for (auto const& e : bag) {
    if (e.round_ <= round &&
        (separate ? e.by_route_ == by_route : !by_route || e.by_route_) &&
        e.crit_.template dominates<SearchDir>(crit)) {
      return false;
    }
  }
  auto removed = std::size_t{0U};
  for (auto i = std::size_t{0U}; i != bag.size(); ++i) {
    if (bag[i].round_ >= round &&
        (separate ? bag[i].by_route_ == by_route
                  : by_route || !bag[i].by_route_) &&
        crit.template dominates<SearchDir>(bag[i].crit_)) {
      ++removed;
      continue;
    }
    bag[i - removed] = bag[i];
  }
  layer.set_size(l, static_cast<std::uint32_t>(bag.size() - removed));
  layer.push_back(l, {.crit_ = crit,
                      .breadcrumb_ = breadcrumb,
                      .by_route_ = by_route,
                      .round_ = static_cast<std::uint8_t>(round),
                      .dep_ = cur_dep_});
  return true;
}

// same-stop transfer buffer; 0 at a non-intermodal destination (or a virt of
// it) and where the transfer is banned (those labels do not board)
template <direction SearchDir, typename Criteria>
delta_t basic_mcraptor<SearchDir, Criteria>::transfer_buffer(
    std::uint64_t const l) const {
  auto const i = static_cast<std::uint32_t>(l);
  if (!is_intermodal_dest() && is_dest_[to_idx(base_of(i))]) {
    return 0;
  }
  auto const t = get_transfer_time(tt_, rtt_, location_idx_t{i});
  return t == kNoTransferAllowed
             ? delta_t{0}
             : static_cast<delta_t>(dir(adjusted_transfer_time(
                   transfer_time_settings_, static_cast<int>(t.count()))));
}

template <direction SearchDir, typename Criteria>
bool basic_mcraptor<SearchDir, Criteria>::merge_round(std::uint32_t const l,
                                                      Criteria const& crit,
                                                      mc_breadcrumb const& bc,
                                                      unsigned const round,
                                                      bool const by_route) {
  auto const idx = static_cast<std::uint32_t>(state_.breadcrumbs_.size());
  if (!bag_insert(l, crit, idx, round, by_route)) {
    return false;
  }
  state_.breadcrumbs_.push_back(bc);
  return true;
}

template <direction SearchDir, typename Criteria>
void basic_mcraptor<SearchDir, Criteria>::add_start(location_idx_t const l,
                                                    unixtime_t const t,
                                                    via_offset_t) {
  auto const i = to_idx(l);
  seeds_.emplace_back(static_cast<std::uint32_t>(i), unix_to_delta(base(), t));
  state_.station_mark_.set(i, true);
}

template <direction SearchDir, typename Criteria>
void basic_mcraptor<SearchDir, Criteria>::execute(
    unixtime_t const start_time,
    std::uint8_t const max_transfers,
    std::array<unixtime_t, kMaxTransfers + 2U> const& worst_time_at_dest,
    pareto_set<journey>& results) {
  auto const end_k = std::min(max_transfers, kMaxTransfers) + 2U;
  cur_budget_ = end_k - 1U;

  // one bound for all rounds: the loosest
  auto const worst = kFwd ? std::ranges::max(worst_time_at_dest)
                          : std::ranges::min(worst_time_at_dest);
  worst_at_dest_ = get_best(unix_to_delta(base(), worst), worst_at_dest_);

  auto const d_start = unix_to_delta(base(), start_time);
  cur_dep_ = d_start;
  for (auto const& [l, arr] : seeds_) {
    auto const walk =
        seed_walk<SearchDir>(tt_, rtt_, start_, location_idx_t{l}, start_time,
                             static_cast<std::uint16_t>(dir(arr - d_start)));
    bag_insert(l, Criteria::at_start(arr, walk), state_t::kNoBreadcrumb,
               /*round=*/0U, /*by_route=*/false);
  }
  seeds_.clear();

  for (auto k = 1U; k != end_k; ++k) {
    auto any_marked = false;
    state_.station_mark_.for_each_set_bit([&](std::uint64_t const i) {
      auto const l = location_idx_t{i};
      for (auto const& r : static_routes(tt_, rtt_, l)) {
        any_marked = true;
        state_.route_mark_.set(to_idx(r), true);
      }
      if (rtt_ != nullptr) {
        for (auto const& rt_t : rtt_->location_rt_transports_[l]) {
          if (!rtt_->rt_transport_is_cancelled_[to_idx(rt_t)]) {
            any_marked = true;
            state_.rt_transport_mark_.set(to_idx(rt_t), true);
          }
        }
      }
    });

    if (!any_marked) {
      break;
    }

    std::swap(state_.prev_station_mark_, state_.station_mark_);
    state_.station_mark_.zero_out();

    // both read round k - 1 and write round k: order does not matter
    any_marked = loop_routes(k);
    if (rtt_ != nullptr) {
      any_marked = loop_rt_transports(k) || any_marked;
    }

    state_.route_mark_.zero_out();
    state_.rt_transport_mark_.zero_out();
    if (!any_marked) {
      break;
    }

    update_footpaths(k);

    collect_dest_journeys(k, start_time, results);
  }
}

template <direction SearchDir, typename Criteria>
bool basic_mcraptor<SearchDir, Criteria>::alight(
    unsigned const k,
    std::uint32_t const l_idx,
    stop_idx_t const stop_idx,
    delta_t const by_transport,
    std::pair<int, int> const local,
    route_label const& rl,
    clasz const ride_clasz,
    std::uint32_t const transport,
    day_idx_t const day) {
  auto const lb = effective_lb(l_idx);
  if (!is_better(by_transport, worst_at_dest_) || lb == kUnreachable ||
      !is_better(by_transport + dir(lb), worst_at_dest_) ||
      bound_prunes(k, l_idx, by_transport)) {
    ++stats_.route_update_prevented_by_lower_bound_;
    return false;
  }
  auto const ride_crit =
      Criteria::from_ride(by_transport, local, ride_clasz, rl.carried_, ctx_);
  if (!guide_feasible(l_idx, k, by_transport, ride_crit.extras_) ||
      dest_dominates(k,
                     ride_crit.projected_to(clamp(by_transport + dir(lb))))) {
    ++stats_.route_update_prevented_by_lower_bound_;
    return false;
  }
  auto const post_crit = ride_crit.with_transfer(transfer_buffer(l_idx));
  if (!merge_round(l_idx, post_crit,
                   {.transport_ = transport,
                    .parent_ = rl.parent_,
                    .day_ = day,
                    .board_ = rl.board_,
                    .alight_ = stop_idx,
                    .arr_ = post_crit.arr_},
                   k, /*by_route=*/true)) {
    return false;
  }
  ++stats_.n_earliest_arrival_updated_by_route_;
  state_.station_mark_.set(l_idx, true);
  // no buffer at a station destination; a virt arrives at its base
  auto const dest_l = is_intermodal_dest()
                          ? l_idx
                          : static_cast<std::uint32_t>(to_idx(base_of(l_idx)));
  if (is_dest_[dest_l]) {
    if (dest_l != l_idx) {
      merge_round(dest_l, ride_crit,
                  {.transport_ = transport,
                   .parent_ = rl.parent_,
                   .day_ = day,
                   .board_ = rl.board_,
                   .alight_ = stop_idx,
                   .arr_ = ride_crit.arr_},
                  k, /*by_route=*/false);
    }
    dest_bag_add(k, ride_crit);
  }
  return true;
}

template <direction SearchDir, typename Criteria>
bool basic_mcraptor<SearchDir, Criteria>::loop_routes(unsigned const k) {
  auto any_marked = false;
  state_.route_mark_.for_each_set_bit([&](auto const r_idx) {
    auto const r = route_idx_t{r_idx};
    if (!is_allowed(allowed_claszes_, tt_.route_clasz_[r]) ||
        (is_wheelchair_ &&
         !tt_.is_flag_set(route_flag::kWheelchairAccessible, r))) {
      return;
    }
    ++stats_.n_routes_visited_;
    any_marked |= update_route(k, r);
  });
  return any_marked;
}

// Per stop: alight route_bag_, then board the last round's labels. A section
// that is not ridable drops the boarded trips.
template <direction SearchDir, typename Criteria>
template <typename Stops, typename Ridable, typename Alight, typename Board>
bool basic_mcraptor<SearchDir, Criteria>::scan_stops(
    unsigned const k,
    Stops const& stop_seq,
    std::span<location_idx_t const> const virts,
    Ridable&& ridable,
    Alight&& alight_label,
    Board&& board_label) {
  auto const n = static_cast<unsigned>(stop_seq.size());
  auto any_marked = false;
  route_bag_.clear();
  for (auto i = 0U; i != n; ++i) {
    auto const stop_idx = static_cast<stop_idx_t>(kFwd ? i : n - i - 1U);
    auto const stp = stop{stop_seq[stop_idx]};
    auto const l = virts.empty() || virts[stop_idx] == location_idx_t::invalid()
                       ? stp.location_idx()
                       : virts[stop_idx];
    auto const l_idx = static_cast<std::uint32_t>(to_idx(l));

    if (i != 0U && !ridable(stop_idx)) {
      route_bag_.clear();
    }

    if (i != 0U && stp.can_finish<SearchDir>(is_wheelchair_)) {
      for (auto const& rl : route_bag_) {
        any_marked |= alight_label(rl, l_idx, stop_idx);
      }
    }

    if (i == n - 1U || !stp.can_start<SearchDir>(is_wheelchair_) ||
        !state_.prev_station_mark_[l_idx]) {
      continue;
    }

    auto const lb = effective_lb(l_idx);
    if (lb == kUnreachable) {
      break;
    }

    auto const banned = no_transfer(l_idx);
    for_each_label(l_idx, k - 1U, [&](typename state_t::label const& e) {
      if (!(banned && e.by_route_) &&
          !dest_dominates(
              k, e.crit_.projected_to(clamp(e.crit_.arr_ + dir(lb))))) {
        board_label(e.crit_, e.breadcrumb_, stop_idx, l);
      }
    });
  }
  return any_marked;
}

// pareto over (trip order, carried criteria)
template <direction SearchDir, typename Criteria>
void basic_mcraptor<SearchDir, Criteria>::board(
    std::uint32_t const key,
    transport const t,
    stop_idx_t const stop_idx,
    std::uint32_t const parent,
    typename Criteria::carried const carried) {
  for (auto& rl : route_bag_) {
    if (rl.key_ == key && rl.carried_ == carried) {
      // same trip and criteria: board closest to the exit
      rl.board_ = stop_idx;
      rl.parent_ = parent;
      return;
    }
    if (!is_earlier_trip(key, rl.key_) &&
        Criteria::carried_dominates(rl.carried_, carried)) {
      return;
    }
  }
  std::erase_if(route_bag_, [&](route_label const& rl) {
    return !is_earlier_trip(rl.key_, key) &&
           Criteria::carried_dominates(carried, rl.carried_);
  });
  route_bag_.push_back({t, key, stop_idx, parent, carried});
}

template <direction SearchDir, typename Criteria>
bool basic_mcraptor<SearchDir, Criteria>::update_route(unsigned const k,
                                                       route_idx_t const r) {
  // wheelchair: accessible on all sections, or check the section ridden into
  auto const all_sections =
      !is_wheelchair_ ||
      tt_.route_flags_[route_flag::kWheelchairAccessible].test(to_idx(r) * 2U);
  return scan_stops(
      k, tt_.route_location_seq_[r], {},
      [&](stop_idx_t const stop_idx) {
        return all_sections ||
               tt_.route_flags_per_section_[route_flag::kWheelchairAccessible]
                                           [r][kFwd ? stop_idx - 1 : stop_idx];
      },
      [&](route_label const& rl, std::uint32_t const l_idx,
          stop_idx_t const stop_idx) {
        auto const arr = time_at_stop(r, rl.t_, stop_idx, kArrEv);
        return alight(
            k, l_idx, stop_idx, arr,
            Criteria::extras_t::kRideTimes
                ? local_mams(tt_.route_location_seq_[r], rl.board_, stop_idx,
                             time_at_stop(r, rl.t_, rl.board_, kDepEv), arr,
                             clock_.trip_tz(rl.t_.t_idx_))
                : std::pair{0, 0},
            rl, tt_.route_clasz_[r], to_idx(rl.t_.t_idx_), rl.t_.day_);
      },
      [&](Criteria const& crit, std::uint32_t const parent,
          stop_idx_t const stop_idx, location_idx_t const l) {
        // skip the lookup if a boarded trip already dominates
        for (auto const& rl : route_bag_) {
          if (Criteria::carried_dominates(rl.carried_, crit.extras_) &&
              is_better(time_at_stop(r, rl.t_, stop_idx, kDepEv), crit.arr_)) {
            return;
          }
        }
        auto const [day, mam] = split(crit.arr_);
        auto const et = get_earliest_transport(r, stop_idx, day, mam, l);
        if (et.is_valid()) {
          board(trip_order_key(r, et), et, stop_idx, parent, crit.extras_);
        }
      });
}

template <direction SearchDir, typename Criteria>
bool basic_mcraptor<SearchDir, Criteria>::loop_rt_transports(unsigned const k) {
  auto any_marked = false;
  state_.rt_transport_mark_.for_each_set_bit([&](auto const rt_t_idx) {
    auto const rt_t = rt_transport_idx_t{rt_t_idx};
    if ((allowed_claszes_ != all_clasz_allowed() &&
         !is_allowed(allowed_claszes_,
                     rtt_->rt_transport_section_clasz_[rt_t][0])) ||
        (is_wheelchair_ &&
         !rtt_->is_flag_set(route_flag::kWheelchairAccessible, rt_t))) {
      return;
    }
    ++stats_.n_routes_visited_;
    any_marked |= update_rt_transport(k, rt_t);
  });
  return any_marked;
}

template <direction SearchDir, typename Criteria>
bool basic_mcraptor<SearchDir, Criteria>::update_rt_transport(
    unsigned const k, rt_transport_idx_t const rt_t) {
  auto const all_sections =
      !is_wheelchair_ ||
      rtt_->rt_transport_flags_[route_flag::kWheelchairAccessible].test(
          to_idx(rt_t) * 2U);
  return scan_stops(
      k, rtt_->rt_transport_location_seq_[rt_t],
      profile_uses_rt_virts(prf_idx_) ? rtt_->rt_stop_virts(rt_t)
                                      : std::span<location_idx_t const>{},
      [&](stop_idx_t const stop_idx) {
        return all_sections ||
               rtt_->rt_flags_per_section_[route_flag::kWheelchairAccessible]
                                          [rt_t]
                                          [kFwd ? stop_idx - 1 : stop_idx];
      },
      [&](route_label const& rl, std::uint32_t const l_idx,
          stop_idx_t const stop_idx) {
        auto const arr = rt_time_at_stop(rt_t, stop_idx, kArrEv);
        return alight(
            k, l_idx, stop_idx, arr,
            Criteria::extras_t::kRideTimes
                ? local_mams(rtt_->rt_transport_location_seq_[rt_t], rl.board_,
                             stop_idx, rt_time_at_stop(rt_t, rl.board_, kDepEv),
                             arr, clock_.trip_tz(rt_t))
                : std::pair{0, 0},
            rl, rtt_->rt_transport_section_clasz_[rt_t][0], to_idx(rt_t),
            day_idx_t::invalid());
      },
      [&](Criteria const& crit, std::uint32_t const parent,
          stop_idx_t const stop_idx, location_idx_t) {
        if (is_better_or_eq(crit.arr_,
                            rt_time_at_stop(rt_t, stop_idx, kDepEv))) {
          board(0U, transport{}, stop_idx, parent, crit.extras_);
        }
      });
}

// traffic day << 16 | trip offset: orders the trips of a route
template <direction SearchDir, typename Criteria>
std::uint32_t basic_mcraptor<SearchDir, Criteria>::trip_order_key(
    route_idx_t const r, transport const t) const {
  auto const t_offset = static_cast<std::uint32_t>(
      to_idx(t.t_idx_) - to_idx(tt_.route_transport_ranges_[r].from_));
  assert(t_offset < (1U << 16U));
  return (static_cast<std::uint32_t>(as_int(t.day_)) << 16U) | t_offset;
}

// egress and footpaths from round k's by-route arrivals
template <direction SearchDir, typename Criteria>
void basic_mcraptor<SearchDir, Criteria>::update_footpaths(unsigned const k) {
  auto const intermodal = is_intermodal_dest();
  state_.station_mark_.for_each_set_bit([&](std::uint64_t const i) {
    if (state_.bag_.empty(static_cast<std::uint32_t>(i))) {
      return;
    }
    auto const l = location_idx_t{i};
    // td footpaths replace the static ones of their source (as raptor)
    auto const has_td_fps = [&](location_idx_t const x, bool const out) {
      if (rtt_ == nullptr || prf_idx_ == 0U) {
        return false;
      }
      auto const& has = (out ? rtt_->has_td_footpaths_out_
                             : rtt_->has_td_footpaths_in_)[prf_idx_];
      return to_idx(x) < has.size() && has.test(x);
    };
    auto const use_td_fps = has_td_fps(l, kFwd);
    // static footpaths of static locations, rt footpaths (rt virts) of all
    auto const& static_fps = kFwd ? tt_.locations_.footpaths_out_[prf_idx_]
                                  : tt_.locations_.footpaths_in_[prf_idx_];
    auto const fps = i < static_fps.size()
                         ? std::span<footpath const>{static_fps[l].begin(),
                                                     static_fps[l].size()}
                         : std::span<footpath const>{};
    auto const rt_fps =
        rtt_ != nullptr && profile_uses_rt_virts(prf_idx_)
            ? std::span<footpath const>{(kFwd ? rtt_->rt_footpaths_out_
                                              : rtt_->rt_footpaths_in_)[l]
                                            .begin(),
                                        (kFwd ? rtt_->rt_footpaths_out_
                                              : rtt_->rt_footpaths_in_)[l]
                                            .size()}
            : std::span<footpath const>{};
    auto const egress_ok = intermodal && dist_to_end_[i] != kUnreachable;
    auto const td_egress = (intermodal && !td_dist_to_end_.empty())
                               ? td_dist_to_end_.find(l)
                               : end(td_dist_to_end_);
    auto const td_egress_ok = td_egress != end(td_dist_to_end_);
    if (!use_td_fps && fps.empty() && rt_fps.empty() && !egress_ok &&
        !td_egress_ok) {
      return;
    }

    // copy first (inserts go into the same bag), minus the transfer buffer
    auto const buf = transfer_buffer(i);
    fp_labels_.clear();
    for_each_label(static_cast<std::uint32_t>(i), k,
                   [&](typename state_t::label const& e) {
                     if (e.by_route_) {
                       fp_labels_.push_back(e);
                       fp_labels_.back().crit_ = e.crit_.with_transfer(-buf);
                     }
                   });
    if (fp_labels_.empty()) {
      return;
    }

    auto const relax_egress = [&](typename state_t::label const& te,
                                  std::uint16_t const duration,
                                  std::uint16_t const walk) {
      auto const end_crit = te.crit_.with_walk(dir(duration), walk);
      if (!is_better(end_crit.arr_, worst_at_dest_)) {
        return;
      }
      auto bc = state_.breadcrumbs_[te.breadcrumb_];
      bc.arr_ = end_crit.arr_;
      if (!merge_round(to_idx(kIntermodalTarget), end_crit, bc, k,
                       /*by_route=*/false)) {
        return;
      }
      dest_bag_add(k, end_crit);
    };

    // duration: incl. td waiting
    auto const relax_fp =
        [&](typename state_t::label const& te, std::uint32_t const target,
            std::uint16_t const duration, std::uint16_t const walk) {
          auto const fp_crit = te.crit_.with_walk(dir(duration), walk);
          if (!is_better(fp_crit.arr_, worst_at_dest_)) {
            return;
          }
          auto const lb = effective_lb(target);
          if (lb == kUnreachable ||
              !is_better(fp_crit.arr_ + dir(lb), worst_at_dest_) ||
              dest_dominates(
                  k, fp_crit.projected_to(clamp(fp_crit.arr_ + dir(lb)))) ||
              !guide_feasible(target, k, fp_crit.arr_, fp_crit.extras_) ||
              bound_prunes(k, target, fp_crit.arr_, transfer_buffer(target))) {
            ++stats_.fp_update_prevented_by_lower_bound_;
            return;
          }
          // the ride's breadcrumb: reconstruction derives the footpath
          auto bc = state_.breadcrumbs_[te.breadcrumb_];
          bc.arr_ = fp_crit.arr_;
          if (!merge_round(target, fp_crit, bc, k, /*by_route=*/false)) {
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
        relax_egress(te, dist_to_end_[i], dist_to_end_[i]);
      }
    }

    if (td_egress_ok) {
      for (auto const& te : fp_labels_) {
        auto const fp = get_td_duration<SearchDir>(td_egress->second,
                                                   to_unix(te.crit_.arr_));
        if (fp.has_value()) {
          relax_egress(
              te, static_cast<std::uint16_t>(fp->first.count()),
              static_cast<std::uint16_t>(fp->second.duration_.count()));
        }
      }
    }

    if (use_td_fps) {
      auto const& td_fps = kFwd ? rtt_->td_footpaths_out_[prf_idx_][l]
                                : rtt_->td_footpaths_in_[prf_idx_][l];
      for (auto const& te : fp_labels_) {
        for_each_footpath<SearchDir>(
            td_fps, to_unix(te.crit_.arr_),
            [&](location_idx_t const fp_target, duration_t const duration,
                duration_t const walk) {
              ++stats_.n_footpaths_visited_;
              auto const target = to_idx(fp_target);
              if (target == i) {
                return;
              }
              relax_fp(te, static_cast<std::uint32_t>(target),
                       static_cast<std::uint16_t>(duration.count()),
                       static_cast<std::uint16_t>(walk.count()));
            });
      }
    }
    auto const relax_fps = [&](std::span<footpath const> const bucket) {
      for (auto const& fp : bucket) {
        if (!kFwd && use_td_fps && has_td_fps(fp.target(), true)) {
          continue;
        }
        ++stats_.n_footpaths_visited_;
        auto const target = to_idx(fp.target());
        if (target == i) {
          continue;
        }
        auto const fp_duration = adjusted_transfer_time(transfer_time_settings_,
                                                        fp.duration().count());
        // between locations of one stop (virts) it is a transfer, no walk
        auto const walk = static_cast<std::uint16_t>(
            base_of(static_cast<std::uint32_t>(i)) ==
                    base_of(static_cast<std::uint32_t>(target))
                ? 0
                : fp_duration);
        for (auto const& te : fp_labels_) {
          relax_fp(te, static_cast<std::uint32_t>(target),
                   static_cast<std::uint16_t>(fp_duration), walk);
        }
      }
    };
    if (!kFwd || !use_td_fps) {
      relax_fps(fps);
    }
    relax_fps(rt_fps);
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

  auto const event_times = tt_.event_times_at_stop(r, stop_idx, kDepEv);

  auto const n = static_cast<unsigned>(event_times.size());
  auto const t_offset_at = [&](unsigned const pos) {
    return kFwd ? pos : n - 1U - pos;
  };

  auto const seek_first_day = [&]() {
    auto lo = 0U;
    auto hi = n;
    while (lo < hi) {
      auto const mid = lo + (hi - lo) / 2U;
      if (is_better(event_times[t_offset_at(mid)].mam(), mam_at_stop.count())) {
        lo = mid + 1U;
      } else {
        hi = mid;
      }
    }
    return lo;
  };

  auto const lb = dir(effective_lb(to_idx(l)));
  constexpr auto const kNDaysToIterate =
      kMaxTravelTime / std::chrono::days{1} + 1U;
  for (auto i = day_idx_t::value_t{0U}; i != kNDaysToIterate; ++i) {
    auto const day = kFwd ? day_at_stop + i : day_at_stop - i;

    if (!tt_.is_route_active(r, day)) {
      continue;
    }

    for (auto pos = i == 0U ? seek_first_day() : 0U; pos < n; ++pos) {
      auto const t_offset = t_offset_at(pos);
      auto const ev = event_times[t_offset];
      auto const ev_mam = ev.mam();

      if (is_better_or_eq(worst_at_dest_, to_delta(day, ev_mam) + lb)) {
        return {transport_idx_t::invalid(), day_idx_t::invalid()};
      }

      auto const t = event_times.transport(t_offset);
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

template <direction SearchDir, typename Criteria>
void basic_mcraptor<SearchDir, Criteria>::collect_dest_journeys(
    unsigned const k,
    unixtime_t const start_time,
    pareto_set<journey>& results) {
  is_dest_.for_each_set_bit([&](std::uint64_t const i) {
    for_each_label(static_cast<std::uint32_t>(i), k,
                   [&](typename state_t::label const& e) {
                     auto const& e_crit = e.crit_;
                     auto j = journey{};
                     j.start_time_ = start_time;
                     j.dest_time_ = to_unix(e_crit.arr_);
                     j.dest_ = location_idx_t{i};
                     j.transfers_ = static_cast<std::uint8_t>(k - 1U);
                     e_crit.apply_to(j);
                     if (!results.is_dominated(j)) {
                       chain_.clear();
                       for (auto li = static_cast<std::uint32_t>(e.breadcrumb_);
                            li != state_t::kNoBreadcrumb;
                            li = state_.breadcrumbs_[li].parent_) {
                         chain_.push_back(state_.breadcrumbs_[li]);
                       }
                       mc_add_legs<SearchDir>(tt_, rtt_, prf_idx_, base_,
                                              is_intermodal_dest(), chain_, j);
                       results.add(std::move(j));
                     }
                   });
  });
}

template <direction SearchDir, typename Criteria>
void basic_mcraptor<SearchDir, Criteria>::reconstruct(query const& q,
                                                      journey& j) {
  mc_reconstruct_offsets<SearchDir>(tt_, rtt_, q, j);
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
delta_t basic_mcraptor<SearchDir, Criteria>::rt_time_at_stop(
    rt_transport_idx_t const rt_t,
    stop_idx_t const stop_idx,
    event_type const ev_type) const {
  return to_delta(rtt_->base_day_idx_,
                  rtt_->event_time(rt_t, stop_idx, ev_type));
}

template <direction SearchDir, typename Criteria>
bool basic_mcraptor<SearchDir, Criteria>::is_transport_active(
    transport_idx_t const t, day_idx_t const day) const {
  return rtt_ != nullptr ? rtt_->is_transport_active(t, day)
                         : tt_.is_transport_active(t, day);
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

template <direction SearchDir, typename Criteria>
void basic_mcraptor<SearchDir, Criteria>::take_guide() {
  auto& g = state_.guide_[kDirIdx];
  for (auto const l : g.stops_) {
    g.n_[l] = 0U;
  }
  g.stops_.clear();
  g.labels_.clear();
  g.begin_.resize(state_.bag_.bags_.size());
  g.n_.resize(state_.bag_.bags_.size(), 0U);
  state_.bag_.touched_.for_each_set_bit([&](auto const i) {
    auto const l = static_cast<std::uint32_t>(i);
    auto const bag = state_.bag_.span(l);
    if (bag.empty()) {
      return;
    }
    g.begin_[l] = static_cast<std::uint32_t>(g.labels_.size());
    g.n_[l] = static_cast<std::uint32_t>(bag.size());
    g.stops_.push_back(l);
    for (auto const& e : bag) {
      g.labels_.push_back({e.crit_.arr_, e.round_, e.crit_.extras_});
    }
  });
}

template <direction SearchDir, typename Criteria>
void basic_mcraptor<SearchDir, Criteria>::set_guide(
    std::vector<journey const*> const& targets) {
  guide_targets_.clear();
  for (auto const* j : targets) {
    guide_targets_.push_back(
        {.trips_ = static_cast<std::uint8_t>(j->transfers_ + 1U),
         .extras_ =
             static_cast<std::uint16_t>(Criteria::extras_t::of_journey(*j))});
  }
}

template <direction SearchDir, typename Criteria>
bool basic_mcraptor<SearchDir, Criteria>::guide_feasible(
    std::uint32_t const l,
    unsigned const k,
    delta_t const t,
    std::uint32_t const extras) const {
  if (guide_targets_.empty()) {
    return true;
  }
  auto const& g = state_.guide_[kDirIdx ^ 1U];
  auto const labels =
      l < g.n_.size() && g.n_[l] != 0U
          ? std::span<mc_guide_entry const>{g.labels_.data() + g.begin_[l],
                                            g.n_[l]}
          : std::span<mc_guide_entry const>{};
  return routing::guide_feasible<SearchDir, typename Criteria::extras_t>(
      labels, guide_targets_, k, t, extras,
      adjusted_transfer_time(
          transfer_time_settings_,
          get_transfer_time(tt_, rtt_, location_idx_t{l}).count()));
}

#define NIGIRI_MC_INSTANTIATE(C)                          \
  template struct basic_mcraptor<direction::kForward, C>; \
  template struct basic_mcraptor<direction::kBackward, C>;

NIGIRI_MC_INSTANTIATE(arr_criteria)
NIGIRI_MC_INSTANTIATE(arr_non_transit_criteria)
NIGIRI_MC_INSTANTIATE(arr_avoided_modes_criteria)
NIGIRI_MC_INSTANTIATE(arr_mode_switches_criteria)
NIGIRI_MC_INSTANTIATE(arr_non_transit_avoided_modes_criteria)
NIGIRI_MC_INSTANTIATE(arr_non_transit_mode_switches_criteria)
NIGIRI_MC_INSTANTIATE(arr_avoided_modes_mode_switches_criteria)
NIGIRI_MC_INSTANTIATE(arr_non_transit_avoided_modes_mode_switches_criteria)
NIGIRI_MC_INSTANTIATE(arr_night_transfer_criteria)
NIGIRI_MC_INSTANTIATE(arr_non_transit_night_transfer_criteria)
NIGIRI_MC_INSTANTIATE(arr_avoided_modes_night_transfer_criteria)
NIGIRI_MC_INSTANTIATE(arr_mode_switches_night_transfer_criteria)
NIGIRI_MC_INSTANTIATE(arr_non_transit_avoided_modes_night_transfer_criteria)
NIGIRI_MC_INSTANTIATE(arr_non_transit_mode_switches_night_transfer_criteria)
NIGIRI_MC_INSTANTIATE(arr_avoided_modes_mode_switches_night_transfer_criteria)
NIGIRI_MC_INSTANTIATE(
    arr_non_transit_avoided_modes_mode_switches_night_transfer_criteria)

#undef NIGIRI_MC_INSTANTIATE

}  // namespace nigiri::routing
