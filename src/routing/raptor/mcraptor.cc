#include "nigiri/routing/raptor/mcraptor.h"

#include <cassert>
#include <algorithm>

#include "utl/helpers/algorithm.h"
#include "utl/verify.h"

#include "nigiri/for_each_meta.h"
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

template <direction SearchDir>
void mc_reconstruct_offsets(timetable const& tt,
                            rt_timetable const* rtt,
                            query const& q,
                            journey& j) {
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
      return matches(tt, front_match_mode, x.target(), from) &&
             front_ok(x.duration());
    });
    auto front = std::optional<offset>{};
    auto front_dep = dep_time;  // set with the offset, see the td case
    if (o != end(offsets)) {
      front = *o;
      front_dep = dep_time - o->duration();
    } else if (auto const& td = is_fwd ? q.td_start_ : q.td_dest_;
               td.contains(from)) {
      // latest departure making the boarding, then strip the wait
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
    j.legs_.insert(begin(j.legs_),
                   journey::leg{direction::kForward, special, from, front_dep,
                                front_dep + front->duration(), *front});
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
      return matches(tt, back_match_mode, x.target(), to) &&
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
    j.legs_.push_back(journey::leg{direction::kForward, to, special, back_dep,
                                   back_dep + back->duration(), *back});
    j.dest_ = special;
  }

  // the start footpath that seeded round 0
  if (q.start_match_mode_ != location_match_mode::kIntermodal) {
    auto const is_journey_start = [&](location_idx_t const l) {
      return utl::any_of(q.start_, [&](offset const& o) {
        return matches(tt, q.start_match_mode_, o.target(), l);
      });
    };
    auto const start_l = is_fwd ? j.legs_.front().from_ : j.legs_.back().to_;
    auto const start_t =
        is_fwd ? j.legs_.front().dep_time_ : j.legs_.back().arr_time_;
    auto const direct_start_ok =
        is_fwd ? j.start_time_ <= start_t : j.start_time_ >= start_t;
    if (!is_journey_start(start_l) || !direct_start_ok) {
      auto const fps = is_fwd
                           ? tt.locations_.footpaths_in_[q.prf_idx_][start_l]
                           : tt.locations_.footpaths_out_[q.prf_idx_][start_l];
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

  // td footpath legs: walking time only, the wait is spent at the stop
  if (rtt != nullptr && q.prf_idx_ != 0U) {
    auto const& has_td = is_fwd ? rtt->has_td_footpaths_in_[q.prf_idx_]
                                : rtt->has_td_footpaths_out_[q.prf_idx_];
    auto const& td_fps = is_fwd ? rtt->td_footpaths_in_[q.prf_idx_]
                                : rtt->td_footpaths_out_[q.prf_idx_];
    if (!td_fps.empty()) {
      for (auto& lg : j.legs_) {
        if (!std::holds_alternative<footpath>(lg.uses_)) {
          continue;
        }
        auto const key_l = is_fwd ? lg.to_ : lg.from_;
        auto const target_l = is_fwd ? lg.from_ : lg.to_;
        if (!has_td.test(key_l)) {
          continue;
        }
        auto const t = lg.arr_time_;
        for_each_footpath<SearchDir>(td_fps[key_l], t,
                                     [&](location_idx_t const fp_target,
                                         duration_t, duration_t const walk) {
                                       if (fp_target != target_l) {
                                         return utl::cflow::kContinue;
                                       }
                                       lg.dep_time_ = t - walk;
                                       lg.arr_time_ = t;
                                       lg.uses_ = footpath{lg.to_, walk};
                                       return utl::cflow::kBreak;
                                     });
      }
    }
  }
}

template void mc_reconstruct_offsets<direction::kForward>(timetable const&,
                                                          rt_timetable const*,
                                                          query const&,
                                                          journey&);
template void mc_reconstruct_offsets<direction::kBackward>(timetable const&,
                                                           rt_timetable const*,
                                                           query const&,
                                                           journey&);

template <direction SearchDir, typename Criteria, bool RangeReuse>
basic_mcraptor<SearchDir, Criteria, RangeReuse>::basic_mcraptor(
    timetable const& tt,
    rt_timetable const* rtt,
    state_t& state,
    bitvec& is_dest,
    std::array<bitvec, kMaxVias>&,
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
      state_{state.resize(tt.n_locations(),
                          tt.n_routes(),
                          rtt != nullptr ? rtt->n_rt_transports() : 0U)},
      is_dest_{is_dest},
      dist_to_end_{dist_to_dest},
      td_dist_to_end_{td_dist_to_dest},
      lb_{lb},
      base_{base},
      allowed_claszes_{allowed_claszes},
      is_wheelchair_{is_wheelchair},
      transfer_time_settings_{tts},
      prf_idx_{prf_idx} {
  utl::verify(via_stops.empty(), "mcraptor: via stops not supported");
  utl::verify(!require_bike_transport && !require_car_transport &&
                  !no_compulsory_reservation,
              "mcraptor: bike/car transport, reservation filter not supported");
  reset_arrivals();
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

template <direction SearchDir, typename Criteria, bool RangeReuse>
template <typename Fn>
void basic_mcraptor<SearchDir, Criteria, RangeReuse>::for_each_label(
    std::uint32_t const l, unsigned const round, Fn&& fn) const {
  for (auto const& e : state_.bag_.span(l)) {
    if (e.round_ == round && e.dep_ == cur_dep_) {
      fn(e.crit_, e.breadcrumb_);
    }
  }
}

// Pareto insert; true iff inserted. A candidate is rejected only by labels of
// no higher round (by-route ones only by by-route labels: they still relax
// their footpaths) and evicts only labels of no lower round (by-transfer ones
// only by-transfer labels), across start times.
template <direction SearchDir, typename Criteria, bool RangeReuse>
bool basic_mcraptor<SearchDir, Criteria, RangeReuse>::bag_insert(
    std::uint32_t const l,
    Criteria const& crit,
    std::uint32_t const flagged_breadcrumb,
    unsigned const round,
    bool const by_route) {
  auto& layer = state_.bag_;
  if (layer.empty(l)) {
    layer.touched_.set(l);
  }
  auto const bag = layer.span(l);
  for (auto const& e : bag) {
    if (e.round_ <= round &&
        (!by_route || (e.breadcrumb_ & state_t::kByRoute) != 0U) &&
        e.crit_.template dominates<SearchDir>(crit)) {
      return false;
    }
  }
  auto removed = std::size_t{0U};
  for (auto i = std::size_t{0U}; i != bag.size(); ++i) {
    if (bag[i].round_ >= round &&
        (by_route || (bag[i].breadcrumb_ & state_t::kByRoute) == 0U) &&
        crit.template dominates<SearchDir>(bag[i].crit_)) {
      ++removed;
      continue;
    }
    bag[i - removed] = bag[i];
  }
  layer.set_size(l, static_cast<std::uint32_t>(bag.size() - removed));
  layer.push_back(l, {crit, flagged_breadcrumb,
                      static_cast<std::uint8_t>(round), cur_dep_});
  return true;
}

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
    mc_breadcrumb const& bc,
    unsigned const round,
    bool const by_route) {
  auto const idx = static_cast<std::uint32_t>(state_.breadcrumbs_.size());
  if (!bag_insert(l, crit, by_route ? idx | state_t::kByRoute : idx, round,
                  by_route)) {
    return false;
  }
  state_.breadcrumbs_.push_back(bc);
  return true;
}

template <direction SearchDir, typename Criteria, bool RangeReuse>
void basic_mcraptor<SearchDir, Criteria, RangeReuse>::add_start(
    location_idx_t const l, unixtime_t const t, via_offset_t) {
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

  worst_at_dest_ =
      get_best(unix_to_delta(base(), worst_time_at_dest), worst_at_dest_);

  // Round-0 labels.
  auto const d_start = unix_to_delta(base(), start_time);
  cur_dep_ = d_start;
  for (auto const& [l, arr] : seeds_) {
    auto const walk =
        seed_walk<SearchDir>(td_start_, location_idx_t{l}, start_time,
                             static_cast<std::uint16_t>(dir(arr - d_start)));
    bag_insert(l, Criteria::at_start(arr, walk), state_t::kNoBreadcrumb,
               /*round=*/0U, /*by_route=*/false);
  }
  seeds_.clear();

  for (auto k = 1U; k != end_k; ++k) {
    auto any_marked = false;
    state_.station_mark_.for_each_set_bit([&](std::uint64_t const i) {
      for (auto const& r : tt_.location_routes_[location_idx_t{i}]) {
        any_marked = true;
        state_.route_mark_.set(to_idx(r), true);
      }
      if (rtt_ != nullptr) {
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

    // one fused pass for intermodal egress and footpaths
    update_footpaths(k);

    collect_dest_journeys(k, start_time, results);
  }
}

// Alights rl at l_idx; true iff the arrival (plus transfer buffer) was
// inserted.
template <direction SearchDir, typename Criteria, bool RangeReuse>
template <typename Sections>
bool basic_mcraptor<SearchDir, Criteria, RangeReuse>::alight(
    unsigned const k,
    std::uint32_t const l_idx,
    stop_idx_t const stop_idx,
    delta_t const by_transport,
    route_label const& rl,
    Sections const& sections,
    clasz const fallback,
    std::uint32_t const transport,
    day_idx_t const day) {
  auto const lb = lb_[l_idx];
  if (!is_better(by_transport, worst_at_dest_) || lb == kUnreachable ||
      !is_better(by_transport + dir(lb), worst_at_dest_)) {
    ++stats_.route_update_prevented_by_lower_bound_;
    return false;
  }
  // clasz of the alighted section (route_clasz_ is the first section's)
  auto const ride_crit = Criteria::from_ride(
      by_transport,
      ride_clasz(sections, fallback,
                 static_cast<unsigned>(kFwd ? stop_idx - 1 : stop_idx)),
      rl.carried_);
  if (dest_dominates(k,
                     ride_crit.projected_to(clamp(by_transport + dir(lb))))) {
    ++stats_.route_update_prevented_by_lower_bound_;
    return false;
  }
  // Post-buffer dominance equals pre-buffer (the buffer is per-stop constant).
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
  // the buffer is 0 at a station destination: ride_crit is the arrival
  if (is_dest_[l_idx]) {
    dest_bag_add(k, ride_crit);
  }
  return true;
}

template <direction SearchDir, typename Criteria, bool RangeReuse>
bool basic_mcraptor<SearchDir, Criteria, RangeReuse>::loop_routes(
    unsigned const k) {
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

// One trip sequence's stops: alight route_bag_, then board the last round's
// labels (board_label decides on the trip, board() updates the bag). A ride
// into a stop that is not ridable (a section filter) drops the boarded trips.
template <direction SearchDir, typename Criteria, bool RangeReuse>
template <typename Stops, typename Ridable, typename Alight, typename Board>
bool basic_mcraptor<SearchDir, Criteria, RangeReuse>::scan_stops(
    unsigned const k,
    Stops const& stop_seq,
    Ridable&& ridable,
    Alight&& alight_label,
    Board&& board_label) {
  auto const n = static_cast<unsigned>(stop_seq.size());
  auto any_marked = false;
  route_bag_.clear();
  for (auto i = 0U; i != n; ++i) {
    auto const stop_idx = static_cast<stop_idx_t>(kFwd ? i : n - i - 1U);
    auto const stp = stop{stop_seq[stop_idx]};
    auto const l_idx = static_cast<std::uint32_t>(to_idx(stp.location_idx()));

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

    if (lb_[l_idx] == kUnreachable) {
      break;
    }

    for_each_label(
        l_idx, k - 1U,
        [&](Criteria const& crit, std::uint32_t const breadcrumb) {
          // even the optimistic completion is dominated
          if (!dest_dominates(
                  k, crit.projected_to(clamp(crit.arr_ + dir(lb_[l_idx]))))) {
            board_label(crit, breadcrumb & state_t::kBreadcrumbMask, stop_idx,
                        stp.location_idx());
          }
        });
  }
  return any_marked;
}

// pareto over (trip order, carried criteria)
template <direction SearchDir, typename Criteria, bool RangeReuse>
void basic_mcraptor<SearchDir, Criteria, RangeReuse>::board(
    std::uint32_t const key,
    transport const t,
    stop_idx_t const stop_idx,
    std::uint32_t const parent,
    typename Criteria::carried const& carried) {
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

template <direction SearchDir, typename Criteria, bool RangeReuse>
bool basic_mcraptor<SearchDir, Criteria, RangeReuse>::update_route(
    unsigned const k, route_idx_t const r) {
  // wheelchair: accessible on all sections, or check the section ridden into
  auto const all_sections =
      !is_wheelchair_ ||
      tt_.route_flags_[route_flag::kWheelchairAccessible].test(to_idx(r) * 2U);
  return scan_stops(
      k, tt_.route_location_seq_[r],
      [&](stop_idx_t const stop_idx) {
        return all_sections ||
               tt_.route_flags_per_section_[route_flag::kWheelchairAccessible]
                                           [r][kFwd ? stop_idx - 1 : stop_idx];
      },
      [&](route_label const& rl, std::uint32_t const l_idx,
          stop_idx_t const stop_idx) {
        return alight(k, l_idx, stop_idx,
                      time_at_stop(r, rl.t_, stop_idx, kArrEv), rl,
                      tt_.route_section_clasz_[r], tt_.route_clasz_[r],
                      to_idx(rl.t_.t_idx_), rl.t_.day_);
      },
      [&](Criteria const& crit, std::uint32_t const parent,
          stop_idx_t const stop_idx, location_idx_t const l) {
        // skip the lookup if a boarded trip already dominates
        for (auto const& rl : route_bag_) {
          if (Criteria::carried_dominates(rl.carried_, crit.d_) &&
              is_better(time_at_stop(r, rl.t_, stop_idx, kDepEv), crit.arr_)) {
            return;
          }
        }
        auto const [day, mam] = split(crit.arr_);
        auto const et = get_earliest_transport(r, stop_idx, day, mam, l);
        if (et.is_valid()) {
          board(trip_order_key(r, et), et, stop_idx, parent, crit.d_);
        }
      });
}

template <direction SearchDir, typename Criteria, bool RangeReuse>
bool basic_mcraptor<SearchDir, Criteria, RangeReuse>::loop_rt_transports(
    unsigned const k) {
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

// update_route for an rt run: a single trip with absolute event times.
template <direction SearchDir, typename Criteria, bool RangeReuse>
bool basic_mcraptor<SearchDir, Criteria, RangeReuse>::update_rt_transport(
    unsigned const k, rt_transport_idx_t const rt_t) {
  auto const all_sections =
      !is_wheelchair_ ||
      rtt_->rt_transport_flags_[route_flag::kWheelchairAccessible].test(
          to_idx(rt_t) * 2U);
  return scan_stops(
      k, rtt_->rt_transport_location_seq_[rt_t],
      [&](stop_idx_t const stop_idx) {
        return all_sections ||
               rtt_->rt_flags_per_section_[route_flag::kWheelchairAccessible]
                                          [rt_t]
                                          [kFwd ? stop_idx - 1 : stop_idx];
      },
      [&](route_label const& rl, std::uint32_t const l_idx,
          stop_idx_t const stop_idx) {
        return alight(k, l_idx, stop_idx,
                      rt_time_at_stop(rt_t, stop_idx, kArrEv), rl,
                      rtt_->rt_transport_section_clasz_[rt_t], clasz::kOther,
                      to_idx(rt_t), day_idx_t::invalid());
      },
      [&](Criteria const& crit, std::uint32_t const parent,
          stop_idx_t const stop_idx, location_idx_t) {
        if (is_better_or_eq(crit.arr_,
                            rt_time_at_stop(rt_t, stop_idx, kDepEv))) {
          board(0U, transport{}, stop_idx, parent, crit.d_);
        }
      });
}

// traffic day << 16 | trip offset: orders the trips of a route
template <direction SearchDir, typename Criteria, bool RangeReuse>
std::uint32_t basic_mcraptor<SearchDir, Criteria, RangeReuse>::trip_order_key(
    route_idx_t const r, transport const t) const {
  auto const t_offset = static_cast<std::uint32_t>(
      to_idx(t.t_idx_) - to_idx(tt_.route_transport_ranges_[r].from_));
  assert(t_offset < (1U << 16U));
  return (static_cast<std::uint32_t>(as_int(t.day_)) << 16U) | t_offset;
}

// Egress and footpaths from round k's by-route arrivals.
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
    auto const use_td_fps = rtt_ != nullptr && prf_idx_ != 0U &&
                            (kFwd ? rtt_->has_td_footpaths_out_
                                  : rtt_->has_td_footpaths_in_)[prf_idx_]
                                .test(l);
    auto const& fps = kFwd ? tt_.locations_.footpaths_out_[prf_idx_][l]
                           : tt_.locations_.footpaths_in_[prf_idx_][l];
    auto const egress_ok = intermodal && dist_to_end_[i] != kUnreachable;
    // empty for non-td queries: keep the lookup off the hot path
    auto const td_egress = (intermodal && !td_dist_to_end_.empty())
                               ? td_dist_to_end_.find(l)
                               : end(td_dist_to_end_);
    auto const td_egress_ok = td_egress != end(td_dist_to_end_);
    if (!use_td_fps && fps.empty() && !egress_ok && !td_egress_ok) {
      return;
    }

    // copy first (inserts go into the same bag), without transfer buffer
    auto const buf = transfer_buffer(i);
    fp_labels_.clear();
    for_each_label(
        static_cast<std::uint32_t>(i), k,
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
                                  std::uint16_t const duration,
                                  std::uint16_t const walk) {
      auto const end_crit = te.crit_.with_walk(dir(duration), walk);
      // window bound
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

    // duration: adjusted static footpath or td duration incl. waiting
    auto const relax_fp =
        [&](typename state_t::label const& te, std::uint32_t const target,
            std::uint16_t const duration, std::uint16_t const walk) {
          auto const fp_crit = te.crit_.with_walk(dir(duration), walk);
          if (!is_better(fp_crit.arr_, worst_at_dest_)) {
            return;
          }
          auto const lb = lb_[target];
          if (lb == kUnreachable ||
              !is_better(fp_crit.arr_ + dir(lb), worst_at_dest_) ||
              dest_dominates(
                  k, fp_crit.projected_to(clamp(fp_crit.arr_ + dir(lb))))) {
            ++stats_.fp_update_prevented_by_lower_bound_;
            return;
          }
          // keeps the ride's payload; reconstruction derives the footpath
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

    // td durations depend on the arrival, so they are evaluated per label
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
                   static_cast<std::uint16_t>(fp_duration),
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

  auto const lb = dir(lb_[to_idx(l)]);
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

template <direction SearchDir, typename Criteria, bool RangeReuse>
void basic_mcraptor<SearchDir, Criteria, RangeReuse>::collect_dest_journeys(
    unsigned const k,
    unixtime_t const start_time,
    pareto_set<journey>& results) {
  is_dest_.for_each_set_bit([&](std::uint64_t const i) {
    for_each_label(
        static_cast<std::uint32_t>(i), k,
        [&](Criteria const& e_crit, std::uint32_t const e_breadcrumb) {
          auto j = journey{};
          j.start_time_ = start_time;
          j.dest_time_ = to_unix(e_crit.arr_);
          j.dest_ = location_idx_t{i};
          j.transfers_ = static_cast<std::uint8_t>(k - 1U);
          e_crit.apply_to(j);
          // most candidates are dominated: only the others get legs
          if (!results.is_dominated(j)) {
            add_legs(j, e_breadcrumb & state_t::kBreadcrumbMask);
            results.add(std::move(j));
          }
        });
  });
}

// Chases the breadcrumbs back to the start.
template <direction SearchDir, typename Criteria, bool RangeReuse>
void basic_mcraptor<SearchDir, Criteria, RangeReuse>::add_legs(
    journey& j, std::uint32_t const breadcrumb_idx) {
  auto cur_l = j.dest_;
  auto li = breadcrumb_idx;
  while (li != state_t::kNoBreadcrumb) {
    auto const& bc = state_.breadcrumbs_[li];
    auto const cur_arr = bc.arr_;  // arrival at cur_l

    auto train_arr = kInvalid;
    auto dep_at_board = kInvalid;
    auto board_loc = location_idx_t::invalid();
    auto alight_loc = location_idx_t::invalid();
    auto run = rt::run{};
    auto const set_stops = [&](auto const stop_seq) {
      board_loc = stop{stop_seq[bc.board_]}.location_idx();
      alight_loc = stop{stop_seq[bc.alight_]}.location_idx();
      run.stop_range_ = interval<stop_idx_t>{
          stop_idx_t{0U}, static_cast<stop_idx_t>(stop_seq.size())};
    };
    if (bc.day_ == day_idx_t::invalid()) {
      auto const rt_t = rt_transport_idx_t{bc.transport_};
      set_stops(rtt_->rt_transport_location_seq_[rt_t]);
      train_arr = rt_time_at_stop(rt_t, bc.alight_, kArrEv);
      dep_at_board = rt_time_at_stop(rt_t, bc.board_, kDepEv);
      run.t_ = rtt_->resolve_static(rt_t);
      run.rt_ = rt_t;
    } else {
      auto const t = transport{transport_idx_t{bc.transport_}, bc.day_};
      auto const r = tt_.transport_route_[t.t_idx_];
      set_stops(tt_.route_location_seq_[r]);
      train_arr = time_at_stop(r, t, bc.alight_, kArrEv);
      dep_at_board = time_at_stop(r, t, bc.board_, kDepEv);
      run.t_ = t;
    }

    if (is_intermodal_dest() && cur_l == kIntermodalTarget) {
      // the last mile is added by reconstruct()
      j.dest_ = alight_loc;
    } else if (!j.legs_.empty() || alight_loc != cur_l ||
               train_arr != cur_arr) {
      // footpaths are always emitted, even zero-minute transfers
      j.legs_.emplace_back(journey::leg{
          SearchDir, alight_loc, cur_l, to_unix(train_arr), to_unix(cur_arr),
          footpath{cur_l, duration_t{static_cast<duration_t::rep>(
                              static_cast<std::uint16_t>(
                                  dir(cur_arr - train_arr)))}}});
    }
    j.legs_.emplace_back(
        journey::leg{SearchDir, board_loc, alight_loc, to_unix(dep_at_board),
                     to_unix(train_arr),
                     journey::run_enter_exit{run, bc.board_, bc.alight_}});

    cur_l = board_loc;
    li = bc.parent_;
  }

  if constexpr (kFwd) {
    std::reverse(begin(j.legs_), end(j.legs_));
  }

  // a backward search re-anchors footpaths to the previous trip's arrival
  for (auto z = std::size_t{1U}; z < j.legs_.size(); ++z) {
    if (std::holds_alternative<footpath>(j.legs_[z].uses_)) {
      auto const dur = std::get<footpath>(j.legs_[z].uses_).duration();
      j.legs_[z].dep_time_ = j.legs_[z - 1U].arr_time_;
      j.legs_[z].arr_time_ = j.legs_[z].dep_time_ + dur;
    }
  }
}

template <direction SearchDir, typename Criteria, bool RangeReuse>
void basic_mcraptor<SearchDir, Criteria, RangeReuse>::reconstruct(
    query const& q, journey& j) {
  mc_reconstruct_offsets<SearchDir>(tt_, rtt_, q, j);
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
  return rtt_ != nullptr ? rtt_->is_transport_active(t, day)
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
NIGIRI_MC_INSTANTIATE(arr_non_transit_criteria)
NIGIRI_MC_INSTANTIATE(arr_mode_filter_criteria)
NIGIRI_MC_INSTANTIATE(arr_mode_switches_criteria)
NIGIRI_MC_INSTANTIATE(arr_non_transit_mode_filter_criteria)
NIGIRI_MC_INSTANTIATE(arr_non_transit_mode_switches_criteria)
NIGIRI_MC_INSTANTIATE(arr_mode_filter_mode_switches_criteria)
NIGIRI_MC_INSTANTIATE(arr_non_transit_mode_filter_mode_switches_criteria)

#undef NIGIRI_MC_INSTANTIATE

}  // namespace nigiri::routing
