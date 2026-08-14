#include "nigiri/routing/raptor/reconstruct.h"

#include <algorithm>

#include "utl/overloaded.h"

#include "nigiri/for_each_meta.h"
#include "nigiri/routing/journey.h"
#include "nigiri/rt/frun.h"
#include "nigiri/rt/rt_timetable.h"
#include "nigiri/timetable.h"
#include "nigiri/types.h"

namespace nigiri::routing {

void optimize_initial_departure(timetable const& tt,
                                rt_timetable const* rtt,
                                query const& q,
                                journey& j) {
  if (j.legs_.size() <= 1 || !holds_alternative<offset>(j.legs_[0].uses_) ||
      !holds_alternative<journey::run_enter_exit>(j.legs_[1].uses_)) {
    return;
  }

  auto& offset_leg = j.legs_[0];
  auto& transport_leg = j.legs_[1];
  auto& ree = get<journey::run_enter_exit>(transport_leg.uses_);
  auto offset_dur_best = get<offset>(offset_leg.uses_).duration();

  auto const& offsets = q.start_;

  auto r = rt::run{ree.r_};
  r.stop_range_ = {0U, static_cast<stop_idx_t>(ree.stop_range_.to_ - 1U)};
  for (auto const stp : rt::frun{tt, rtt, r}) {
    if (!q.via_stops_.empty() &&
        matches(tt, location_match_mode::kEquivalent, stp.get_location_idx(),
                q.via_stops_.front().location_)) {
      // don't skip over via stops
      break;
    }
    if (!stp.in_allowed()) {
      continue;
    }
    for (auto const& o : offsets) {
      if (offset_dur_best <= o.duration() ||
          !matches(tt, location_match_mode::kExact, o.target(),
                   stp.get_location_idx())) {
        continue;
      }

      auto const dep = stp.time(event_type::kDep);
      auto const o_start = dep - o.duration();
      if (offset_leg.dep_time_ <= o_start) {
        offset_leg.to_ = stp.get_location_idx();
        offset_leg.dep_time_ = o_start;
        offset_leg.arr_time_ = dep;
        offset_leg.uses_ = o;
        transport_leg.from_ = stp.get_location_idx();
        transport_leg.dep_time_ = dep;
        ree.stop_range_.from_ = stp.stop_idx_;
        offset_dur_best = o.duration();
      }
    }
  }
}

void optimize_initial_start_footpath(timetable const& tt,
                                     rt_timetable const* rtt,
                                     query const& q,
                                     journey& j) {
  if (j.legs_.size() <= 1 ||
      !holds_alternative<footpath>(j.legs_.front().uses_) ||
      !holds_alternative<journey::run_enter_exit>(j.legs_[1].uses_)) {
    return;
  }

  auto& fp_leg = j.legs_.front();
  auto& transport_leg = j.legs_[1];
  auto& ree = get<journey::run_enter_exit>(transport_leg.uses_);

  auto const start_loc = fp_leg.from_;

  if (rtt != nullptr && q.prf_idx_ != 0U &&
      rtt->has_td_footpaths_out_[q.prf_idx_].test(start_loc)) {
    return;  // TODO(felix) handle shortening of td-footpaths
  }

  auto fp_dur_best = adjusted_transfer_time(
      q.transfer_time_settings_, get<footpath>(fp_leg.uses_).duration());

  auto const& footpaths = tt.locations_.footpaths_out_[q.prf_idx_][start_loc];

  auto r = rt::run{ree.r_};
  r.stop_range_ = {0U, static_cast<stop_idx_t>(ree.stop_range_.to_ - 1U)};
  for (auto const stp : rt::frun{tt, rtt, r}) {
    if (!q.via_stops_.empty() &&
        matches(tt, location_match_mode::kEquivalent, stp.get_location_idx(),
                q.via_stops_[0].location_)) {
      break;  // don't skip over via stops
    }

    if (!stp.in_allowed()) {
      continue;
    }

    for (auto const& fp : footpaths) {
      if (fp.target() != stp.get_location_idx()) {
        continue;
      }
      auto const fp_dur =
          adjusted_transfer_time(q.transfer_time_settings_, fp.duration());
      if (fp_dur < fp_dur_best) {
        auto const dep = stp.time(event_type::kDep);
        auto const fp_start = dep - fp_dur;
        if (fp_leg.dep_time_ <= fp_start) {
          fp_leg.to_ = stp.get_location_idx();
          fp_leg.dep_time_ = fp_start;
          fp_leg.arr_time_ = dep;
          fp_leg.uses_ = footpath{stp.get_location_idx(), fp_dur};
          transport_leg.from_ = stp.get_location_idx();
          transport_leg.dep_time_ = dep;
          ree.stop_range_.from_ = stp.stop_idx_;
          fp_dur_best = fp_dur;
        }
      }
      break;
    }
  }
}

void optimize_last_arrival(timetable const& tt,
                           rt_timetable const* rtt,
                           query const& q,
                           journey& j) {

  if (j.legs_.size() <= 1 || !holds_alternative<offset>(j.legs_.back().uses_) ||
      !holds_alternative<journey::run_enter_exit>(rbegin(j.legs_)[1].uses_)) {
    return;
  }

  auto& offset_leg = j.legs_.back();
  auto& transport_leg = rbegin(j.legs_)[1];
  auto& ree = get<journey::run_enter_exit>(transport_leg.uses_);
  auto offset_dur_best = get<offset>(offset_leg.uses_).duration();

  auto const* offsets = &q.destination_;

  auto fr = rt::frun{tt, rtt, ree.r_};
  auto range_from = static_cast<stop_idx_t>(ree.stop_range_.from_ + 1U);

  if (!q.via_stops_.empty()) {
    // don't skip the last via stop
    for (auto i = stop_idx_t{0U}; i < fr.size(); ++i) {
      auto idx = static_cast<stop_idx_t>(fr.size() - i - 1U);
      if (matches(tt, location_match_mode::kEquivalent,
                  fr[idx].get_location_idx(), q.via_stops_.back().location_)) {
        range_from = std::max(range_from, idx);
        break;
      }
    }
  }

  fr.stop_range_ = {range_from, fr.size()};

  for (auto const stp : fr) {
    if (!stp.out_allowed()) {
      continue;
    }
    for (auto const& o : *offsets) {
      if (offset_dur_best <= o.duration() ||
          !matches(tt, location_match_mode::kExact, o.target(),
                   stp.get_location_idx())) {
        continue;
      }
      auto const arr = stp.time(event_type::kArr);
      auto const o_end = arr + o.duration();
      if (o_end <= offset_leg.arr_time_) {
        offset_leg.from_ = stp.get_location_idx();
        offset_leg.dep_time_ = arr;
        offset_leg.arr_time_ = o_end;
        offset_leg.uses_ = o;
        transport_leg.to_ = stp.get_location_idx();
        transport_leg.arr_time_ = arr;
        ree.stop_range_.to_ = stp.stop_idx_ + 1U;
        offset_dur_best = o.duration();
      }
    }
  }
}

void optimize_final_egress_footpath(timetable const& tt,
                                    rt_timetable const* rtt,
                                    query const& q,
                                    journey& j) {
  if (j.legs_.size() <= 1 ||
      !holds_alternative<footpath>(j.legs_.back().uses_) ||
      !holds_alternative<journey::run_enter_exit>(rbegin(j.legs_)[1].uses_)) {
    return;
  }

  auto& fp_leg = j.legs_.back();
  auto& transport_leg = rbegin(j.legs_)[1];
  auto& ree = get<journey::run_enter_exit>(transport_leg.uses_);

  auto const dest_loc = fp_leg.to_;

  if (rtt != nullptr && q.prf_idx_ != 0U &&
      rtt->has_td_footpaths_in_[q.prf_idx_].test(dest_loc)) {
    return;  // TODO(felix) handle shortening of td-footpaths
  }

  auto fp_dur_best = adjusted_transfer_time(
      q.transfer_time_settings_, get<footpath>(fp_leg.uses_).duration());

  auto const& footpaths = tt.locations_.footpaths_in_[q.prf_idx_][dest_loc];

  auto fr = rt::frun{tt, rtt, ree.r_};
  auto range_from = static_cast<stop_idx_t>(ree.stop_range_.from_ + 1U);

  if (!q.via_stops_.empty()) {
    // don't alight before the last via stop
    auto const& last_via = q.via_stops_.back();
    for (auto i = stop_idx_t{0U}; i < fr.size(); ++i) {
      auto idx = static_cast<stop_idx_t>(fr.size() - i - 1U);
      if (matches(tt, location_match_mode::kEquivalent,
                  fr[idx].get_location_idx(), last_via.location_)) {
        range_from = std::max(range_from, idx);
        break;
      }
    }
  }

  fr.stop_range_ = {range_from, fr.size()};

  for (auto const stp : fr) {
    if (!stp.out_allowed()) {
      continue;
    }
    for (auto const& fp : footpaths) {
      if (fp.target() != stp.get_location_idx()) {
        continue;
      }
      auto const fp_dur =
          adjusted_transfer_time(q.transfer_time_settings_, fp.duration());
      if (fp_dur < fp_dur_best) {
        auto const arr = stp.time(event_type::kArr);
        auto const fp_end = arr + fp_dur;
        if (fp_end <= fp_leg.arr_time_) {
          fp_leg.from_ = stp.get_location_idx();
          fp_leg.dep_time_ = arr;
          fp_leg.arr_time_ = fp_end;
          fp_leg.uses_ = footpath{dest_loc, fp_dur};
          transport_leg.to_ = stp.get_location_idx();
          transport_leg.arr_time_ = arr;
          ree.stop_range_.to_ = stp.stop_idx_ + 1U;
          fp_dur_best = fp_dur;
        }
      }
      break;
    }
  }
}

// A transfer point is preferred if a transfers.txt type 0/1 rule names it.
// Both endpoints match through their parent chain, so a station level rule
// covers the platforms of that station.
bool is_preferred_transfer(timetable const& tt,
                           location_idx_t const from,
                           location_idx_t const to,
                           trip_idx_t const from_trip,
                           trip_idx_t const to_trip) {
  auto const& preferred = tt.locations_.preferred_transfers_;
  auto const route_of = [&](trip_idx_t const t) {
    return t == trip_idx_t::invalid() ? route_id_idx_t::invalid()
                                      : tt.trip_route_id_[t];
  };
  auto const side_matches = [&](trip_idx_t const rule_trip,
                                route_id_idx_t const rule_route,
                                trip_idx_t const observed) {
    if (rule_trip != trip_idx_t::invalid()) {
      return rule_trip == observed;
    }
    if (rule_route != route_id_idx_t::invalid()) {
      return rule_route == route_of(observed);
    }
    return true;  // unscoped
  };
  auto const covers = [&](location_idx_t const rule_stop,
                          location_idx_t const l) {
    return rule_stop == l || rule_stop == tt.locations_.parents_[l];
  };

  for (auto f = from; f != location_idx_t::invalid();
       f = tt.locations_.parents_[f]) {
    if (to_idx(f) >= preferred.size()) {
      continue;
    }
    for (auto const& p : preferred[f]) {
      if (side_matches(p.from_trip_, p.from_route_, from_trip) &&
          side_matches(p.to_trip_, p.to_route_, to_trip) && covers(p.to_, to)) {
        return true;
      }
    }
  }
  return false;
}

double get_penalty(timetable const& tt,
                   duration_t const duration,
                   duration_t const buffer,
                   location_idx_t const from,
                   location_idx_t const to,
                   trip_idx_t const from_trip,
                   trip_idx_t const to_trip) {
  // Basic weight: footpath duration.
  auto weight = static_cast<double>(duration.count());

  // Adjust for station size.
  if (matches(tt, location_match_mode::kEquivalent, from, to)) {
    auto const x = tt.locations_.get_root_idx(from);
    weight -= (static_cast<double>(tt.locations_.location_importance_[x]) /
               tt.locations_.max_importance_) *
              3U;
  }

  // Consider transfer time buffer (clamped + weighted)
  constexpr auto kMaxRewardedBuffer = duration_t{15};
  constexpr auto kBufferWeight = 0.2;
  weight -= kBufferWeight *
            static_cast<double>(
                std::clamp(buffer, duration_t{0}, kMaxRewardedBuffer).count());

  // Prefer the transfer points recommended by the input data.
  constexpr auto kPreferredBonus = 3.0;
  if (is_preferred_transfer(tt, from, to, from_trip, to_trip)) {
    weight -= kPreferredBonus;
  }

  return weight;
}

void optimize_transfers(timetable const& tt,
                        rt_timetable const* rtt,
                        query const& q,
                        journey& j) {
  // v = vias reached in previous legs
  auto v = via_offset_t{0};

  for (auto i = 0U; i + 2 < j.legs_.size(); ++i) {
    if (i > 0U) {
      auto const& prev_leg = j.legs_[i - 1];
      std::visit(utl::overloaded{
                     [&](journey::run_enter_exit const& t) {
                       auto const fr = rt::frun{tt, rtt, t.r_};
                       for (auto s = t.stop_range_.from_;
                            s != t.stop_range_.to_; ++s) {
                         if (v != q.via_stops_.size() &&
                             q.via_stops_[v].stay_ == 0_minutes &&
                             matches(tt, location_match_mode::kEquivalent,
                                     q.via_stops_[v].location_,
                                     fr[s].get_location_idx())) {
                           ++v;
                         }
                       }
                     },
                     [&](footpath const&) {
                       if (v != q.via_stops_.size() &&
                           matches(tt, location_match_mode::kEquivalent,
                                   q.via_stops_[v].location_, prev_leg.from_)) {
                         ++v;
                       }
                       if (v != q.via_stops_.size() &&
                           matches(tt, location_match_mode::kEquivalent,
                                   q.via_stops_[v].location_, prev_leg.to_)) {
                         ++v;
                       }
                     }},
                 prev_leg.uses_);
    }

    auto& leg_from = j.legs_[i];
    auto& leg_footpath = j.legs_[i + 1];
    auto& leg_to = j.legs_[i + 2];
    if (!holds_alternative<journey::run_enter_exit>(leg_from.uses_) ||
        !holds_alternative<footpath>(leg_footpath.uses_) ||
        !holds_alternative<journey::run_enter_exit>(leg_to.uses_)) {
      continue;
    }

    auto& ree_from = get<journey::run_enter_exit>(leg_from.uses_);
    auto& ree_to = get<journey::run_enter_exit>(leg_to.uses_);

    auto current_v = v;
    // footpaths from or to a via stop with stay duration != 0 are kept as is
    auto keep_transfer = false;

    auto fr_from = rt::frun{tt, rtt, ree_from.r_};
    auto from_start = static_cast<stop_idx_t>(ree_from.stop_range_.from_ + 1U);

    // make sure that no via stops are skipped in ree_from
    for (auto s = ree_from.stop_range_.from_; s < fr_from.size(); ++s) {
      if (current_v < q.via_stops_.size() &&
          matches(tt, location_match_mode::kEquivalent,
                  fr_from[s].get_location_idx(),
                  q.via_stops_[current_v].location_)) {
        if (q.via_stops_[current_v].stay_ == 0_minutes) {
          from_start = std::max(from_start, s);
          ++current_v;
        } else {
          keep_transfer = true;
          break;
        }
      }
    }

    fr_from.stop_range_ = {from_start, fr_from.size()};

    auto fr_to = rt::frun{tt, rtt, ree_to.r_};
    auto to_end = static_cast<stop_idx_t>(ree_to.stop_range_.to_ - 1U);

    // make sure that no via stops are skipped in ree_to
    for (auto s = ree_to.stop_range_.from_; s < fr_to.size(); ++s) {
      if (current_v < q.via_stops_.size() &&
          matches(tt, location_match_mode::kEquivalent,
                  fr_to[s].get_location_idx(),
                  q.via_stops_[current_v].location_)) {
        if (q.via_stops_[current_v].stay_ == 0_minutes) {
          to_end = std::min(to_end, static_cast<stop_idx_t>(s + 1U));
          ++current_v;
        } else {
          keep_transfer = true;
          break;
        }
      }
    }

    fr_to.stop_range_ = {stop_idx_t{0U}, to_end};

    if (keep_transfer) {
      continue;
    }

    auto const trip_at = [&](rt::run const& r, stop_idx_t const stop_idx,
                             event_type const ev_type) {
      auto const fr = rt::frun{tt, rtt, r};
      return fr.is_scheduled() ? fr[stop_idx].get_trip_idx(ev_type)
                               : trip_idx_t::invalid();
    };

    auto penalty_best = get_penalty(
        tt, get<footpath>(leg_footpath.uses_).duration(),
        leg_to.dep_time_ - leg_footpath.arr_time_, leg_from.from_, leg_to.to_,
        trip_at(ree_from.r_,
                static_cast<stop_idx_t>(ree_from.stop_range_.to_ - 1U),
                event_type::kArr),
        trip_at(ree_to.r_, ree_to.stop_range_.from_, event_type::kDep));
    for (auto stp_from : fr_from) {
      if (!stp_from.out_allowed()) {
        continue;
      }
      auto const from_trip = fr_from.is_scheduled()
                                 ? stp_from.get_trip_idx(event_type::kArr)
                                 : trip_idx_t::invalid();
      for (auto stp_to : fr_to) {
        if (!stp_to.in_allowed()) {
          continue;
        }

        for (auto const& fp :
             tt.locations_
                 .footpaths_out_[q.prf_idx_][stp_from.get_location_idx()]) {
          if (fp.target() != stp_to.get_location_idx()) {
            continue;
          }

          auto const fp_dur =
              adjusted_transfer_time(q.transfer_time_settings_, fp.duration());
          auto const arr = stp_from.time(event_type::kArr);
          auto const dep = stp_to.time(event_type::kDep);
          auto const arr_fp = arr + fp_dur;
          if (arr_fp <= dep) {
            auto const to_trip = fr_to.is_scheduled()
                                     ? stp_to.get_trip_idx(event_type::kDep)
                                     : trip_idx_t::invalid();
            auto const penalty = get_penalty(
                tt, fp_dur, dep - arr_fp, stp_from.get_location_idx(),
                stp_to.get_location_idx(), from_trip, to_trip);
            if (penalty < penalty_best) {
              leg_from.to_ = stp_from.get_location_idx();
              leg_from.arr_time_ = arr;
              ree_from.stop_range_.to_ =
                  stp_from.stop_idx_ + 1U;  // half open interval

              leg_to.from_ = stp_to.get_location_idx();
              leg_to.dep_time_ = dep;
              ree_to.stop_range_.from_ = stp_to.stop_idx_;

              leg_footpath.from_ = stp_from.get_location_idx();
              leg_footpath.to_ = stp_to.get_location_idx();
              leg_footpath.dep_time_ = arr;
              leg_footpath.arr_time_ = arr_fp;
              leg_footpath.uses_ = fp;

              penalty_best = penalty;
            }
          }
          break;
        }
      }
    }
  }
}

void optimize_footpaths(timetable const& tt,
                        rt_timetable const* rtt,
                        query const& q,
                        journey& j) {
  optimize_initial_departure(tt, rtt, q, j);
  optimize_initial_start_footpath(tt, rtt, q, j);
  optimize_last_arrival(tt, rtt, q, j);
  optimize_final_egress_footpath(tt, rtt, q, j);
  optimize_transfers(tt, rtt, q, j);
}

}  // namespace nigiri::routing
