#include "nigiri/routing/raptor/reconstruct.h"

#include "utl/overloaded.h"

#include "nigiri/for_each_meta.h"
#include "nigiri/routing/journey.h"
#include "nigiri/rt/frun.h"
#include "nigiri/rt/rt_timetable.h"
#include "nigiri/timetable.h"

namespace nigiri::routing {

template <direction SearchDir>
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

  auto const& offsets =
      (SearchDir == direction::kBackward) ? q.destination_ : q.start_;

  auto r = rt::run{ree.r_};
  r.stop_range_ = {0U, static_cast<stop_idx_t>(ree.stop_range_.to_ - 1U)};
  for (auto const stp : rt::frun{tt, rtt, r}) {
    if (!q.via_stops_.empty() &&
        matches(tt, location_match_mode::kEquivalent, stp.get_location_idx(),
                q.via_stops_[0].location_)) {
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

template <direction SearchDir>
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
  if constexpr (SearchDir == direction::kBackward) {
    offsets = &q.start_;
  }

  auto fr = rt::frun{tt, rtt, ree.r_};
  auto range_from = static_cast<stop_idx_t>(ree.stop_range_.from_ + 1U);

  if (!q.via_stops_.empty()) {
    // don't skip the last via stop
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

double get_penalty(timetable const& tt,
                   duration_t const duration,
                   location_idx_t const from,
                   location_idx_t const to) {
  auto weight = static_cast<double>(duration.count());
  if (matches(tt, location_match_mode::kEquivalent, from, to)) {
    auto const x = tt.locations_.get_root_idx(from);
    weight -= (static_cast<double>(tt.locations_.location_importance_[x]) /
               tt.locations_.max_importance_) *
              3U;
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
        !holds_alternative<footpath>(j.legs_[i + 1].uses_) ||
        !holds_alternative<journey::run_enter_exit>(j.legs_[i + 2].uses_)) {
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

    auto penalty_best =
        get_penalty(tt, get<footpath>(leg_footpath.uses_).duration(),
                    leg_from.from_, leg_to.to_);
    for (auto stp_from : fr_from) {
      if (!stp_from.out_allowed()) {
        continue;
      }
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
          auto const penalty =
              get_penalty(tt, fp_dur, stp_from.get_location_idx(),
                          stp_to.get_location_idx());
          if (penalty >= penalty_best) {
            continue;
          }

          auto const arr = stp_from.time(event_type::kArr);
          auto const dep = stp_to.time(event_type::kDep);
          auto const arr_fp = arr + fp_dur;
          if (arr_fp <= dep) {
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
          break;
        }
      }
    }
  }
}

template <direction SearchDir>
void optimize_footpaths(timetable const& tt,
                        rt_timetable const* rtt,
                        query const& q,
                        journey& j) {
  optimize_initial_departure<SearchDir>(tt, rtt, q, j);
  optimize_last_arrival<SearchDir>(tt, rtt, q, j);
  optimize_transfers(tt, rtt, q, j);
}

template void optimize_footpaths<direction::kForward>(timetable const&,
                                                      rt_timetable const*,
                                                      query const&,
                                                      journey&);
template void optimize_footpaths<direction::kBackward>(timetable const&,
                                                       rt_timetable const*,
                                                       query const&,
                                                       journey&);

}  // namespace nigiri::routing
