#include "nigiri/routing/transfer_durations.h"

#include <chrono>
#include <algorithm>
#include <optional>
#include <variant>
#include <vector>

#include "nigiri/routing/transfer_time_settings.h"
#include "nigiri/routing/transfers.h"
#include "nigiri/rt/frun.h"
#include "nigiri/rt/rt_timetable.h"
#include "nigiri/transfer_rule_sides.h"

namespace nigiri::routing {

std::optional<duration_t> get_rule_transfer_time(
    timetable const& tt,
    rt_timetable const* rtt,
    journey::run_enter_exit const& a,
    journey::run_enter_exit const& b,
    location_idx_t const from,
    location_idx_t const to) {
  auto const signature = [&](rt::run const& r, stop_idx_t const i,
                             location_idx_t const base) {
    auto sig = std::vector<transfer_rule_side_idx>{};
    if (r.is_scheduled() &&
        i < tt.route_location_seq_[tt.transport_route_[r.t_.t_idx_]].size()) {
      get_stop_signature(tt, r.t_.t_idx_, i, base, sig);
    }
    return sig;
  };
  auto const from_base = rt::base(tt, rtt, from);
  auto const to_base = rt::base(tt, rtt, to);
  auto const sig_a = signature(
      a.r_, static_cast<stop_idx_t>(a.stop_range_.to_ - 1U), from_base);
  auto const sig_b = signature(b.r_, b.stop_range_.from_, to_base);
  if (auto const rule = get_change_rule(tt, sig_a, sig_b, from_base, to_base);
      rule != transfer_rule_idx_t::invalid()) {
    return tt.transfer_rules_.rules_[rule].duration_;
  }
  return shortest_transfer(tt, rtt, kDefaultProfile, from_base, to_base);
}

std::optional<journey::run_enter_exit> run_of(journey::leg const& l) {
  auto const* r = std::get_if<journey::run_enter_exit>(&l.uses_);
  return r == nullptr ? std::nullopt : std::optional{*r};
}

void fix_transfer_durations(timetable const& tt,
                            rt_timetable const* rtt,
                            query const& q,
                            journey& j,
                            std::optional<journey::run_enter_exit> const prev,
                            std::optional<journey::run_enter_exit> const next) {
  if (q.prf_idx_ != kDefaultProfile || tt.transfer_rules_.rules_.empty()) {
    return;
  }
  auto const clamp = [](duration_t const d, unixtime_t::duration const gap) {
    return std::min(d, std::max(duration_t{0},
                                std::chrono::duration_cast<duration_t>(gap)));
  };
  auto const n = j.legs_.size();
  for (auto i = 0U; i != n; ++i) {
    auto& leg = j.legs_[i];
    auto const* fp = std::get_if<footpath>(&leg.uses_);
    auto const a = i == 0U ? prev : run_of(j.legs_[i - 1U]);
    auto const b = i + 1U == n ? next : run_of(j.legs_[i + 1U]);
    if (fp == nullptr || !a.has_value() || !b.has_value()) {
      continue;
    }
    auto const d = get_rule_transfer_time(tt, rtt, *a, *b, leg.from_, leg.to_);
    if (!d.has_value() || *d == footpath::kBan) {
      continue;
    }
    auto const duration = adjusted_transfer_time(q.transfer_time_settings_, *d);
    leg.uses_ = footpath{fp->target(), *d};
    if (i == 0U) {
      auto const arr_prev =
          rt::frun{tt, rtt,
                   a->r_}[static_cast<stop_idx_t>(a->stop_range_.to_ - 1U)]
              .time(event_type::kArr);
      leg.dep_time_ = leg.arr_time_ - clamp(duration, leg.arr_time_ - arr_prev);
    } else {
      auto const dep_next =
          i + 1U == n ? rt::frun{tt, rtt, b->r_}[b->stop_range_.from_].time(
                            event_type::kDep)
                      : j.legs_[i + 1U].dep_time_;
      leg.arr_time_ = leg.dep_time_ + clamp(duration, dep_next - leg.dep_time_);
    }
  }
}

}  // namespace nigiri::routing
