#include "nigiri/routing/leg_alternatives.h"

#include "nigiri/rt/rt_timetable.h"
#include "nigiri/stop.h"
#include "nigiri/timetable.h"

namespace nigiri::routing {

std::vector<journey::leg_alternative> find_equivalent_transports(
    timetable const& tt,
    rt_timetable const* rtt,
    profile_idx_t const prf_idx,
    location_idx_t const boarding_loc,
    location_idx_t const unboarding_loc,
    unixtime_t const dep_time,
    unixtime_t const arr_time,
    transport_idx_t const exclude) {
  auto result = std::vector<journey::leg_alternative>{};
  auto const base = tt.internal_interval_days().from_;

  auto const is_transport_active = [&](transport_idx_t const t,
                                       day_idx_t const day) {
    auto const d = to_idx(day);
    return (rtt != nullptr)
               ? rtt->bitfields_[rtt->transport_traffic_days_[t]].test(d)
               : tt.bitfields_[tt.transport_traffic_days_[t]].test(d);
  };

  for (auto const r_cand : tt.location_routes_[boarding_loc]) {
    auto const loc_seq = tt.route_location_seq_[r_cand];

    // Find the first in_allowed boarding stop then the first subsequent
    // out_allowed alighting stop in this candidate route.
    auto enter_si = std::optional<stop_idx_t>{};
    auto exit_si = std::optional<stop_idx_t>{};
    for (auto si = stop_idx_t{0U};
         si != static_cast<stop_idx_t>(loc_seq.size()); ++si) {
      auto const s = stop{loc_seq[si]};
      if (!enter_si.has_value()) {
        if (s.location_idx() == boarding_loc && s.in_allowed(prf_idx)) {
          enter_si = si;
        }
      } else {
        if (s.location_idx() == unboarding_loc && s.out_allowed(prf_idx)) {
          exit_si = si;
          break;
        }
      }
    }
    if (!enter_si.has_value() || !exit_si.has_value()) {
      continue;
    }

    for (auto const t_cand : tt.route_transport_ranges_[r_cand]) {
      if (t_cand == exclude) {
        continue;
      }

      // Compute the traffic day for t_cand such that it departs at dep_time
      // from enter_si.  event_time = base + day*1440min + mam
      // => day = (dep_time - base - mam) / 1440
      auto const mam = tt.event_mam(t_cand, *enter_si, event_type::kDep);
      auto const offset = (dep_time - base).count() - mam.as_duration().count();
      if (offset < 0 || offset % 1440 != 0) {
        continue;
      }
      auto const start_day =
          day_idx_t{static_cast<cista::base_t<day_idx_t>>(offset / 1440)};

      if (tt.event_time(transport{t_cand, start_day}, *enter_si,
                        event_type::kDep) != dep_time) {
        continue;
      }
      if (tt.event_time(transport{t_cand, start_day}, *exit_si,
                        event_type::kArr) != arr_time) {
        continue;
      }
      if (!is_transport_active(t_cand, start_day)) {
        continue;
      }

      result.push_back({.transport_idx_ = t_cand,
                        .day_idx_ = start_day,
                        .enter_stop_idx_ = *enter_si,
                        .exit_stop_idx_ = *exit_si});
    }
  }

  return result;
}

}  // namespace nigiri::routing
