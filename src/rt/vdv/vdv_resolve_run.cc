#include "nigiri/rt/vdv/vdv_resolve_run.h"

#include "nigiri/rt/vdv/vdv_run.h"
#include "nigiri/timetable.h"

namespace nigiri::rt {

std::optional<location_idx_t> match_location(timetable const& tt,
                                             std::string_view vdv_stop_id) {
  for (auto l = 0U; l != tt.n_locations(); ++l) {
    auto const loc_idx = location_idx_t{l};
    if (std::string_view{begin(tt.locations_.ids_[loc_idx]),
                         end(tt.locations_.ids_[loc_idx])} == vdv_stop_id) {
      return loc_idx;
    }
  }
  return std::nullopt;
}

template <event_type ET>
void match_time(timetable const& tt,
                location_idx_t const loc_idx,
                unixtime_t const time,
                hash_set<transport>& matches) {
  auto const [base_day, base_mam] = tt.day_idx_mam(time);
  auto const time_of_day_intvl = interval{base_mam.count() - kAllowedError,
                                          base_mam.count() + kAllowedError + 1};

  for (auto const route_idx : tt.location_routes_[loc_idx]) {
    auto const loc_seq = tt.route_location_seq_[route_idx];
    for (auto stop_idx = 0U + (ET == event_type::kArr ? 1U : 0U);
         stop_idx != loc_seq.size() - (ET == event_type::kDep ? 1U : 0U);
         ++stop_idx) {

      auto const stp = stop{loc_seq[stop_idx]};
      if (stp.location_idx() != loc_idx) {
        continue;
      }

      auto const event_times_at_stop = tt.event_times_at_stop(
          route_idx, static_cast<stop_idx_t>(stop_idx), ET);
      for (auto i = 0U; i != event_times_at_stop.size(); ++i) {

        auto const normalize_event_time =
            [&time_of_day_intvl](auto const event_time) -> std::int16_t {
          if (time_of_day_intvl.to_ < event_time) {
            return event_time - 1440;
          } else if (event_time < time_of_day_intvl.from_) {
            return event_time + 1440;
          } else {
            return event_time;
          }
        };

        auto const normalized_event_time =
            normalize_event_time(event_times_at_stop[i].mam());
        if (!time_of_day_intvl.contains(normalized_event_time)) {
          continue;
        }
        auto const midnight_shift = normalized_event_time < 0      ? -1
                                    : normalized_event_time < 1440 ? 0
                                                                   : 1;
        auto const transport_day = day_idx_t{
            base_day.v_ - event_times_at_stop[i].days() + midnight_shift};
        auto const transport_idx = tt.route_transport_ranges_[route_idx][i];
        if (tt.bitfields_[tt.transport_traffic_days_[transport_idx]].test(
                transport_day.v_)) {
          matches.insert({transport_idx, transport_day});
        }
      }
    }
  }
}

hash_set<transport> match_transport(timetable const& tt, vdv_run const& r) {
  // make these static to reduce number of allocations?
  auto global_matches = hash_set<transport>{};
  auto local_matches = hash_set<transport>{};
  for (auto& vdv_stop : r.stops_) {

    auto const loc_idx = match_location(tt, vdv_stop.stop_id_);
    if (!loc_idx.has_value()) {
      log(log_lvl::error, "vdv_update.match_stops",
          "could not match stop {} to a location in the timetable",
          vdv_stop.stop_id_);
      continue;
    }

    if (vdv_stop.t_dep_.has_value()) {
      match_time<event_type::kDep>(tt, loc_idx.value(), vdv_stop.t_dep_.value(),
                                   local_matches);
    } else if (vdv_stop.t_arr_.has_value()) {
      match_time<event_type::kArr>(tt, loc_idx.value(), vdv_stop.t_arr_.value(),
                                   local_matches);
    }

    if (global_matches.empty()) {
      std::swap(global_matches, local_matches);
    } else {
      std::erase_if(global_matches,
                    [&](auto const& t) { return !local_matches.contains(t); });
      local_matches.clear();
    }

    if (global_matches.size() == 1) {
      break;
    }
  }

  return global_matches;
}

template void match_time<event_type::kDep>(timetable const&,
                                           location_idx_t const,
                                           unixtime_t const,
                                           hash_set<transport>&);
template void match_time<event_type::kArr>(timetable const&,
                                           location_idx_t const,
                                           unixtime_t const,
                                           hash_set<transport>&);
}  // namespace nigiri::rt