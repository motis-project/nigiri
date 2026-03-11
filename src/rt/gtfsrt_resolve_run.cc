#include "nigiri/rt/gtfsrt_resolve_run.h"

#include "nigiri/common/day_list.h"
#include "nigiri/common/mam_dist.h"
#include "nigiri/common/split_duration.h"
#include "nigiri/rt/rt_timetable.h"
#include "nigiri/rt/trip_update.h"
#include "nigiri/timetable.h"
#include "nigiri/types.h"
#include "utl/enumerate.h"

#include "utl/pipes/find.h"

namespace nigiri::rt {
std::pair<date::days, duration_t> split(duration_t const i) {
  auto const a = i.count() / 1440;
  auto const b = i.count() % 1440;
  return {date::days{a}, duration_t{b}};
}

void resolve_static(date::sys_days const today,
                    timetable const& tt,
                    source_idx_t const src,
                    transit_realtime::TripDescriptor const& td,
                    run& r,
                    trip_idx_t& trip) {
  resolve_static(today, tt, src, td, [&](run const& a, trip_idx_t const b) {
    r = a;
    trip = b;
    return utl::continue_t::kBreak;
  });
}

void resolve_rt(rt_timetable const& rtt,
                run& output,
                std::string_view rt_changed_trip_id,
                source_idx_t const src) {
  auto const it = rtt.static_trip_lookup_.find(output.t_);
  if (it != end(rtt.static_trip_lookup_)) {
    output.rt_ = it->second;
    return;
  }
  if (output.is_scheduled()) {
    return;
  }
  auto const rt_add_idx =
      rtt.additional_trips_.at(src).ids_.find(rt_changed_trip_id);
  if (rt_add_idx.has_value()) {
    output.rt_ = rtt.additional_trips_.at(src).transports_[*rt_add_idx];
    if (output.stop_range_.size() == 0) {
      output.stop_range_ = {
          static_cast<stop_idx_t>(0U),
          static_cast<stop_idx_t>(
              rtt.rt_transport_location_seq_.at(output.rt_).size())};
    }
  }
}

std::pair<run, trip_idx_t> gtfsrt_resolve_run(
    date::sys_days const today,
    timetable const& tt,
    rt_timetable const* rtt,
    source_idx_t const src,
    transit_realtime::TripDescriptor const& td,
    std::string_view rt_changed_trip_id) {
  auto r = run{};
  trip_idx_t trip;
  resolve_static(today, tt, src, td, r, trip);
  if (rtt != nullptr) {
    resolve_rt(*rtt, r,
               rt_changed_trip_id.empty() ? td.trip_id() : rt_changed_trip_id,
               src);
  }
  return {r, trip};
}

run gtfsrt_vp_resolve_run(timetable const& tt,
                          source_idx_t src,
                          transit_realtime::VehiclePosition const& vp,
                          vehicle_trip_matching* vtm) {

  if (std::chrono::system_clock::now() - vtm->last_cleanup >
      std::chrono::hours{12}) {
    vtm->clean_up();
  }

  constexpr auto kExactMatchScore = 230;
  constexpr auto kAdditionalMatchThreshold = 0.975;
  constexpr auto match_threshold = 0.7;

  // if vehicle_id is key in vehicle_idx_run_ -> return value
  // if vehicle_id is not key in vehicle_idx_known_vps_ -> create new entry
  // try matching
  // if vehicle is key in vehicle_idx_known_stops_ -> update known_stops
  // try matching

  auto const vp_ts = vp.has_timestamp()
                         ? unixtime_t{std::chrono::duration_cast<i32_minutes>(
                               std::chrono::seconds{vp.timestamp()})}
                         : std::chrono::time_point_cast<i32_minutes>(
                               std::chrono::system_clock::now());

  auto vehicle_idx = vehicle_idx_t::invalid();

  auto new_stop_loc = location_idx_t::invalid();

  if (vp.has_stop_id()) {
    new_stop_loc = tt.find(location_id{vp.stop_id(), src})
                       .value_or(location_idx_t::invalid());
  }
  if (!vp.has_stop_id() || new_stop_loc == location_idx_t::invalid()) {
    auto const vp_pos =
        geo::latlng{vp.position().latitude(), vp.position().longitude()};
    auto const box = geo::box{vp_pos, 500};

    auto nearest_stop_loc = location_idx_t::invalid();
    auto dist_nearest_stop = std::numeric_limits<double>::infinity();
    tt.locations_.rtree_.search(
        box.min_.lnglat_float(), box.max_.lnglat_float(),
        [&](auto, auto, location_idx_t const loc_search) {
          if (tt.locations_.src_[loc_search] != src) {
            return false;
          }
          if (tt.locations_.types_[loc_search] != location_type::kStation) {
            return false;
          }
          auto const app_dist_lng_deg_vp =
              geo::approx_distance_lng_degrees(vp_pos);
          if (dist_nearest_stop > geo::approx_squared_distance(
                                      tt.locations_.coordinates_[loc_search],
                                      vp_pos, app_dist_lng_deg_vp)) {
            nearest_stop_loc = loc_search;
            dist_nearest_stop = geo::approx_squared_distance(
                tt.locations_.coordinates_[loc_search], vp_pos,
                app_dist_lng_deg_vp);
          }
          return true;
        });
    new_stop_loc = nearest_stop_loc;
  }

  // if it is not possible to pinpoint a stop -> no valuable information in
  // terms of matching
  if (new_stop_loc == location_idx_t::invalid()) {
    return run{};
  }

  if (vp.has_vehicle() && vp.vehicle().has_id()) {
    auto const lb = std::lower_bound(
        begin(vtm->vehicle_id_to_idx_), end(vtm->vehicle_id_to_idx_),
        vp.vehicle().id(),
        [&](pair<vehicle_id_idx_t, vehicle_idx_t> const& a, auto&& b) {
          return std::tuple{vtm->vehicle_id_src_[a.first],
                            vtm->vehicle_id_strings_[a.first].view()} <
                 std::tuple{src, static_cast<std::string_view>(b)};
        });

    auto const id_matches = [&](vehicle_id_idx_t const v_id_idx) {
      return vtm->vehicle_id_src_[v_id_idx] == src &&
             vtm->vehicle_id_strings_[v_id_idx].view() == vp.vehicle().id();
    };

    if (lb != end(vtm->vehicle_id_to_idx_) && id_matches(lb->first)) {
      vehicle_idx = lb->second;
      // if we already have a run for this vehicle
      if (vtm->vehicle_idx_run_.count(vehicle_idx) > 0) {
        return vtm->vehicle_idx_run_[vehicle_idx];
      }
    } else {
      vehicle_idx = vehicle_idx_t{vtm->vehicle_id_to_idx_.size() - 1};
      vtm->vehicle_id_strings_.emplace_back(vp.vehicle().id());
      vtm->vehicle_id_src_.emplace_back(src);
      vtm->vehicle_id_to_idx_.emplace_back(
          vehicle_id_idx_t{vtm->vehicle_id_strings_.size() - 1}, vehicle_idx);
    }

    if (vtm->vehicle_idx_known_stop_locs_.count(vehicle_idx) > 0) {
      if (vtm->vehicle_idx_known_stop_locs_[vehicle_idx].back() !=
          new_stop_loc) {
        vtm->vehicle_idx_known_stop_locs_[vehicle_idx].emplace_back(
            new_stop_loc);
      }
    } else {
      vtm->vehicle_idx_known_stop_locs_[vehicle_idx] = {new_stop_loc};
    }
  }

  auto vp_candidates = vector<vp_candidate>{};

  auto const stop_locs = vehicle_idx != vehicle_idx_t::invalid()
                             ? vtm->vehicle_idx_known_stop_locs_[vehicle_idx]
                             : vector{new_stop_loc};
  for (auto const& stop_loc_cand : stop_locs) {
    for (auto const l : tt.locations_.equivalences_[stop_loc_cand]) {
      for (auto const route : tt.location_routes_[l]) {
        auto const location_seq = tt.route_location_seq_[route];
        for (auto const [stop_idx, s] : utl::enumerate(location_seq)) {
          if (stop{s}.location_idx() != l) {
            continue;
          }

          auto const ev_type =
              stop_idx == 0 ? event_type::kDep : event_type::kArr;

          // alle Events an dem Stop angucken
          for (auto const [nigiri_ev_time_idx, nigiri_ev_time] :
               utl::enumerate(tt.event_times_at_stop(
                   route, static_cast<stop_idx_t>(stop_idx), ev_type))) {
            auto const [vp_day_idx, vp_mam] = tt.day_idx_mam(vp_ts);

            auto const [error, day_shift] =
                mam_dist(vp_mam, i32_minutes{nigiri_ev_time.mam()});
            auto const local_score =
                kExactMatchScore - error.count() * error.count();
            if (local_score < 0) {
              continue;
            }
            // wenn timing passt: entprechenden transport holen
            auto const tr = transport{
                tt.route_transport_ranges_[route][nigiri_ev_time_idx],
                vp_day_idx -
                    day_idx_t{nigiri_ev_time.days() - day_shift.count()}};

            if (tt.bitfields_[tt.transport_traffic_days_[tr.t_idx_]].test(
                    to_idx(tr.day_))) {
              auto candidate =
                  std::find_if(begin(vp_candidates), end(vp_candidates),
                               [&](auto const& c) { return c.r_.t_ == tr; });

              if (candidate != end(vp_candidates) &&
                  stop_idx < candidate->r_.stop_range_.from_) {
                continue;
              }

              if (candidate == end(vp_candidates)) {
                vp_candidates.emplace_back(
                    run{tr,
                        interval{static_cast<stop_idx_t>(stop_idx),
                                 static_cast<stop_idx_t>(location_seq.size())}},
                    location_seq.size());
                candidate = end(vp_candidates) - 1;
              }

              candidate->local_best_ =
                  std::max(candidate->local_best_,
                           static_cast<std::uint32_t>(local_score));
            }
          }
        }
      }
    }
    for (auto& c : vp_candidates) {
      c.finish_stop();
    }
  }

  utl::sort(vp_candidates);

  auto matches = vector<run>{};
  auto const is_match = [&](auto const& c) {
    return c.score_ > vp_candidates.front().score_ * kAdditionalMatchThreshold;
  };
  if (!vp_candidates.empty() &&
      vp_candidates.front().score_ >
          stop_locs.size() * kExactMatchScore * match_threshold) {
    for (auto const c : vp_candidates) {
      if (is_match(c)) {
        matches.emplace_back(c.r_);
      }
    }
  }
  if (matches.size() == 1) {
    if (vp.has_vehicle() && vp.vehicle().has_id()) {
      std::chrono::sys_seconds now{
          std::chrono::time_point_cast<std::chrono::seconds>(
              std::chrono::system_clock::now())};
      vtm->vehicle_idx_run_.emplace(vehicle_idx, matches.front());
      vtm->vehicle_idx_last_access_.emplace(vehicle_idx, now);
    }
    return matches.front();
  }

  return run{};
}

}  // namespace nigiri::rt
