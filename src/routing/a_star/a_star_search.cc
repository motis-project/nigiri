#include "nigiri/routing/a_star/a_star_search.h"

#include <cassert>
#include "nigiri/rt/frun.h"
#include <utility>
#include <vector>

#include "nigiri/routing/get_earliest_transport.h"
#include "nigiri/special_stations.h"

// What penalty do transfers cause in the cost function?
#define ALPHA 10
// What is the maximal possible value of the cost function?
// In the "Praktikum Algorithmen" lab, a journey can't take more than 24h (=
// 1440 min). On top of that, there are certainly not more than 10 transfers.
#define MAX_BUCKET (1440 + 10 * ALPHA)

namespace nigiri::routing::a_star {

routing_result initialize_a_star(timetable const& tt,
                                 search_state& search_state,
                                 tb::query_state& algo_state,
                                 query q) {
  return routing::search<direction::kForward, a_star::a_star_search>{
      tt, nullptr, search_state, algo_state, std::move(q)}
      .execute();
}

a_star_search::a_star_search(
    timetable const& tt,
    rt_timetable const* rtt,
    tb::query_state& state,
    bitvec const& is_dest,
    std::array<bitvec, kMaxVias> const& is_via,
    std::vector<std::uint16_t> const& dist_to_dest,
    hash_map<location_idx_t, std::vector<td_offset>> const& td_dist_to_dest,
    std::vector<std::uint16_t> const& lower_bounds,
    std::vector<via_stop> const& via_stops,
    day_idx_t base_day,
    clasz_mask_t allowed_claszes,
    bool require_bike_transport,
    bool require_car_transport,
    bool is_wheelchair,
    transfer_time_settings const& tts)
    : tt_{tt}, state_{state}, is_dest_{is_dest}, lower_bounds_{lower_bounds} {
  get_bucket_ = [this](open_set_element el) {
    return cost_function(el, arrival_times_.at(el.seg),
                         num_transfers_until_segment_.at(el.seg));
  };
  open_set_ = dial<open_set_element, bucket_fn>(MAX_BUCKET, get_bucket_);
  closed_set_.resize(state_.tbd_.segment_transports_.size());

  // Analogously to the constructor in query_engine.cc for tb,
  // 'state_.end_reachable_' and 'state_.dist_to_dest_' are built here.
  state_.reset();
  auto const mark_dest_segments = [&](location_idx_t const l,
                                      duration_t const d) {
    for (auto const r : tt_.location_routes_[l]) {
      auto const stop_seq = tt_.route_location_seq_[r];
      for (auto i = stop_idx_t{1U}; i != stop_seq.size(); ++i) {
        auto const stp = stop{stop_seq[i]};
        if (stp.location_idx() != l || !stp.out_allowed()) {
          continue;
        }
        for (auto const t : tt_.route_transport_ranges_[r]) {
          auto const segment = state_.tbd_.transport_first_segment_[t] + i - 1;
          state_.end_reachable_.set(segment, true);

          auto const it = state_.dist_to_dest_.find(segment);
          if (it == end(state_.dist_to_dest_)) {
            state_.dist_to_dest_.emplace_hint(it, segment, d);
          } else {
            it->second = std::min(it->second, d);
          }
        }
      }
    }
  };

  if (dist_to_dest.empty()) {
    // stop destination
    is_dest_.for_each_set_bit([&](std::size_t const i) {
      auto const l = location_idx_t{i};
      mark_dest_segments(l, duration_t{0U});
      for (auto const fp :
           tt_.locations_.footpaths_in_[state_.tbd_.prf_idx_][l]) {
        mark_dest_segments(fp.target(), fp.duration());
      }
    });
  } else {
    // coordinate destination (intermodal)
    for (auto const [l_idx, dist] : utl::enumerate(dist_to_dest)) {
      if (dist != std::numeric_limits<std::uint16_t>::max()) {
        mark_dest_segments(location_idx_t{l_idx},
                           duration_t{static_cast<unsigned>(dist)});
      }
    }
  }
}

void a_star_search::add_start(location_idx_t const l, unixtime_t const t) {
  auto const [day, mam] = tt_.day_idx_mam(t);
  for (auto const r : tt_.location_routes_[l]) {
    auto const stop_seq = tt_.route_location_seq_[r];
    for (auto i = stop_idx_t{0U}; i < stop_seq.size() - 1; ++i) {
      auto const stp = stop{stop_seq[i]};
      if (!stp.in_allowed() || stp.location_idx() != l) {
        continue;
      }

      auto const et = get_earliest_transport<direction::kForward>(
          tt_, tt_, 0U, r, i, day, mam, l,
          [](day_idx_t, std::int16_t) { return false; });
      if (!et.is_valid()) {
        continue;
      }

      auto const segment = state_.tbd_.transport_first_segment_[et.t_idx_] + i;
      potential_starts_.push_back({segment, et});
    }
  }
}

void a_star_search::execute(unixtime_t start_time,
                            uint8_t max_transfers,
                            unixtime_t worst_time_at_dest,
                            profile_idx_t prf_idx,
                            pareto_set<journey>& journeys) {
  journey_found_ = false;
  start_time_ = start_time;
  // In the "Praktikum Algorithmen" lab, a journey can't take more than 24h (=
  // 1440 min). On top of that, there are certainly not more than 10 transfers.
  cost_upper_bound_ =
      std::min(1440, (worst_time_at_dest - start_time).count()) +
      std::min(10, static_cast<int>(max_transfers)) * ALPHA;
  for (auto [segment, et] : potential_starts_) {
    open_set_element el{element_type::SEGMENT, segment};
    auto [from, to] = get_segment_locations(segment);
    to_location_[segment] = to;
    segment_day_[segment] = et.day_;
    if (cost_function(el, get_arrival_time(segment), 0) > cost_upper_bound_) {
      continue;
    };

    arrival_times_[segment] = get_arrival_time(segment);
    num_transfers_until_segment_[segment] = 0;
    start_segments_.push_back(segment);
    open_set_.push(open_set_element{element_type::SEGMENT, segment});
  }

  while (!open_set_.empty()) {
    auto el = open_set_.top();
    auto current = el.seg;
    open_set_.pop();
    if (closed_set_.test(current)) {
      continue;
    }
    auto [from, to] = get_segment_locations(current);
    if (el.type == element_type::FINAL_FOOTPATH || is_dest_[to_idx(to)]) {
      journey_found_ = true;
      end_segment_ = current;
      break;
    }
    closed_set_.set(current);
    if (arrival_times_[current] < worst_time_at_dest &&
        num_transfers_until_segment_[current] < max_transfers) {
      expand_node(current);
    }
  }

  if (journey_found_) {
    auto const offset = state_.dist_to_dest_.at(end_segment_);
    auto total_transfers =
        static_cast<uint8_t>(num_transfers_until_segment_[end_segment_]);
    journeys.add(journey{.start_time_ = start_time,
                         .dest_time_ = arrival_times_[end_segment_] + offset,
                         .transfers_ = total_transfers});
  }
}

void a_star_search::reconstruct(query const& q, journey& j) {
  if (!journey_found_) {
    return;
  }

  vector<tb::segment_idx_t> path_segments{};
  tb::segment_idx_t current_seg = end_segment_;
  path_segments.push_back(current_seg);
  while (came_from_.contains(current_seg)) {
    current_seg = came_from_[current_seg];
    path_segments.push_back(current_seg);
  }
  std::ranges::reverse(path_segments);
  location_idx_t first_location_on_current_trip{0};
  stop_idx_t first_stop_on_current_trip{0};
  tb::segment_idx_t first_segment_on_current_trip{0};
  int num_transfers = 0;
  for (int i = 0; i < path_segments.size(); i++) {
    tb::segment_idx_t seg = path_segments[i];
    auto [start_location, end_location] = get_segment_locations(seg);

    auto const has_offset = [&](std::vector<offset> const& offsets,
                                location_match_mode const match_mode,
                                location_idx_t const l) {
      return utl::any_of(offsets, [&](offset const& o) {
        return matches(tt_, match_mode, o.target(), l);
      });
    };

    if (i == 0) {
      if (q.start_match_mode_ == location_match_mode::kIntermodal) {
        // *** first leg ***
        // We may only search an offset with 'j.start_time_ + o.duration() <=
        // get_departure_time(seg)' instead of ==, because there doesn't have to
        // be an offset that starts exactly at the start time. To make sure that
        // the total journey duration is still displayed correctly, we update
        // 'j.start_time_' in the line after that.
        auto const offset_it =
            utl::find_if(q.start_, [&](routing::offset const& o) {
              return o.target() == start_location &&
                     j.start_time_ + o.duration() <= get_departure_time(seg);
            });
        j.start_time_ = get_departure_time(seg) - offset_it->duration();
        j.add(journey::leg(
            direction::kForward, get_special_station(special_station::kStart),
            start_location, get_departure_time(seg) - offset_it->duration(),
            get_departure_time(seg), *offset_it));
      } else {
        for (auto const fp : tt_.locations_.footpaths_in_[state_.tbd_.prf_idx_]
                                                         [start_location]) {
          if (j.start_time_ + fp.duration() <= get_departure_time(seg) &&
              has_offset(q.start_, q.start_match_mode_, fp.target())) {
            j.start_time_ = get_departure_time(seg) - fp.duration();
            j.add(journey::leg(direction::kForward, fp.target(), start_location,
                               get_departure_time(seg) - fp.duration(),
                               get_departure_time(seg), fp));
            break;
          }
        }
      }
    }

    auto const transport_id = state_.tbd_.segment_transports_[seg];
    auto const first = state_.tbd_.transport_first_segment_[transport_id];
    auto const rel = (seg - first);  // relative difference of requested segment
                                     // compared to first segment of this trip
    auto const from_stop = static_cast<stop_idx_t>(to_idx(rel));
    auto const to_stop = static_cast<stop_idx_t>(from_stop + 1U);

    // If first segment of journey on current trip, we update the first
    // location/start variables.
    if (i == 0 || state_.tbd_.segment_transports_[seg] !=
                      state_.tbd_.segment_transports_[path_segments[i - 1]]) {
      first_location_on_current_trip = start_location;
      first_stop_on_current_trip = from_stop;
      first_segment_on_current_trip = seg;
    }

    // If last segment of journey on current trip, we combine all segments to
    // one journey leg.
    if (i == path_segments.size() - 1 ||
        state_.tbd_.segment_transports_[seg] !=
            state_.tbd_.segment_transports_[path_segments[i + 1]]) {
      // ** (intermediate) run **
      j.add(journey::leg(
          direction::kForward, first_location_on_current_trip, end_location,
          get_departure_time(first_segment_on_current_trip),
          get_arrival_time(seg),
          journey::run_enter_exit(
              rt::run{
                  .t_ = {state_.tbd_.segment_transports_[seg],
                         segment_day_[seg]},
                  .stop_range_ = {0U,
                                  static_cast<stop_idx_t>(
                                      tt_.route_location_seq_
                                          [tt_.transport_route_[transport_id]]
                                              .size())}},
              first_stop_on_current_trip, to_stop)));
      if (i != path_segments.size() - 1) {
        // *** intermediate footpath ***
        j.add(journey::leg(
            direction::kForward, end_location, end_location,
            get_arrival_time(seg),
            get_arrival_time(seg) + tt_.locations_.transfer_time_[end_location],
            footpath(end_location,
                     duration_t{std::chrono::duration<int, std::ratio<60>>(
                         tt_.locations_.transfer_time_[end_location])})));
        num_transfers++;
      } else {
        // *** final leg ***
        auto const duration_to_dest = state_.dist_to_dest_.at(seg);
        if (duration_to_dest.count() == 0) {
          j.add(journey::leg(
              direction::kForward, end_location, end_location,
              get_arrival_time(seg), get_arrival_time(seg),
              footpath(
                  end_location,
                  duration_t{std::chrono::duration<int, std::ratio<60>>(0)})));
        } else {
          auto const offset_it =
              std::find_if(begin(q.destination_), end(q.destination_),
                           [&](routing::offset const& o) {
                             return o.target() == end_location &&
                                    o.duration() == duration_to_dest;
                           });  // analogous to tb implementation
          j.add(journey::leg(
              direction::kForward, end_location,
              get_special_station(special_station::kEnd), get_arrival_time(seg),
              get_arrival_time(seg) + duration_to_dest, *offset_it));
        }
      }
    }
  }
}

void a_star_search::expand_node(tb::segment_idx_t current) {
  auto const t = state_.tbd_.segment_transports_[current];
  auto const first = state_.tbd_.transport_first_segment_[t];

  // next segment on same route/trip direction
  auto const seg_range = state_.tbd_.get_segment_range(t);
  if (cista::to_idx(current - first) + 1 < seg_range.size()) {
    auto const next_seg = tb::segment_idx_t{current + 1};
    auto next_seg_day = segment_day_[current];
    progress_neighbor_segment(current, next_seg, next_seg_day, true);
  }

  // all segments reachable from a transfer
  for (auto neighbor_transfer : state_.tbd_.segment_transfers_[current]) {
    auto neighbor = neighbor_transfer.to_segment_;
    auto new_neighbor_day =
        segment_day_[current] + neighbor_transfer.get_day_offset();
    if (state_.tbd_.bitfields_[neighbor_transfer.traffic_days_].test(
            to_idx(segment_day_[current]))) {
      progress_neighbor_segment(current, neighbor, new_neighbor_day, false);
    }
  }

  // footpath directly to destination
  if (state_.end_reachable_[current]) {
    open_set_element el{element_type::FINAL_FOOTPATH, current};
    open_set_.push(el);
    cost_upper_bound_ =
        std::min(cost_upper_bound_,
                 cost_function(el, arrival_times_.at(el.seg),
                               num_transfers_until_segment_.at(el.seg)));
  }
}

void a_star_search::progress_neighbor_segment(tb::segment_idx_t current,
                                              tb::segment_idx_t neighbor,
                                              day_idx_t neighbor_day,
                                              bool is_same_trip) {
  auto [from_location, to_location] = get_segment_locations(neighbor);
  if (closed_set_.test(neighbor) || lower_bounds_.at(to_idx(to_location)) ==
                                        std::numeric_limits<uint16_t>::max()) {
    return;
  }

  auto const t = state_.tbd_.segment_transports_[neighbor];
  auto const to_stop_idx = static_cast<stop_idx_t>(
      to_idx(neighbor - state_.tbd_.transport_first_segment_[t]) + 1);
  auto new_neighbor_arr_time =
      tt_.event_time({t, neighbor_day}, to_stop_idx, event_type::kArr);
  uint32_t new_neighbor_transfer_number;
  if (is_same_trip) {
    new_neighbor_transfer_number = num_transfers_until_segment_[current];
  } else {
    new_neighbor_transfer_number = num_transfers_until_segment_[current] + 1;
  }

  open_set_element neighbor_element{element_type::SEGMENT, neighbor};
  to_location_[neighbor] = to_location;
  auto new_neighbor_cost = cost_function(
      neighbor_element, new_neighbor_arr_time, new_neighbor_transfer_number);
  if (new_neighbor_cost >= cost_upper_bound_) {
    return;
  }
  // If the neighbor has already been reached before, and the path to the
  // neighbor over current isn't better than the best known path to the neighbor
  // so far, we can skip this neighbor.
  if (arrival_times_.contains(neighbor) &&
      new_neighbor_cost >=
          cost_function(neighbor_element, arrival_times_.at(neighbor),
                        num_transfers_until_segment_.at(neighbor))) {
    return;
  }

  segment_day_[neighbor] = neighbor_day;
  arrival_times_[neighbor] = new_neighbor_arr_time;
  num_transfers_until_segment_[neighbor] = new_neighbor_transfer_number;
  came_from_[neighbor] = current;
  open_set_.push(neighbor_element);
}

size_t a_star_search::cost_function(open_set_element el,
                                    unixtime_t arrival_time,
                                    uint32_t num_transfers_until_segment) {
  if (el.type == element_type::SEGMENT) {
    auto to_location_idx = to_idx(to_location_.at(el.seg));
    return (arrival_time - start_time_).count() +
           ALPHA * num_transfers_until_segment +
           lower_bounds_.at(to_location_idx);
  } else {
    return (arrival_time - start_time_).count() +
           ALPHA * num_transfers_until_segment +
           state_.dist_to_dest_.at(el.seg).count();
  }
}

// This method returns start and the end location for a given segment.
pair<location_idx_t, location_idx_t> a_star_search::get_segment_locations(
    tb::segment_idx_t seg) {
  // copied from 'segment_info.h' in 'nigiri/routing/tb'
  auto const t = state_.tbd_.segment_transports_[seg];
  auto const first = state_.tbd_.transport_first_segment_[t];
  auto const rel = (seg - first);
  auto const from_stop = static_cast<stop_idx_t>(to_idx(rel));
  auto const to_stop = static_cast<stop_idx_t>(from_stop + 1U);
  auto const loc_seq = tt_.route_location_seq_[tt_.transport_route_[t]];
  auto const from_location = stop{loc_seq[from_stop]}.location_idx();
  auto const to_location = stop{loc_seq[to_stop]}.location_idx();
  return {from_location, to_location};
}

unixtime_t a_star_search::get_arrival_time(tb::segment_idx_t seg) {
  auto const t = state_.tbd_.segment_transports_[seg];
  auto const to_stop_idx = static_cast<stop_idx_t>(
      to_idx(seg - state_.tbd_.transport_first_segment_[t]) + 1);
  auto arr_time =
      tt_.event_time({t, segment_day_[seg]}, to_stop_idx, event_type::kArr);
  return arr_time;
}

unixtime_t a_star_search::get_departure_time(tb::segment_idx_t seg) {
  auto const t = state_.tbd_.segment_transports_[seg];
  auto const from_stop_idx = static_cast<stop_idx_t>(
      to_idx(seg - state_.tbd_.transport_first_segment_[t]));
  auto dep_time =
      tt_.event_time({t, segment_day_[seg]}, from_stop_idx, event_type::kDep);
  return dep_time;
}

}  // namespace nigiri::routing::a_star
