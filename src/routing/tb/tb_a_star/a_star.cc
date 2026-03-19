#pragma once

#include "nigiri/routing/tb/tb_a_star/a_star.h"

#include <nigiri/special_stations.h>
#include <utl/enumerate.h>

#include "nigiri/routing/get_earliest_transport.h"
#include "nigiri/timetable.h"

#include "nigiri/for_each_meta.h"

namespace nigiri::routing::tb::a_star {

tb_a_star::tb_a_star(timetable const& tt,
               rt_timetable const*,
               a_star_state& state,
               bitvec const& is_dest,
               std::array<bitvec, kMaxVias> const&,
               std::vector<std::uint16_t> const& dist_to_dest,
               hash_map<location_idx_t, std::vector<td_offset>> const&,
               std::vector<std::uint16_t> const& lb,
               std::vector<via_stop> const&,
               day_idx_t,
               clasz_mask_t,
               bool,
               bool,
               bool,
               transfer_time_settings)
    : pred_(state.tbd_.segment_transfers_.size(), segment_idx_t::invalid()),
      day_idx_(state.tbd_.segment_transfers_.size()),
      tt_{tt},
      state_{state},
      queue_{
          1440 + (state.tbd_.segment_transfers_.size() - 1) * transfer_factor,
          cost_func_t()},
      transfers_(state.tbd_.segment_transfers_.size(),
                 std::numeric_limits<uint8_t>::max()),
      travel_time_lower_bound_{lb},
      is_start_segment_(state.tbd_.segment_transfers_.size()){
  state_.end_reachable_.zero_out();
  is_start_segment_.zero_out();
  auto const mark_dest_segments = [&](location_idx_t const l,
                                  duration_t const d) {
    for (auto const r : tt_.location_routes_[l]) {
      auto const stop_seq = tt_.route_location_seq_[r];
      for (auto i = stop_idx_t{1U}; i != stop_seq.size(); ++i) {
        if (auto const stp = stop{stop_seq[i]};
            stp.location_idx() != l || !stp.out_allowed()) {
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

  if (dist_to_dest.empty()) /* Destination is stop. */ {
    is_dest.for_each_set_bit([&](std::size_t const i) {
      auto const l = location_idx_t{i};
      mark_dest_segments(l, duration_t{0U});
      for (auto const fp :
           tt_.locations_.footpaths_in_[state_.tbd_.prf_idx_][l]) {
        mark_dest_segments(fp.target(), fp.duration());
      }
    });
  } else /* Destination is coordinate. */ {
    for (auto const [l_idx, dist] : utl::enumerate(dist_to_dest)) {
      if (dist != kUnreachable) {
        mark_dest_segments(location_idx_t{l_idx}, duration_t{dist});
      }
    }
  }
}

std::pair<std::vector<segment_idx_t>,bool> get_neighbours(segment_idx_t const& s_idx, tb_data const& tbd, day_idx_t const& day) {
  std::vector<segment_idx_t> neighbours;
  for(auto const transfer : tbd.segment_transfers_[s_idx]) {
    if(tbd.bitfields_[transfer.traffic_days_].test(day.v_)) neighbours.push_back(transfer.to_segment_);
  }
  auto next_exists = s_idx+1<tbd.segment_transports_.size()&& tbd.segment_transports_[s_idx]==tbd.segment_transports_[s_idx+1];
  if(next_exists) neighbours.push_back(s_idx+1);
  return {neighbours,next_exists};
}

unixtime_t get_time(segment_idx_t const& idx,timetable const& tt,event_type const& event,tb_data const& tbd,day_idx_t const& day_idx) {
  auto const transport = tbd.segment_transports_[idx];
  auto const stop_idx = static_cast<stop_idx_t>(
        (idx - tbd.transport_first_segment_[transport]+(event==event_type::kArr?1:0)).v_);
  return tt.event_time({transport, day_idx}, stop_idx,event);
}

day_idx_t get_day(unixtime_t const& before,segment_idx_t const& s_idx, timetable const& tt,tb_data const& tbd) {
  for(auto day : tt.date_range_) {
    auto day_idx = tt.day_idx_mam(day).first;
    auto dep_time = get_time(s_idx,tt,event_type::kDep,tbd,day_idx);
    if(dep_time >= before) return day_idx;
  }
  return day_idx_t::invalid();

}

location_idx_t get_location(segment_idx_t const& s,timetable const& tt,tb_data const& tbd,event_type const& event) {
  auto transport = tbd.segment_transports_[s];
  auto stop_idx = s-tbd.transport_first_segment_[transport] + (event==event_type::kArr?1:0);
  auto route_idx = tt.transport_route_[transport];
  auto stop_value = tt.route_location_seq_[route_idx][stop_idx.v_];
  stop stop_place(stop_value);
  return stop_place.location_idx();
}

duration_t tb_a_star::heuristic(segment_idx_t const& s) {
  return duration_t(travel_time_lower_bound_[get_location(s,tt_,state_.tbd_,event_type::kArr).v_]);
}

template <typename T>
bool is_invalid(T v) {
  auto invalid_value = T::invalid();
  return v==invalid_value;
}

void tb_a_star::execute(unixtime_t const start_time,
               std::uint8_t const max_transfers,
               unixtime_t const worst_time_at_dest,
               profile_idx_t const,
               pareto_set<journey>& results) {
  vector_map<segment_idx_t,bool> seen(state_.tbd_.segment_transfers_.size(),false);
  duration_t best_cost_at_dest = duration_t::max();
  unixtime_t arrival_time_limit = std::min(worst_time_at_dest,start_time+duration_t(1441));
  is_start_segment_.for_each_set_bit([&](segment_idx_t const& s) {
    day_idx_[s] = get_day(start_time,s,tt_,state_.tbd_);
    if(is_invalid<day_idx_t>(day_idx_[s])) return;
    auto time = get_time(s,tt_,event_type::kArr,state_.tbd_,day_idx_[s]);
    if(time >= arrival_time_limit) return;
    queue_.push(std::pair(s,time-start_time+heuristic(s)));
    ++algo_stats_.queued_segments_;
  });

  while (true) {
    if(queue_.empty()) {
      if(best_cost_at_dest!=duration_t::max()) break;
      return;
    }
    auto [p,costs] = queue_.top();
    queue_.pop();
    ++algo_stats_.segments_taken_out_;
    if(costs >= best_cost_at_dest) break;
    if(seen[p]) continue;
    seen[p] = true;
    if(state_.end_reachable_.test(p)) {
      auto costs_at_dest = costs + state_.dist_to_dest_[p] - heuristic(p);
      if(costs_at_dest<best_cost_at_dest) {
        best_cost_at_dest = costs_at_dest;
        end_segment_ = p;
      }
    }
    auto arrival = get_time(p,tt_,event_type::kArr,state_.tbd_,day_idx_[p]);
    auto day_of_arrival = tt_.day_idx_mam(arrival).first;
    auto [neighbours,next_exists] = get_neighbours(p,state_.tbd_,day_of_arrival);
    for(int i=0;i<neighbours.size();i++) {
      auto neighbour = neighbours[i];
      auto is_next = next_exists && i==neighbours.size()-1;
      auto transfer_count = transfers_[p] + (is_next?0:1);
      day_idx_[neighbour] = is_next?day_idx_[p]:get_day(arrival,neighbour,tt_,state_.tbd_);
      if(is_invalid<day_idx_t>(day_idx_[neighbour])) continue;
      auto neighbour_arrival = get_time(neighbour,tt_,event_type::kArr,state_.tbd_,day_idx_[neighbour]);
      if(transfer_count<transfers_[neighbour] && neighbour_arrival < arrival_time_limit) {
        transfers_[neighbour] = transfer_count;
        pred_[neighbour] = p;
        duration_t new_costs = neighbour_arrival-start_time + duration_t(transfer_factor*transfer_count) + heuristic(neighbour);
        queue_.push(std::pair(neighbour,new_costs));
        if(seen[neighbour]) algo_stats_.ever_requeued_ = true;
        seen[neighbour] = false;
        ++algo_stats_.queued_segments_;
      }
      else {
        ++algo_stats_.segments_ignored_;
      }
    }
  }
  algo_stats_.max_transfers_reached_ = transfers_[end_segment_]>=max_transfers;
  results.add({.legs_ = {},
                   .start_time_ = start_time,
                   .dest_time_ = get_time(end_segment_,tt_,event_type::kArr,state_.tbd_,day_idx_[end_segment_])+state_.dist_to_dest_[end_segment_],
                   .dest_ = location_idx_t::invalid(),
                   .transfers_ = transfers_[end_segment_]});
}

void tb_a_star::add_start(location_idx_t const l,unixtime_t const t){
  auto const [day, mam] = tt_.day_idx_mam(t);
  for (auto const r : tt_.location_routes_[l]) {
    // iterate stop sequence of route, skip last stop
    auto const stop_seq = tt_.route_location_seq_[r];
    for (auto i = stop_idx_t{0U}; i < stop_seq.size() - 1; ++i) {
      auto const stp = stop{stop_seq[i]};
      if (!stp.in_allowed() || stp.location_idx() != l) {
        continue;
      }

      auto const et = get_earliest_transport<direction::kForward>(
          tt_, tt_, 0U, r, i, day, mam, stp.location_idx(),
          [](day_idx_t, std::int16_t) { return false; });
      if (!et.is_valid()) {
        continue;
      }

      auto const transport_first_segment =
          state_.tbd_.transport_first_segment_[et.t_idx_];
      is_start_segment_.set(transport_first_segment+i);
      transfers_[transport_first_segment+i] = 0;
    }
  }
}

segment_idx_t tb_a_star::get_departure_segment(segment_idx_t const& s) {
  auto p = pred_[s];
  auto prev = s;
  auto invalid_segment = segment_idx_t::invalid();
  while(p!=invalid_segment && state_.tbd_.segment_transports_[p]==state_.tbd_.segment_transports_[prev]) {
    prev = p;
    p = pred_[p];
  }
  return prev;
}

void tb_a_star::reconstruct(query const& q, journey& j) {
  if(is_invalid<segment_idx_t>(end_segment_)) throw std::runtime_error("reconstruction demanded with no route found");
  auto const offset = state_.dist_to_dest_.at(end_segment_);
  auto arrival_location = get_location(end_segment_,tt_,state_.tbd_,event_type::kArr);
  segment_idx_t departure_segment = get_departure_segment(end_segment_);

  auto const has_offset = [&](std::vector<routing::offset> const& offsets,
                              location_match_mode const match_mode,
                              location_idx_t const l) {
    return utl::any_of(offsets, [&](routing::offset const& o) {
      return matches(tt_, match_mode, o.target(), l);
    });
  };

  auto const get_fp = [&](location_idx_t const from, location_idx_t const to) {
    if (from == to) {
      return footpath{to, tt_.locations_.transfer_time_[from]};
    }
    auto const from_fps =
        tt_.locations_.footpaths_out_[state_.tbd_.prf_idx_][from];
    auto const it = utl::find_if(
        from_fps, [&](footpath const& fp) { return fp.target() == to; });
    utl::verify(it != end(from_fps),
                "tb reconstruct: footpath from {} to {} not found",
                loc{tt_, from}, loc{tt_, to});
    return *it;
  };

  auto const get_transport_info = [&](segment_idx_t const s,
                                      event_type const ev_type)
      -> std::tuple<transport, stop_idx_t, location_idx_t, unixtime_t> {
    auto const d = day_idx_[s];
    auto const t = state_.tbd_.segment_transports_[s];
    auto const i = static_cast<stop_idx_t>(
        to_idx(s - state_.tbd_.transport_first_segment_[t] +
               (ev_type == event_type::kArr ? 1 : 0)));
    auto const loc_seq = tt_.route_location_seq_[tt_.transport_route_[t]];
    return {{t, d},
            i,
            stop{loc_seq[i]}.location_idx(),
            tt_.event_time({t, d}, i, ev_type)};
  };

  auto const get_run_leg =
      [&](segment_idx_t const arrival_segment) -> journey::leg {
        auto const [transport, arr_stop_idx, arr_l, arr_time] =
            get_transport_info(arrival_segment, event_type::kArr);
        auto const [_, dep_stop_idx, dep_l, dep_time] =
            get_transport_info(departure_segment, event_type::kDep);
        return {
          direction::kForward,
          dep_l,
          arr_l,
          dep_time,
          arr_time,
          journey::run_enter_exit{
            rt::run{
              .t_ = transport,
              .stop_range_ = {static_cast<stop_idx_t>(0U),
                              static_cast<stop_idx_t>(
                                  tt_.route_location_seq_
                                      [tt_.transport_route_[transport.t_idx_]]
                                          .size())}},
          dep_stop_idx, arr_stop_idx}};
  };

  //Last leg
  if(q.dest_match_mode_ == location_match_mode::kIntermodal) {
    auto const offset_it =
            utl::find_if(q.destination_, [&](routing::offset const& o) {
              return o.target() == arrival_location && o.duration() == offset;
            });
    utl::verify(offset_it!=q.destination_.end(),"Reconstruction failed because end_segment is invalid");
    j.legs_.emplace_back(direction::kForward, arrival_location,
                       get_special_station(special_station::kEnd),
                       get_time(end_segment_, tt_, event_type::kArr, state_.tbd_,
                                day_idx_[end_segment_]),
                       j.arrival_time(), *offset_it);
    j.legs_.push_back(get_run_leg(end_segment_));
  }
  else {
    auto arr_time = get_time(end_segment_,tt_,event_type::kArr,state_.tbd_,day_idx_[end_segment_]);
    auto const handle_fp = [&](footpath const& fp) {
      if (arr_time + fp.duration() != j.arrival_time() ||
          !has_offset(q.destination_, q.dest_match_mode_, fp.target())) {
        return false;
          }
      j.legs_.emplace_back(direction::kForward, arrival_location, fp.target(), arr_time,
                         j.arrival_time(), fp);
      j.legs_.push_back(get_run_leg(end_segment_));
      return true;
    };
    bool found_path = handle_fp(footpath{arrival_location, duration_t{0}});
    if(!found_path) {
      for (auto const fp :
             tt_.locations_.footpaths_out_[state_.tbd_.prf_idx_][arrival_location]) {
        if (handle_fp(fp)) {
          found_path = true;
          break;
        }
      }
    }
    utl::verify(found_path,"Reconstruction found no footpath to destination");
  }
  j.dest_ = j.legs_.back().to_;

  //intermediary legs
  segment_idx_t p = pred_[departure_segment];
  while(!is_invalid<segment_idx_t>(p)) {
    auto const [transport, arr_stop_idx, arr_l, arr_time] =
        get_transport_info(p, event_type::kArr);
    auto const fp = get_fp(arr_l, j.legs_.back().from_);
    j.legs_.emplace_back(direction::kForward, arr_l,
                                      j.legs_.back().from_, arr_time,
                                      arr_time + fp.duration(), fp);
    departure_segment = get_departure_segment(p);
    j.legs_.push_back(get_run_leg(p));
    p = pred_[departure_segment];
  }

  //First leg
  auto const start_time = j.start_time_;
  auto const first_dep_l = j.legs_.back().from_;
  auto const first_dep_time = j.legs_.back().dep_time_;
  if (q.start_match_mode_ == location_match_mode::kIntermodal) {
    auto const offset_it =
        utl::find_if(q.start_, [&](routing::offset const& o) {
          return o.target() == first_dep_l &&
                 start_time + o.duration() <= first_dep_time;
        });
    utl::verify(
        offset_it != end(q.start_),
        "Reconstruction failed because no start offset was found");
    j.legs_.emplace_back(
        direction::kForward, get_special_station(special_station::kStart),
        first_dep_l, first_dep_time - offset_it->duration(), first_dep_time,
        *offset_it);
  }
  else {
    for (auto const fp :
         tt_.locations_.footpaths_in_[state_.tbd_.prf_idx_][first_dep_l]) {
      if (start_time + fp.duration() <= first_dep_time &&
          has_offset(q.start_, q.start_match_mode_, fp.target())) {
        j.legs_.emplace_back(direction::kForward, fp.target(), first_dep_l,
                           first_dep_time - fp.duration(), first_dep_time, fp);
        break;
      }
    }
  }

  std::ranges::reverse(j.legs_);

}

}  // namespace nigiri::routing::tb::a_star
