#include "nigiri/routing/raptor/mc_reconstruct.h"

#include <algorithm>

#include "utl/verify.h"

#include "nigiri/common/delta_t.h"
#include "nigiri/for_each_meta.h"
#include "nigiri/routing/journey.h"
#include "nigiri/routing/query.h"
#include "nigiri/routing/raptor/reconstruct.h"
#include "nigiri/rt/rt_timetable.h"
#include "nigiri/rt/run.h"
#include "nigiri/special_stations.h"
#include "nigiri/td_footpath.h"
#include "nigiri/timetable.h"

namespace nigiri::routing {

template <direction SearchDir>
void mc_add_legs(timetable const& tt,
                 rt_timetable const* rtt,
                 day_idx_t const base_day,
                 bool const is_intermodal_dest,
                 std::span<mc_breadcrumb const> chain,
                 journey& j) {
  constexpr auto const is_fwd = SearchDir == direction::kForward;
  constexpr auto const kArrEv = is_fwd ? event_type::kArr : event_type::kDep;
  constexpr auto const kDepEv = is_fwd ? event_type::kDep : event_type::kArr;
  constexpr auto const kIntermodalTarget =
      get_special_station(special_station::kEnd);
  auto const dir = [](auto const x) { return is_fwd ? x : -x; };
  auto const to_delta = [&](day_idx_t const day, int const mam) {
    return clamp((static_cast<int>(day.v_) - static_cast<int>(base_day.v_)) *
                     1440 +
                 mam);
  };
  auto const base = tt.internal_interval_days().from_ +
                    static_cast<int>(base_day.v_) * date::days{1};
  auto const to_unix = [&](delta_t const d) { return delta_to_unix(base, d); };
  auto const time_at_stop = [&](route_idx_t const r, transport const t,
                                stop_idx_t const stop_idx,
                                event_type const ev_type) {
    return to_delta(t.day_,
                    tt.event_mam(r, t.t_idx_, stop_idx, ev_type).count());
  };
  auto const rt_time_at_stop = [&](rt_transport_idx_t const rt_t,
                                   stop_idx_t const stop_idx,
                                   event_type const ev_type) {
    return to_delta(rtt->base_day_idx_,
                    rtt->event_time(rt_t, stop_idx, ev_type));
  };

  auto cur_l = j.dest_;
  for (auto const& bc : chain) {
    auto const cur_arr = bc.arr_;  // arrival at cur_l

    auto train_arr = delta_t{};
    auto dep_at_board = delta_t{};
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
      set_stops(rtt->rt_transport_location_seq_[rt_t]);
      train_arr = rt_time_at_stop(rt_t, bc.alight_, kArrEv);
      dep_at_board = rt_time_at_stop(rt_t, bc.board_, kDepEv);
      run.t_ = rtt->resolve_static(rt_t);
      run.rt_ = rt_t;
    } else {
      auto const t = transport{transport_idx_t{bc.transport_}, bc.day_};
      auto const r = tt.transport_route_[t.t_idx_];
      set_stops(tt.route_location_seq_[r]);
      train_arr = time_at_stop(r, t, bc.alight_, kArrEv);
      dep_at_board = time_at_stop(r, t, bc.board_, kDepEv);
      run.t_ = t;
    }

    if (is_intermodal_dest && cur_l == kIntermodalTarget) {
      // the last mile is added by mc_reconstruct_offsets
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
  }

  if constexpr (is_fwd) {
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

template <direction SearchDir>
void mc_reconstruct_offsets(timetable const& tt,
                            rt_timetable const* rtt,
                            query const& q,
                            journey& j) {
  constexpr auto const is_fwd = SearchDir == direction::kForward;

  add_offset_legs<SearchDir>(tt, rtt, q, j);
  if constexpr (is_fwd) {
    specify_td_offsets(q, j);
  } else {
    // legs are chronological, q is in search direction
    auto journey_q = q;
    journey_q.flip_dir();
    specify_td_offsets(journey_q, j);
  }

  // td footpath legs: walking time only, the wait is spent at the stop
  if (rtt != nullptr && q.prf_idx_ != 0U) {
    auto const& has_td = is_fwd ? rtt->has_td_footpaths_in_[q.prf_idx_]
                                : rtt->has_td_footpaths_out_[q.prf_idx_];
    auto const& td_fps = is_fwd ? rtt->td_footpaths_in_[q.prf_idx_]
                                : rtt->td_footpaths_out_[q.prf_idx_];
    if (!td_fps.empty()) {
      for (auto z = std::size_t{0U}; z != j.legs_.size(); ++z) {
        auto& lg = j.legs_[z];
        if (!std::holds_alternative<footpath>(lg.uses_)) {
          continue;
        }
        auto const key_l = is_fwd ? lg.to_ : lg.from_;
        auto const target_l = is_fwd ? lg.from_ : lg.to_;
        if (!has_td.test(key_l)) {
          continue;
        }
        if (is_fwd || z == 0U) {
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
        } else {
          // backward: walk as soon as possible after the previous leg, as
          // raptor (the anchor above is the next departure)
          auto const t = j.legs_[z - 1U].arr_time_;
          for_each_footpath<direction::kForward>(
              td_fps[key_l], t,
              [&](location_idx_t const fp_target, duration_t const dur,
                  duration_t const walk) {
                if (fp_target != target_l) {
                  return utl::cflow::kContinue;
                }
                lg.dep_time_ = t + dur - walk;
                lg.arr_time_ = t + dur;
                lg.uses_ = footpath{lg.to_, walk};
                return utl::cflow::kBreak;
              });
        }
      }
    }
  }
}

template void mc_add_legs<direction::kForward>(timetable const&,
                                               rt_timetable const*,
                                               day_idx_t,
                                               bool,
                                               std::span<mc_breadcrumb const>,
                                               journey&);
template void mc_add_legs<direction::kBackward>(timetable const&,
                                                rt_timetable const*,
                                                day_idx_t,
                                                bool,
                                                std::span<mc_breadcrumb const>,
                                                journey&);
template void mc_reconstruct_offsets<direction::kForward>(timetable const&,
                                                          rt_timetable const*,
                                                          query const&,
                                                          journey&);
template void mc_reconstruct_offsets<direction::kBackward>(timetable const&,
                                                           rt_timetable const*,
                                                           query const&,
                                                           journey&);

}  // namespace nigiri::routing
