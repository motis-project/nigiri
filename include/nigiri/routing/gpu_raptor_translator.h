#pragma once

#include <cinttypes>
#include "nigiri/routing/gpu_raptor_state.h"
#include "nigiri/routing/gpu_timetable.h"
#include "nigiri/routing/pareto_set.h"
#include "nigiri/routing/journey.h"
#include "nigiri/routing/query.h"
#include "nigiri/routing/raptor/raptor_state.h"
#include "nigiri/rt/rt_timetable.h"
#include "clasz_mask.h"
#include "nigiri/timetable.h"
#include "nigiri/routing/raptor/debug.h"
#include "nigiri/common/delta_t.h"
#include "nigiri/routing/gpu_raptor.h"
#include "nigiri/routing/gpu_types.h"
#include "nigiri/routing/raptor/raptor.h"
#include "nigiri/rt/frun.h"

using namespace date;
using namespace nigiri;
using namespace nigiri::routing;

template <gpu_direction SearchDir, bool Rt>
struct gpu_raptor;


template <nigiri::direction SearchDir, bool Rt>
struct gpu_raptor_translator {
  static constexpr auto const kInvalid = nigiri::kInvalidDelta<SearchDir>;
  static constexpr bool kUseLowerBounds = true;
  using algo_state_t = mem;
  using algo_stats_t = gpu_raptor_stats;
  static nigiri::direction const cpu_direction_ = SearchDir;
  static gpu_direction const gpu_direction_ =
      static_cast<enum gpu_direction const>(cpu_direction_);
  std::variant<
      std::unique_ptr<gpu_raptor<gpu_direction::kForward, true>>,
      std::unique_ptr<gpu_raptor<gpu_direction::kForward, false>>,
      std::unique_ptr<gpu_raptor<gpu_direction::kBackward, true>>,
      std::unique_ptr<gpu_raptor<gpu_direction::kBackward, false>>
      > gpu_r_;
  gpu_timetable const* gtt_;

  gpu_raptor_translator(
      nigiri::timetable const& tt,
      nigiri::rt_timetable const* rtt,
      gpu_timetable const* gtt,
      algo_state_t & state,
      std::vector<uint8_t>& is_dest,
      std::vector<std::uint16_t>& dist_to_dest,
      std::vector<std::uint16_t>& lb,
      nigiri::day_idx_t const base,
      nigiri::routing::clasz_mask_t const allowed_claszes)
      : tt_{tt},
        rtt_{rtt},
        gtt_{gtt},
        state_{state},
        is_dest_{is_dest},
        dist_to_end_{dist_to_dest},
        lb_{lb},
        base_{base},
        allowed_claszes_{allowed_claszes}{
    auto& gpu_base = *reinterpret_cast<gpu_day_idx_t*>(&base_);
    auto& gpu_allowed_claszes = *reinterpret_cast<gpu_clasz_mask_t*>(&allowed_claszes_);
    gpu_r_ = std::make_unique<gpu_raptor<gpu_direction_,Rt>>(gtt_,state_, is_dest_,dist_to_end_, lb_, gpu_base, gpu_allowed_claszes,tt_.internal_interval_days().size().count());
  }

  algo_stats_t get_stats() {
    if (gpu_direction_ == gpu_direction::kForward && Rt == true) {
      return get<std::unique_ptr<gpu_raptor<gpu_direction::kForward,true>>>(gpu_r_)->get_stats();
    } else if (gpu_direction_ == gpu_direction::kForward && Rt == false) {
      return get<std::unique_ptr<gpu_raptor<gpu_direction::kForward,false>>>(gpu_r_)->get_stats();
    } else if (gpu_direction_ == gpu_direction::kBackward && Rt == true) {
      return get<std::unique_ptr<gpu_raptor<gpu_direction::kBackward,true>>>(gpu_r_)->get_stats();
    } else if (gpu_direction_ == gpu_direction::kBackward && Rt == false) {
      return get<std::unique_ptr<gpu_raptor<gpu_direction::kBackward,false>>>(gpu_r_)->get_stats();
    }
  }

  void reset_arrivals() {
    if (gpu_direction_ == gpu_direction::kForward && Rt == true) {
      get<std::unique_ptr<gpu_raptor<gpu_direction::kForward,true>>>(gpu_r_)->reset_arrivals();
    } else if (gpu_direction_ == gpu_direction::kForward && Rt == false) {
      get<std::unique_ptr<gpu_raptor<gpu_direction::kForward,false>>>(gpu_r_)->reset_arrivals();
    } else if (gpu_direction_ == gpu_direction::kBackward && Rt == true) {
      get<std::unique_ptr<gpu_raptor<gpu_direction::kBackward,true>>>(gpu_r_)->reset_arrivals();
    } else if (gpu_direction_ == gpu_direction::kBackward && Rt == false) {
      get<std::unique_ptr<gpu_raptor<gpu_direction::kBackward,false>>>(gpu_r_)->reset_arrivals();
    }
  }

  void next_start_time() {
    if (gpu_direction_ == gpu_direction::kForward && Rt == true) {
      get<std::unique_ptr<gpu_raptor<gpu_direction::kForward,true>>>(gpu_r_)->next_start_time();
    } else if (gpu_direction_ == gpu_direction::kForward && Rt == false) {
      get<std::unique_ptr<gpu_raptor<gpu_direction::kForward,false>>>(gpu_r_)->next_start_time();
    } else if (gpu_direction_ == gpu_direction::kBackward && Rt == true) {
      get<std::unique_ptr<gpu_raptor<gpu_direction::kBackward,true>>>(gpu_r_)->next_start_time();
    } else if (gpu_direction_ == gpu_direction::kBackward && Rt == false) {
      get<std::unique_ptr<gpu_raptor<gpu_direction::kBackward,false>>>(gpu_r_)->next_start_time();
    }
  }

  void add_start(nigiri::location_idx_t const l,
                 nigiri::unixtime_t const t) {
    trace_upd("adding start {}: {}\n", location{tt_, l}, t);
    auto gpu_l = *reinterpret_cast<const gpu_location_idx_t*>(&l);
    gpu_unixtime_t gpu_t = *reinterpret_cast<gpu_unixtime_t const*>(&t);
    if (gpu_direction_ == gpu_direction::kForward && Rt == true) {
      get<std::unique_ptr<gpu_raptor<gpu_direction::kForward,true>>>(gpu_r_)->add_start(gpu_l,gpu_t);
    } else if (gpu_direction_ == gpu_direction::kForward && Rt == false) {
      get<std::unique_ptr<gpu_raptor<gpu_direction::kForward,false>>>(gpu_r_)->add_start(gpu_l,gpu_t);
    } else if (gpu_direction_ == gpu_direction::kBackward && Rt == true) {
      get<std::unique_ptr<gpu_raptor<gpu_direction::kBackward,true>>>(gpu_r_)->add_start(gpu_l,gpu_t);
    } else if (gpu_direction_ == gpu_direction::kBackward && Rt == false) {
      get<std::unique_ptr<gpu_raptor<gpu_direction::kBackward,false>>>(gpu_r_)->add_start(gpu_l,gpu_t);
    }
  }

  void execute(
      unixtime_t const start_time,
      uint8_t const max_transfers,
      unixtime_t const worst_time_at_dest,
      profile_idx_t const prf_idx,
      nigiri::pareto_set<journey>& results) {
    get_gpu_roundtimes(start_time,max_transfers,worst_time_at_dest,prf_idx);
    auto const end_k = std::min(max_transfers, kMaxTransfers) + 1U;
    for (auto i = 0U; i != tt_.n_locations(); ++i) {
      auto const is_dest = is_dest_[i];
      if (!is_dest) {
        continue;
      }
      for (auto k = 1U; k != end_k; ++k) {
        auto const dest_time = state_.host_.round_times_[k*state_.host_.column_count_round_times_ + i];
        if (dest_time != kInvalid) {
          trace("ADDING JOURNEY: start={}, dest={} @ {}, transfers={}\n",
                start_time, delta_to_unix(base(), state_.round_times_[k][i]),
                location{tt_, location_idx_t{i}}, k - 1);
          auto const [optimal, it, dominated_by] = results.add(
              journey{.legs_ = {},
                      .start_time_ = start_time,
                      .dest_time_ = delta_to_unix(base(), dest_time),
                      .dest_ = location_idx_t{i},
                      .transfers_ = static_cast<std::uint8_t>(k - 1)});
          if (!optimal) {
            trace("  DOMINATED BY: start={}, dest={} @ {}, transfers={}\n",
                  dominated_by->start_time_, dominated_by->dest_time_,
                  location{tt_, dominated_by->dest_}, dominated_by->transfers_);
          }
        }
      }
    }
  }

  void reconstruct(const query& q,
                   journey& j){
    reconstruct_journey_gpu<SearchDir>(tt_, rtt_, q, state_, j, base(), base_);
  }
  nigiri::timetable const& tt_;
  nigiri::rt_timetable const* rtt_{nullptr};
  algo_state_t& state_;
  std::vector<uint8_t>& is_dest_;
  std::vector<std::uint16_t>& dist_to_end_;
  std::vector<std::uint16_t>& lb_;
  nigiri::day_idx_t base_;
  nigiri::routing::clasz_mask_t allowed_claszes_;
private:
  inline int translator_as_int(day_idx_t const d)  { return static_cast<int>(d.v_); };
  date::sys_days base(){
    return tt_.internal_interval_days().from_ +translator_as_int(base_) * date::days{1};
  }
  void get_gpu_roundtimes(
      nigiri::unixtime_t const start_time,
      uint8_t const max_transfers,
      nigiri::unixtime_t const worst_time_at_dest,
      nigiri::profile_idx_t const prf_idx) {
    auto& gpu_start_time = *reinterpret_cast<gpu_unixtime_t const*>(&start_time);

    auto& gpu_worst_time_at_dest = *reinterpret_cast<gpu_unixtime_t const*>(&worst_time_at_dest);;

    auto& gpu_prf_idx = *reinterpret_cast<gpu_profile_idx_t const*>(&prf_idx);

    if (gpu_direction_ == gpu_direction::kForward && Rt == true) {
      get<std::unique_ptr<gpu_raptor<gpu_direction::kForward,true>>>(gpu_r_)->execute(gpu_start_time,max_transfers,gpu_worst_time_at_dest,prf_idx);
    } else if (gpu_direction_ == gpu_direction::kForward && Rt == false) {
      get<std::unique_ptr<gpu_raptor<gpu_direction::kForward,false>>>(gpu_r_)->execute(gpu_start_time,max_transfers,gpu_worst_time_at_dest,prf_idx);
    } else if (gpu_direction_ == gpu_direction::kBackward && Rt == true) {
      get<std::unique_ptr<gpu_raptor<gpu_direction::kBackward,true>>>(gpu_r_)->execute(gpu_start_time,max_transfers,gpu_worst_time_at_dest,prf_idx);
    } else if (gpu_direction_ == gpu_direction::kBackward && Rt == false) {
      get<std::unique_ptr<gpu_raptor<gpu_direction::kBackward,false>>>(gpu_r_)->execute(gpu_start_time,max_transfers,gpu_worst_time_at_dest,prf_idx);
    }
  }
};

static gpu_timetable* translate_tt_in_gtt(nigiri::timetable tt) {

  gpu_locations locations_ = gpu_locations(
      reinterpret_cast<gpu_vector_map<gpu_location_idx_t, gpu_u8_minutes>*>(
          &tt.locations_.transfer_time_),
      reinterpret_cast<gpu_vecvec<gpu_location_idx_t, nigiri::gpu_footpath>*>(
          &tt.locations_.footpaths_out_),
      reinterpret_cast<gpu_vecvec<gpu_location_idx_t, nigiri::gpu_footpath>*>(
          &tt.locations_.footpaths_in_));

  uint32_t n_locations = tt.n_locations();;
  uint32_t n_routes = tt.n_routes();
  auto gtt_stop_time_ranges =
      reinterpret_cast<gpu_vector_map<gpu_route_idx_t,
                                      nigiri::gpu_interval<std::uint32_t>>*>(
          &tt.route_stop_time_ranges_);
  auto gtt = create_gpu_timetable(
      reinterpret_cast<gpu_delta*>(tt.route_stop_times_.data()),
      tt.route_stop_times_.size(),
      reinterpret_cast<gpu_vecvec<gpu_route_idx_t, gpu_value_type>*>(
          &tt.route_location_seq_),
      reinterpret_cast<gpu_vecvec<gpu_location_idx_t, gpu_route_idx_t>*>(
          &tt.location_routes_),
      n_locations,
      n_routes,
      gtt_stop_time_ranges,
      reinterpret_cast<gpu_vector_map<
          gpu_route_idx_t, nigiri::gpu_interval<gpu_transport_idx_t>>*>(
          &tt.route_transport_ranges_),
      reinterpret_cast<gpu_vector_map<gpu_bitfield_idx_t,gpu_bitfield>*>(
          &tt.bitfields_),
      reinterpret_cast<
          gpu_vector_map<gpu_transport_idx_t, gpu_bitfield_idx_t>*>(
          &tt.transport_traffic_days_),
      reinterpret_cast<nigiri::gpu_interval<gpu_sys_days>*>(&tt.date_range_),
      &locations_,
      reinterpret_cast<gpu_vector_map<gpu_route_idx_t, gpu_clasz>*>(
          &tt.route_clasz_));
  return gtt;
}


