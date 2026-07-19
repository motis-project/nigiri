#pragma once

#include <array>
#include <memory>
#include <vector>

#include "nigiri/common/delta_t.h"
#include "nigiri/routing/clasz_mask.h"
#include "nigiri/routing/journey.h"
#include "nigiri/routing/limits.h"
#include "nigiri/routing/pareto_set.h"
#include "nigiri/routing/query.h"
#include "nigiri/routing/raptor/raptor_stats.h"
#include "nigiri/routing/raptor/reconstruct.h"
#include "nigiri/routing/transfer_time_settings.h"
#include "nigiri/rt/rt_timetable.h"
#include "nigiri/special_stations.h"
#include "nigiri/timetable.h"
#include "nigiri/types.h"

namespace nigiri::routing::gpu {

inline bool gpu_supported(query const& q, rt_timetable const* = nullptr) {
  return q.via_stops_.empty();
}

struct gpu_timetable {
  explicit gpu_timetable(timetable const&);
  ~gpu_timetable();

  struct impl;
  std::unique_ptr<impl> impl_;
};

struct gpu_rt_timetable {
  gpu_rt_timetable(timetable const&, rt_timetable const&);
  ~gpu_rt_timetable();

  struct impl;
  std::unique_ptr<impl> impl_;
};

// Type-erased version for rt_timetable
std::unique_ptr<void, void (*)(void*)> make_gpu_rtt(timetable const&,
                                                    rt_timetable const&);

struct gpu_raptor_state {
  explicit gpu_raptor_state(gpu_timetable const&);
  ~gpu_raptor_state();

  struct impl;
  std::unique_ptr<impl> impl_;
};

// Ping-bounds pruning (see pong.cc): monotone-fill the ping search's device
// round_times into the state's device bounds buffer (16-bit times, row k =
// best over rounds <= k). Launches on the state's stream -- ordered before
// the pong's reset_arrivals wipes the shared round_times, no sync needed.
// Returns the device pointer to hand to gpu_raptor::set_bounds.
// Stops whose ping-direction footpaths are time-dependent (rtt != nullptr,
// prf_idx != 0) get a pass-everything bound instead (see the CPU
// fill_bounds in pong.cc for the reasoning).
template <direction SearchDir>
delta_t const* fill_bounds(gpu_raptor_state&,
                           std::size_t n_rows,
                           rt_timetable const* rtt,
                           profile_idx_t prf_idx);

template <direction SearchDir>
struct gpu_raptor {
  using algo_state_t = gpu_raptor_state;
  using algo_stats_t = raptor_stats;

  static constexpr bool kUseLowerBounds = false;
  static constexpr auto const kDirIdx =
      SearchDir == direction::kForward ? 0U : 1U;

  gpu_raptor(
      timetable const& tt,
      rt_timetable const* rtt,
      gpu_raptor_state& state,
      bitvec& is_dest,
      std::array<bitvec, kMaxVias> const& is_via,
      std::vector<std::uint16_t> const& dist_to_dest,
      hash_map<location_idx_t, std::vector<td_offset>> const& td_dist_to_dest,
      std::vector<std::uint16_t> const& lb,
      std::vector<via_stop> const& via_stops,
      day_idx_t const base,
      clasz_mask_t const allowed_claszes,
      bool const require_bike_transport,
      bool const require_car_transport,
      bool const is_wheelchair,
      transfer_time_settings const& tts);

  raptor_stats get_stats() const { return stats_; }
  void reset_arrivals();
  void next_start_time();

  // === ping-bounds pruning (see pong.cc) ==================================
  // bounds = device pointer from gpu::fill_bounds; a label written in round
  // k is checked against the ping's bound row last_round - k (direct +
  // footpath rescue) and pruned if no ping journey can complete it.
  void set_bounds(delta_t const* const bounds,
                  unsigned const last_round,
                  profile_idx_t const prf_idx) {
    bounds_ = bounds;
    bounds_last_k_ = last_round;
    bounds_prf_idx_ = prf_idx;
  }

  // Loose pruning keeps labels that merely *equal* the current time at
  // destination (the ping runs loose so its round_times cover every stop of
  // an equal-arrival/later-departure journey the pong needs).
  void set_loose_pruning(bool const loose) { loose_pruning_ = loose; }

  void add_start(location_idx_t, unixtime_t);

  // Component lower bounds (pong ping side only): runs the time-independent
  // component RAPTOR towards this search's destinations on the GPU and
  // enables per-transfer pruning in execute(). No-op if the component graph
  // is not uploaded (NIGIRI_GPU_PING_LB=0) or the timetable has realtime
  // transports (rt trips may undercut the static minimum durations).
  // Call once per query (destinations don't change between start times).
  void compute_component_bounds(std::uint16_t max_travel_time);

  // Ad-hoc transfer patterns (pong ping side only): mark l as a transfer
  // stop of a previously found connection. execute() then runs a restricted
  // prelude search (enter/exit only at marked stops) whose real arrivals
  // seed time_at_dest_ for early pruning. No-op with NIGIRI_GPU_ADHOC_TP=0.
  void add_adhoc_stop(location_idx_t);

  void execute(unixtime_t start_time,
               std::uint8_t max_transfers,
               unixtime_t worst_time_at_dest,
               profile_idx_t prf_idx,
               pareto_set<journey>& results);

  void reconstruct(query const&, journey&);

private:
  date::sys_days base() const {
    return tt_.internal_interval_days().from_ + to_idx(base_) * date::days{1};
  }

  timetable const& tt_;
  rt_timetable const* rtt_;
  gpu_rt_timetable const* gpu_rtt_;
  std::uint32_t n_locations_;
  gpu_raptor_state& state_;
  bitvec const& is_dest_;
  day_idx_t base_;
  raptor_stats stats_;
  clasz_mask_t allowed_claszes_;
  bool require_bike_transport_;
  bool require_car_transport_;
  bool is_wheelchair_;
  transfer_time_settings transfer_time_settings_;

  delta_t const* bounds_{nullptr};
  unsigned bounds_last_k_{0U};
  profile_idx_t bounds_prf_idx_{0U};
  bool loose_pruning_{false};

  std::vector<std::pair<location_idx_t, unixtime_t>> starts_;

  bool use_lb_{false};  // component bounds computed for this query
  bool adhoc_dirty_{false};  // host adhoc bits changed since last upload
  bool adhoc_any_{false};  // at least one adhoc transfer stop marked
  bitvec adhoc_tp_;  // host accumulator, uploaded on demand
};

}  // namespace nigiri::routing::gpu