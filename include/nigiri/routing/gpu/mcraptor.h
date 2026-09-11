#pragma once

#include <memory>
#include <vector>

#include "nigiri/routing/clasz_mask.h"
#include "nigiri/routing/gpu/raptor.h"
#include "nigiri/routing/journey.h"
#include "nigiri/routing/limits.h"
#include "nigiri/routing/pareto_set.h"
#include "nigiri/routing/query.h"
#include "nigiri/routing/raptor/bmrap_bounds.h"
#include "nigiri/routing/raptor/raptor_stats.h"
#include "nigiri/routing/transfer_time_settings.h"
#include "nigiri/types.h"

namespace nigiri::routing::gpu {

// GPU McRAPTOR supports the CPU mcraptor scope (see mcraptor_supported):
// no via stops and no bike/car transport requirements. Realtime (rt
// transports + time-dependent footpaths), time-dependent first/last-mile
// offsets, wheelchair, clasz filters and transfer time settings are
// supported.
//
// One state serves every criteria configuration (the packed 64-bit label is
// the same width regardless); the distinct types select the algorithm in the
// raptor_search/benchmark dispatch.
struct gpu_mcraptor_state {
  explicit gpu_mcraptor_state(gpu_timetable const&);
  ~gpu_mcraptor_state();

  struct impl;
  std::unique_ptr<impl> impl_;
};

struct gpu_mcraptor_cost_state : gpu_mcraptor_state {
  using gpu_mcraptor_state::gpu_mcraptor_state;
};

// The label configurations the device mcraptor implements, each result-equal
// to the CPU mcraptor with the matching criteria (see mc_crit_of in
// bmrap_common.h). They all fit the 16-bit "crit" slot of the packed label:
//   arr                      arrival only (crit == 0 everywhere)
//   cost                     arrival + generalized cost (arr_cost_criteria)
//   non_transit              arrival + minutes on foot (arr_with<non_transit>)
//   mode_filter              arrival + "uses an avoided class" bit (AIR)
//   non_transit_mode_filter  both of the above  (nt in bits 1..15, mf in bit 0)
enum class mc_crit : std::uint8_t {
  arr,
  cost,
  non_transit,
  mode_filter,
  non_transit_mode_filter
};

template <direction SearchDir, mc_crit Crit>
struct gpu_mcraptor {
  using algo_state_t = std::conditional_t<Crit == mc_crit::cost,
                                          gpu_mcraptor_cost_state,
                                          gpu_mcraptor_state>;
  using algo_stats_t = raptor_stats;

  // unlike the single-criterion GPU raptor, mcraptor DOES use lower
  // bounds: the lb-projected destination pruning is what keeps the pareto
  // bags small (without it, labels the CPU never stores flood the
  // fixed-capacity device bags). lb affects pruning only, never results.
  static constexpr bool kUseLowerBounds = true;
  static constexpr auto const kDirIdx =
      SearchDir == direction::kForward ? 0U : 1U;

  static constexpr bool kHasCost = Crit == mc_crit::cost;
  static constexpr bool kHasNonTransit =
      Crit == mc_crit::non_transit || Crit == mc_crit::non_transit_mode_filter;
  static constexpr bool kHasModeFilter =
      Crit == mc_crit::mode_filter || Crit == mc_crit::non_transit_mode_filter;

  gpu_mcraptor(
      timetable const& tt,
      rt_timetable const* rtt,
      gpu_mcraptor_state& state,
      bitvec& is_dest,
      std::array<bitvec, kMaxVias> const& is_via,
      std::vector<std::uint16_t> const& dist_to_dest,
      hash_map<location_idx_t, std::vector<td_offset>> const& td_dist_to_dest,
      std::vector<std::uint16_t> const& lb,
      std::vector<via_stop> const& via_stops,
      day_idx_t base,
      clasz_mask_t allowed_claszes,
      bool require_bike_transport,
      bool require_car_transport,
      bool is_wheelchair,
      transfer_time_settings const& tts);

  raptor_stats get_stats() const { return stats_; }

  // pong-side engines: reuse-frontier rejections restricted to entries
  // of the SAME departure (= the same merged anchor run). Cross-anchor
  // rejections were observed to over-prune without a real dominating
  // journey behind them (q#45 trace, 2026-07-11).
  void set_reuse_same_dep() { reuse_same_dep_ = true; }

  // tight starts (pong ping): re-anchor collected journeys at their
  // latest feasible departure instead of the step start, so the result
  // pareto prices real dep-normalized cost - the one-step ping window
  // otherwise collapses cost-pareto variants under phantom waiting
  // (see basic_mcraptor::set_tight_start). The device reconstruct
  // reports the shift per journey (gpu_journey::start_shift_).
  void set_tight_start() { tight_start_ = true; }

  // BM-RAPTOR: tau_dep^<- / tau_arr^-> pruning, uploaded to the device
  // here; nullptr disables it. Mirrors basic_mcraptor::set_bounds().
  void set_bounds(bmrap_bounds const*);

  void reset_arrivals();
  void next_start_time();
  void add_start(location_idx_t, unixtime_t);

  void execute(unixtime_t start_time,
               std::uint8_t max_transfers,
               unixtime_t worst_time_at_dest,
               profile_idx_t prf_idx,
               pareto_set<journey>& results);

  // Core legs are materialized by the device breadcrumb chase; this only
  // adds first/last-mile offset legs and the start footpath (host side,
  // where the query offsets live).
  void reconstruct(query const&, journey&);

private:
  date::sys_days base() const {
    return tt_.internal_interval_days().from_ + to_idx(base_) * date::days{1};
  }

  timetable const& tt_;
  rt_timetable const* rtt_;
  gpu_rt_timetable const* gpu_rtt_;
  std::uint32_t n_locations_;
  gpu_mcraptor_state& state_;
  bitvec const& is_dest_;
  day_idx_t base_;
  raptor_stats stats_;
  clasz_mask_t allowed_claszes_;
  bool is_wheelchair_;
  transfer_time_settings transfer_time_settings_;

  // pure search-window bound in delta units, persisted across start times
  // within one query (never journey-tightened - the device dest frontier
  // owns all destination pruning)
  delta_t worst_at_dest_;
  bool reuse_same_dep_{false};
  bool tight_start_{false};  // see set_tight_start()

  std::vector<std::pair<location_idx_t, unixtime_t>> starts_;

  std::uint32_t bounds_n_locations_{0U};
  std::uint8_t bounds_budget_{0U};
  bool has_bounds_{false};
};

}  // namespace nigiri::routing::gpu
