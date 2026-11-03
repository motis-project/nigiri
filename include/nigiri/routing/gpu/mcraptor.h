#pragma once

#include <memory>
#include <optional>
#include <vector>

#include "nigiri/routing/clasz_mask.h"
#include "nigiri/routing/gpu/raptor.h"
#include "nigiri/routing/journey.h"
#include "nigiri/routing/limits.h"
#include "nigiri/routing/pareto_set.h"
#include "nigiri/routing/query.h"
#include "nigiri/routing/raptor/bmrap_bounds.h"
#include "nigiri/routing/raptor/mcraptor.h"
#include "nigiri/routing/raptor/raptor_stats.h"
#include "nigiri/routing/transfer_time_settings.h"
#include "nigiri/types.h"

namespace nigiri::routing::gpu {

// et tasks (boardable stops of marked routes) per round with a stop bag, the
// rest falls back to the sequential scan
inline constexpr auto kMcEtTaskMax = 1U << 22U;  // 4M: 210 MB of stop bags

// Preallocated capacities; unset = sized from the timetable. Running out of
// et task slots only slows a round down; a full stop bag, bag pool or
// breadcrumb arena (or the sequential scan's route bag) fails the search, since
// it would be lossy.
struct gpu_mcraptor_limits {
  std::uint32_t et_tasks_{kMcEtTaskMax};
  std::optional<std::uint32_t> bag_pool_blocks_{};
  std::optional<std::uint32_t> breadcrumbs_{};
};

// scope as mcraptor_supported(); the subtypes only select the criteria
struct gpu_mcraptor_state {
  using criteria_t = arr_criteria;

  explicit gpu_mcraptor_state(gpu_timetable const&,
                              gpu_mcraptor_limits const& = {});
  ~gpu_mcraptor_state();

  struct impl;
  std::unique_ptr<impl> impl_;
};

template <typename Criteria>
struct gpu_mcraptor_crit_state : gpu_mcraptor_state {
  using criteria_t = Criteria;
  using gpu_mcraptor_state::gpu_mcraptor_state;
};

using gpu_mcraptor_non_transit_state =
    gpu_mcraptor_crit_state<arr_non_transit_criteria>;
using gpu_mcraptor_avoided_modes_state =
    gpu_mcraptor_crit_state<arr_avoided_modes_criteria>;
using gpu_mcraptor_non_transit_avoided_modes_state =
    gpu_mcraptor_crit_state<arr_non_transit_avoided_modes_criteria>;
using gpu_mcraptor_mode_switches_state =
    gpu_mcraptor_crit_state<arr_mode_switches_criteria>;
using gpu_mcraptor_non_transit_mode_switches_state =
    gpu_mcraptor_crit_state<arr_non_transit_mode_switches_criteria>;
using gpu_mcraptor_avoided_modes_mode_switches_state =
    gpu_mcraptor_crit_state<arr_avoided_modes_mode_switches_criteria>;
using gpu_mcraptor_non_transit_avoided_modes_mode_switches_state =
    gpu_mcraptor_crit_state<
        arr_non_transit_avoided_modes_mode_switches_criteria>;

template <direction SearchDir, typename Criteria>
struct gpu_mcraptor {
  using algo_state_t = gpu_mcraptor_state;
  using algo_stats_t = raptor_stats;

  // unlike gpu raptor: without lb pruning the fixed-size device bags overflow
  static constexpr bool kUseLowerBounds = true;
  static constexpr auto const kDirIdx =
      SearchDir == direction::kForward ? 0U : 1U;

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
      bool no_compulsory_reservation,
      transfer_time_settings const& tts,
      profile_idx_t prf_idx,
      mc_context const& = {});

  raptor_stats get_stats() const { return stats_; }

  void reset_arrivals();
  void next_start_time();
  void add_start(location_idx_t, unixtime_t, via_offset_t = 0U);
  void set_td_start(td_offsets_t const& td) { start_.td_ = &td; }
  void set_start_stops(std::vector<offset> const& offsets,
                       location_match_mode const mode) {
    start_.offsets_ = &offsets;
    start_.mode_ = mode;
  }

  void execute(
      unixtime_t start_time,
      std::uint8_t max_transfers,
      std::array<unixtime_t, kMaxTransfers + 2U> const& worst_time_at_dest,
      pareto_set<journey>& results);

  void reconstruct(query const&, journey&);

  // Pong side: next_start_time() clears the bags instead of demoting them, so
  // only labels of the same departure (one merged anchor run) prune, as on
  // the CPU engine without range reuse; cross-anchor ones were seen
  // over-pruning.
  void set_reuse_same_dep() { reuse_same_dep_ = true; }

  // BM-RAPTOR pruning bounds, uploaded here; nullptr disables. See
  // basic_mcraptor::set_bounds().
  void set_bounds(bmrap_bounds const*);

private:
  date::sys_days base() const {
    return tt_.internal_interval_days().from_ + to_idx(base_) * date::days{1};
  }

  timetable const& tt_;
  rt_timetable const* rtt_;
  gpu_rt_timetable const* gpu_rtt_;
  gpu_mcraptor_state& state_;
  bitvec const& is_dest_;
  day_idx_t base_;
  raptor_stats stats_;
  clasz_mask_t allowed_claszes_;
  mc_context ctx_;
  bool is_wheelchair_;
  transfer_time_settings transfer_time_settings_;
  profile_idx_t prf_idx_;

  // search-window bound; the device dest frontier owns destination pruning
  delta_t worst_at_dest_;
  bool reuse_same_dep_{false};

  mc_start start_;
  std::vector<std::pair<location_idx_t, unixtime_t>> starts_;

  std::uint32_t bounds_n_locations_{0U};
  std::uint8_t bounds_budget_{0U};
  bool has_bounds_{false};
};

}  // namespace nigiri::routing::gpu
