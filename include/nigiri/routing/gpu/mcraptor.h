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
#include "nigiri/routing/raptor/raptor_stats.h"
#include "nigiri/routing/transfer_time_settings.h"
#include "nigiri/types.h"

struct CUstream_st;  // cudaStream_t

namespace nigiri::routing::gpu {

// Device configurations, one per CPU criteria configuration: a set of the
// non_transit, mode_filter and mode_switches criteria, packed into the label's
// 16 bit extras slot (see mc_extras in mc_criteria.h).
enum class mc_crit : std::uint8_t {
  arr = 0U,
  non_transit = 1U,
  mode_filter = 2U,
  non_transit_mode_filter = 3U,
  mode_switches = 4U,
  non_transit_mode_switches = 5U,
  mode_filter_mode_switches = 6U,
  non_transit_mode_filter_mode_switches = 7U
};

constexpr bool has(mc_crit const c, mc_crit const part) {
  return (static_cast<std::uint8_t>(c) & static_cast<std::uint8_t>(part)) != 0U;
}

// et tasks (boardable stops of marked routes) per round with a stop bag; the
// others are looked up by the sequential scan
inline constexpr auto kMcEtTaskMax = 1U << 22U;  // 4M: 210 MB of stop bags

// labels of one guide copy (take_guide(), one per direction), 6 bytes each
inline constexpr auto kMcGuideLabels = 1U << 23U;  // 8M: 48 MB

// Preallocated capacities; unset = sized from the timetable. Running out of
// et task slots only slows a round down; a full stop bag, bag pool or
// breadcrumb arena (or the sequential scan's route bag) fails the search, since
// it would be lossy.
struct gpu_mcraptor_limits {
  std::uint32_t et_tasks_{kMcEtTaskMax};
  std::optional<std::uint32_t> bag_pool_blocks_{};
  std::optional<std::uint32_t> breadcrumbs_{};
  std::uint32_t guide_labels_{kMcGuideLabels};
};

// The owning scalar state's per-round scratch (n_route_stops entries each)
// and stream, borrowed instead of allocated (gpu_raptor_state::try_mc_state()):
// both engines rewrite the scratch every round before reading it, and on one
// stream their kernels never overlap.
struct gpu_mcraptor_scratch {
  std::uint32_t* et_result_;
  std::uint32_t* et_task_list_;
  CUstream_st* stream_;
};

// Same scope as the CPU mcraptor (mcraptor_supported). One state type serves
// every configuration; the subtypes only select it in the dispatch (kCrit).
struct gpu_mcraptor_state {
  static constexpr auto kCrit = mc_crit::arr;

  explicit gpu_mcraptor_state(
      gpu_timetable const&,
      gpu_mcraptor_limits const& = {},
      std::optional<gpu_mcraptor_scratch> const& = std::nullopt);
  ~gpu_mcraptor_state();

  struct impl;
  std::unique_ptr<impl> impl_;
};

template <mc_crit Crit>
struct gpu_mcraptor_crit_state : gpu_mcraptor_state {
  static constexpr auto kCrit = Crit;
  using gpu_mcraptor_state::gpu_mcraptor_state;
};

using gpu_mcraptor_non_transit_state =
    gpu_mcraptor_crit_state<mc_crit::non_transit>;
using gpu_mcraptor_mode_filter_state =
    gpu_mcraptor_crit_state<mc_crit::mode_filter>;
using gpu_mcraptor_non_transit_mode_filter_state =
    gpu_mcraptor_crit_state<mc_crit::non_transit_mode_filter>;
using gpu_mcraptor_mode_switches_state =
    gpu_mcraptor_crit_state<mc_crit::mode_switches>;
using gpu_mcraptor_non_transit_mode_switches_state =
    gpu_mcraptor_crit_state<mc_crit::non_transit_mode_switches>;
using gpu_mcraptor_mode_filter_mode_switches_state =
    gpu_mcraptor_crit_state<mc_crit::mode_filter_mode_switches>;
using gpu_mcraptor_non_transit_mode_filter_mode_switches_state =
    gpu_mcraptor_crit_state<mc_crit::non_transit_mode_filter_mode_switches>;

template <direction SearchDir, mc_crit Crit>
struct gpu_mcraptor {
  using algo_state_t = gpu_mcraptor_state;
  using algo_stats_t = raptor_stats;

  // unlike gpu raptor: without lb pruning the fixed-size device bags overflow
  static constexpr bool kUseLowerBounds = true;
  // ... unless bounded (bmrap_profile.cc): the BM-RAPTOR bounds prune harder,
  // so it runs on an all-zero lb, as basic_mcraptor's effective_lb()
  static constexpr bool kNeedsLbWhenBounded = false;
  static constexpr auto const kDirIdx =
      SearchDir == direction::kForward ? 0U : 1U;

  static constexpr bool kHasNonTransit = has(Crit, mc_crit::non_transit);
  static constexpr bool kHasModeFilter = has(Crit, mc_crit::mode_filter);
  static constexpr bool kHasModeSwitches = has(Crit, mc_crit::mode_switches);

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
      profile_idx_t prf_idx);

  raptor_stats get_stats() const { return stats_; }

  void reset_arrivals();
  void next_start_time();
  void add_start(location_idx_t, unixtime_t, via_offset_t = 0U);
  void set_td_start(td_offsets_t const& td) { td_start_ = &td; }

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

  // Guided searches (bmrap_profile.cc): take_guide() copies this search's
  // labels per stop into the state, one copy per direction; a later search of
  // the other direction with set_guide() keeps only the labels that join one
  // of them into one of `targets` (trips and criteria within, catchable at the
  // stop). Empty targets, or more than the device buffer holds: not guided. A
  // copy beyond gpu_mcraptor_limits::guide_labels_ fails the guided search.
  void take_guide();
  void set_guide(std::vector<journey const*> const& targets);

  // BM-RAPTOR pruning bounds, read in place from the scalar state that built
  // them (gpu_raptor::build_reach_bounds); nullptr disables. See
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
  clasz_mask_t avoided_;  // mode_filter_dim::avoided_mask() at construction
  bool is_wheelchair_;
  transfer_time_settings transfer_time_settings_;
  profile_idx_t prf_idx_;

  // search-window bound; the device dest frontier owns destination pruning
  delta_t worst_at_dest_;
  bool reuse_same_dep_{false};

  td_offsets_t const* td_start_{nullptr};
  std::vector<std::pair<location_idx_t, unixtime_t>> starts_;

  delta_t const* bounds_{nullptr};
  std::size_t bounds_size_{0U};
  std::uint32_t bounds_n_locations_{0U};
  std::uint8_t bounds_budget_{0U};
  bool bounds_staggered_{false};
  bool has_bounds_{false};
  std::uint32_t n_guide_targets_{0U};
};

}  // namespace nigiri::routing::gpu
