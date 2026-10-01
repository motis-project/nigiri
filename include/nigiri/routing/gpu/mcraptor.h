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

// Same scope as the CPU mcraptor (mcraptor_supported). One state type serves
// every configuration; the subtypes only select it in the dispatch.
struct gpu_mcraptor_state {
  explicit gpu_mcraptor_state(gpu_timetable const&);
  ~gpu_mcraptor_state();

  struct impl;
  std::unique_ptr<impl> impl_;
};

struct gpu_mcraptor_non_transit_state : gpu_mcraptor_state {
  using gpu_mcraptor_state::gpu_mcraptor_state;
};

struct gpu_mcraptor_mode_filter_state : gpu_mcraptor_state {
  using gpu_mcraptor_state::gpu_mcraptor_state;
};

struct gpu_mcraptor_non_transit_mode_filter_state : gpu_mcraptor_state {
  using gpu_mcraptor_state::gpu_mcraptor_state;
};

// Device configurations, each equal to the matching CPU criteria; the extras
// fit the label's 16 bit slot (non_transit_mode_filter: minutes in bits 1..15,
// avoided-class flag (AIR) in bit 0).
enum class mc_crit : std::uint8_t {
  arr,
  non_transit,
  mode_filter,
  non_transit_mode_filter
};

template <direction SearchDir, mc_crit Crit>
struct gpu_mcraptor {
  using algo_state_t = gpu_mcraptor_state;
  using algo_stats_t = raptor_stats;

  // unlike gpu raptor: without lb pruning the fixed-size device bags overflow
  static constexpr bool kUseLowerBounds = true;
  static constexpr auto const kDirIdx =
      SearchDir == direction::kForward ? 0U : 1U;

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
      bool no_compulsory_reservation,
      transfer_time_settings const& tts,
      profile_idx_t prf_idx);

  raptor_stats get_stats() const { return stats_; }

  void reset_arrivals();
  void next_start_time();
  void add_start(location_idx_t, unixtime_t, via_offset_t = 0U);

  void execute(unixtime_t start_time,
               std::uint8_t max_transfers,
               unixtime_t worst_time_at_dest,
               pareto_set<journey>& results);

  // adds offset legs and the start footpath to the device's core legs
  void reconstruct(query const&, journey&);

  // Pong side: reuse-frontier rejections only against entries of the same
  // departure (one merged anchor run); cross-anchor ones were seen
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
  bool is_wheelchair_;
  transfer_time_settings transfer_time_settings_;
  profile_idx_t prf_idx_;

  // search-window bound; the device dest frontier owns destination pruning
  delta_t worst_at_dest_;
  bool reuse_same_dep_{false};

  std::vector<std::pair<location_idx_t, unixtime_t>> starts_;

  std::uint32_t bounds_n_locations_{0U};
  std::uint8_t bounds_budget_{0U};
  bool bounds_staggered_{false};
  bool has_bounds_{false};
};

}  // namespace nigiri::routing::gpu
