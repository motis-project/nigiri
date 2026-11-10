#pragma once

#include <memory>
#include <vector>

#include "nigiri/routing/clasz_mask.h"
#include "nigiri/routing/gpu/raptor.h"
#include "nigiri/routing/journey.h"
#include "nigiri/routing/limits.h"
#include "nigiri/routing/pareto_set.h"
#include "nigiri/routing/query.h"
#include "nigiri/routing/raptor/raptor_stats.h"
#include "nigiri/routing/transfer_time_settings.h"
#include "nigiri/types.h"

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

// Same scope as the CPU mcraptor (mcraptor_supported). One state type serves
// every configuration; the subtypes only select it in the dispatch (kCrit).
struct gpu_mcraptor_state {
  static constexpr auto kCrit = mc_crit::arr;

  explicit gpu_mcraptor_state(gpu_timetable const&);
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

  void execute(unixtime_t start_time,
               std::uint8_t max_transfers,
               unixtime_t worst_time_at_dest,
               pareto_set<journey>& results);

  void reconstruct(query const&, journey&);

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

  td_offsets_t const* td_start_{nullptr};
  std::vector<std::pair<location_idx_t, unixtime_t>> starts_;
};

}  // namespace nigiri::routing::gpu
