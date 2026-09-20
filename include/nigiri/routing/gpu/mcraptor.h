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

// GPU McRAPTOR supports the CPU mcraptor scope (see mcraptor_supported).
// One state serves every criteria configuration (the packed label has the same
// width); the distinct types select the algorithm in the dispatch.
struct gpu_mcraptor_state {
  explicit gpu_mcraptor_state(gpu_timetable const&);
  ~gpu_mcraptor_state();

  struct impl;
  std::unique_ptr<impl> impl_;
};

struct gpu_mcraptor_cost_state : gpu_mcraptor_state {
  using gpu_mcraptor_state::gpu_mcraptor_state;
};

// The label configurations the device implements, each result-equal to the CPU
// mcraptor with the matching criteria (mc_crit_of in bmrap_common.h). All fit
// the 16-bit "crit" slot of the packed label:
//   arr                      arrival only
//   cost                     arrival + generalized cost
//   non_transit              arrival + minutes on foot
//   mode_filter              arrival + "uses an avoided class" bit (AIR)
//   non_transit_mode_filter  both (non_transit in bits 1..15, mode_filter bit
//   0)
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

  // Unlike the scalar GPU raptor this uses lower bounds: the lb-projected
  // destination pruning keeps the bags small, and without it labels the CPU
  // never stores flood the fixed-capacity device bags. Pruning only.
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
      bool no_compulsory_reservation,
      transfer_time_settings const& tts,
      profile_idx_t prf_idx);

  raptor_stats get_stats() const { return stats_; }

  // Pong side: reuse-frontier rejections only against entries of the same
  // departure (one merged anchor run); cross-anchor ones were seen
  // over-pruning.
  void set_reuse_same_dep() { reuse_same_dep_ = true; }

  // Tight starts (pong ping), see basic_mcraptor::set_tight_start(); the device
  // reconstruct reports the shift per journey (gpu_journey::start_shift_).
  void set_tight_start() { tight_start_ = true; }

  // BM-RAPTOR pruning bounds, uploaded here; nullptr disables. See
  // basic_mcraptor::set_bounds().
  void set_bounds(bmrap_bounds const*);

  void reset_arrivals();
  void next_start_time();
  void add_start(location_idx_t, unixtime_t);

  void execute(unixtime_t start_time,
               std::uint8_t max_transfers,
               unixtime_t worst_time_at_dest,
               pareto_set<journey>& results);

  // The device breadcrumb chase materializes the core legs; this adds offset
  // legs and the start footpath on the host.
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
  profile_idx_t prf_idx_;

  // search-window bound; the device dest frontier owns destination pruning
  delta_t worst_at_dest_;
  bool reuse_same_dep_{false};
  bool tight_start_{false};  // see set_tight_start()

  std::vector<std::pair<location_idx_t, unixtime_t>> starts_;

  std::uint32_t bounds_n_locations_{0U};
  std::uint8_t bounds_budget_{0U};
  bool has_bounds_{false};
};

}  // namespace nigiri::routing::gpu
