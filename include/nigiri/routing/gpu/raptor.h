#pragma once

#include <array>
#include <memory>
#include <vector>

#include "nigiri/common/delta_t.h"
#include "nigiri/routing/clasz_mask.h"
#include "nigiri/routing/journey.h"
#include "nigiri/routing/limits.h"
#include "nigiri/routing/mc_context.h"
#include "nigiri/routing/pareto_set.h"
#include "nigiri/routing/query.h"
#include "nigiri/routing/raptor/bmrap_bounds.h"
#include "nigiri/routing/raptor/raptor_stats.h"
#include "nigiri/routing/raptor/reconstruct.h"
#include "nigiri/routing/transfer_time_settings.h"
#include "nigiri/rt/rt_timetable.h"
#include "nigiri/special_stations.h"
#include "nigiri/timetable.h"
#include "nigiri/types.h"

namespace nigiri::routing::gpu {

constexpr auto kRtLocationCapacity = 8192U;

// An rt search needs the rt timetable's device copy (make_gpu_rtt()), which a
// failed upload leaves out.
inline bool is_gpu_supported(query const& q,
                             rt_timetable const* rtt = nullptr) {
  return q.via_stops_.empty() &&
         (rtt == nullptr || (rtt->gpu_rtt_.ptr_ != nullptr &&
                             rtt->n_rt_locations() <= kRtLocationCapacity));
}

// a usable CUDA device (a CUDA build may run without a driver)
bool gpu_available();

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

struct gpu_mcraptor_state;
struct gpu_mcraptor_limits;

struct gpu_raptor_state {
  explicit gpu_raptor_state(gpu_timetable const&);
  // mc: capacities of the device mc state (try_mc_state())
  gpu_raptor_state(gpu_timetable const&, gpu_mcraptor_limits const& mc);
  ~gpu_raptor_state();

  // The device state of BM-RAPTOR's mc phases, built on first use and kept
  // (too expensive per query); borrows this state's scratch and stream. The
  // allocation can fail: try_mc_state() first (a failure is remembered).
  gpu_mcraptor_state* try_mc_state();
  gpu_mcraptor_state& mc_state();

  struct impl;
  std::unique_ptr<impl> impl_;
};

template <direction SearchDir, bool WithBounds>
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
      bitvec const& is_dest,
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
      bool const no_compulsory_reservation_,
      transfer_time_settings const& tts,
      profile_idx_t const prf_idx,
      mc_context const& = {});

  raptor_stats get_stats() const { return stats_; }
  void reset_arrivals();
  void next_start_time();

  void fill_bounds(std::size_t n_rows);

  void set_bounds(unsigned const last_round)
    requires(WithBounds)
  {
    bounds_last_k_ = last_round;
  }

  void add_start(location_idx_t, unixtime_t, via_offset_t = 0U);

  void execute(
      unixtime_t start_time,
      std::uint8_t max_transfers,
      std::array<unixtime_t, kMaxTransfers + 2U> const& worst_time_at_dest,
      pareto_set<journey>& results);

  void reconstruct(query const&, journey&);

  // BM-RAPTOR hooks as on the CPU raptor; the matrix must be one this state
  // built (build_reach_bounds)
  void set_bounds(bmrap_bounds const*);
  // staggered rounds: add_start writes round k, the scan runs k + 1 .. end_k
  void set_start_round(unsigned k) {
    start_round_ = static_cast<std::uint8_t>(k);
  }
  // relaxed target pruning, so the round times form a valid tau_arr^->(v, i)
  // matrix; floor/cap are relax_arr()'s clamps
  void set_dest_relax(unixtime_t origin,
                      double factor,
                      int add_minutes,
                      double floor_min = 0.0,
                      double cap_min = 0.0);
  // reach_matrix() on the device, into the state's slot for its kind
  void build_reach_bounds(bmrap_bounds& out,
                          std::uint8_t budget,
                          bool sub_transfer,
                          bool host_copy);

private:
  date::sys_days base() const {
    return tt_.internal_interval_days().from_ + to_idx(base_) * date::days{1};
  }

  timetable const& tt_;
  rt_timetable const* rtt_;
  gpu_rt_timetable const* gpu_rtt_;
  std::uint32_t n_locations_;
  gpu_raptor_state& state_;
  // kept to re-upload the query when another engine of this direction used
  // the state in between
  bitvec const& is_dest_;
  std::vector<std::uint16_t> const* dist_to_dest_;
  hash_map<location_idx_t, std::vector<td_offset>> const* td_dist_to_dest_;
  day_idx_t base_;
  raptor_stats stats_;
  clasz_mask_t allowed_claszes_;
  bool require_bike_transport_;
  bool require_car_transport_;
  bool is_wheelchair_;
  bool no_compulsory_reservation_;
  transfer_time_settings transfer_time_settings_;

  profile_idx_t prf_idx_;
  delta_t const* bounds_;
  unsigned bounds_last_k_{0U};

  std::vector<std::pair<location_idx_t, unixtime_t>> starts_;

  // BM-RAPTOR (see the setters above)
  bmrap_bounds const* bm_bounds_{nullptr};
  std::uint8_t start_round_{0U};
  unixtime_t relax_origin_{};
  double relax_factor_{1.0};
  double relax_floor_{0.0};
  double relax_cap_{0.0};
  int relax_add_{0};
  bool relax_on_{false};
};

}  // namespace nigiri::routing::gpu
