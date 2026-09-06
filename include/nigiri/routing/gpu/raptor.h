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
#include "nigiri/routing/raptor/bmrap_bounds.h"
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

// Is there a usable CUDA device? False when the toolkit is present but the
// driver is not (a CUDA-enabled build on a machine without a GPU), which
// lets tests skip instead of reporting failures.
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

struct gpu_raptor_state {
  explicit gpu_raptor_state(gpu_timetable const&);
  ~gpu_raptor_state();

  struct impl;
  std::unique_ptr<impl> impl_;
};

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

  void add_start(location_idx_t, unixtime_t);

  void execute(unixtime_t start_time,
               std::uint8_t max_transfers,
               unixtime_t worst_time_at_dest,
               profile_idx_t prf_idx,
               pareto_set<journey>& results);

  void reconstruct(query const&, journey&);

  // --- BM-RAPTOR hooks, mirroring the CPU raptor ---
  // tau_dep^<- / tau_arr^-> pruning. The matrix is uploaded to the device
  // here; nullptr disables pruning.
  void set_bounds(bmrap_bounds const*);
  // staggered round alignment: add_start writes round k, the scan runs
  // start_round_+1 .. end_k
  void set_start_round(unsigned k) { start_round_ = static_cast<std::uint8_t>(k); }
  // relaxed target pruning, so this search's round times form a valid
  // tau_arr^->(v, i) matrix; floor/cap are the clamps relax_arr() applies
  void set_dest_relax(unixtime_t origin,
                      double factor,
                      int add_minutes,
                      double floor_min = 0.0,
                      double cap_min = 0.0);
  // Host copy of the device round times, unpacked and laid out exactly like
  // raptor_state::get_round_times() so build_reach_matrix() can consume it:
  // (kMaxTransfers + 2) rows of n_locations entries.
  void copy_round_times(std::vector<std::array<delta_t, 1>>& out);

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

  std::vector<std::pair<location_idx_t, unixtime_t>> starts_;

  // BM-RAPTOR state (see the setters above)
  std::uint32_t bounds_n_locations_{0U};
  std::uint8_t bounds_budget_{0U};
  bool has_bounds_{false};
  std::uint8_t start_round_{0U};
  unixtime_t relax_origin_{};
  double relax_factor_{1.0};
  double relax_floor_{0.0};
  double relax_cap_{0.0};
  int relax_add_{0};
  bool relax_on_{false};
};

}  // namespace nigiri::routing::gpu