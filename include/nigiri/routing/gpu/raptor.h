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

inline bool gpu_supported(query const& q) {
  return q.allowed_claszes_ == all_clasz_allowed() && q.td_start_.empty() &&
         q.td_dest_.empty() && q.transfer_time_settings_.default_ &&
         !q.require_bike_transport_ && !q.require_car_transport_ &&
         q.via_stops_.empty() && q.prf_idx_ == 0U;
}

struct gpu_timetable {
  explicit gpu_timetable(timetable const&);
  ~gpu_timetable();

  struct impl;
  std::unique_ptr<impl> impl_;
};

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

private:
  date::sys_days base() const {
    return tt_.internal_interval_days().from_ + to_idx(base_) * date::days{1};
  }

  timetable const& tt_;
  std::uint32_t n_locations_;
  gpu_raptor_state& state_;
  bitvec const& is_dest_;
  day_idx_t base_;
  raptor_stats stats_;
  clasz_mask_t allowed_claszes_;
  transfer_time_settings transfer_time_settings_;

  std::vector<std::pair<location_idx_t, unixtime_t>> starts_;
};

}  // namespace nigiri::routing::gpu