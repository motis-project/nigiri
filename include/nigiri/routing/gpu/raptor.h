#pragma once

#include "nigiri/routing/raptor/raptor.h"

namespace nigiri::routing::gpu {

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

template <direction SearchDir, bool Rt, via_offset_t Vias>
struct gpu_raptor {
  using algo_state_t = gpu_raptor_state;
  using algo_stats_t = raptor_stats;

  static constexpr bool kUseLowerBounds = true;
  static constexpr auto const kInvalid = kInvalidDelta<SearchDir>;
  static constexpr auto const kInvalidArray = []() {
    auto a = std::array<delta_t, Vias + 1>{};
    a.fill(kInvalid);
    return a;
  }();

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
      bool const is_wheelchair,
      transfer_time_settings const& tts);

  raptor_stats get_stats() const { return stats_; }
  void reset_arrivals();
  void next_start_time();

  void add_start(location_idx_t, unixtime_t);

  void execute(unixtime_t const start_time,
               std::uint8_t const max_transfers,
               unixtime_t const worst_time_at_dest,
               profile_idx_t const prf_idx,
               pareto_set<journey>& results);

  void reconstruct(query const&, journey&);

private:
  void sync_round_times();
  date::sys_days base() const {
    return tt_.internal_interval_days().from_ + to_idx(base_) * date::days{1};
  }

  timetable const& tt_;
  rt_timetable const* rtt_{nullptr};
  int n_days_;
  std::uint32_t n_locations_, n_routes_, n_rt_transports_;
  gpu_raptor_state& state_;
  bitvec const& is_dest_;
  std::array<bitvec, kMaxVias> const& is_via_;
  std::vector<std::uint16_t> const& dist_to_end_;
  hash_map<location_idx_t, std::vector<td_offset>> const& td_dist_to_end_;
  std::vector<std::uint16_t> const& lb_;
  std::vector<via_stop> const& via_stops_;
  std::array<delta_t, kMaxTransfers + 1> time_at_dest_;
  day_idx_t base_;
  raptor_stats stats_;
  clasz_mask_t allowed_claszes_;
  bool require_bike_transport_;
  bool is_wheelchair_;
  transfer_time_settings transfer_time_settings_;

  std::vector<std::pair<location_idx_t, unixtime_t>> starts_;
};

}  // namespace nigiri::routing::gpu