#pragma once

#include "nigiri/routing/search.h"
#include "nigiri/routing/tb/query_engine.h"
#include "nigiri/timetable.h"

namespace nigiri::routing::a_star {

routing_result initialize_a_star(timetable const& tt,
                                 search_state& s_state,
                                 tb::query_state& r_state,
                                 query q);

enum class element_type { SEGMENT, FINAL_FOOTPATH };

struct open_set_element {
  element_type type;
  tb::segment_idx_t seg;
};

struct a_star_search {
  static constexpr bool kUseLowerBounds = true;
  using algo_state_t = tb::query_state;
  using algo_stats_t = tb::query_stats;
  a_star_search(
      timetable const& tt,
      rt_timetable const* rtt,
      tb::query_state& state,
      bitvec const& is_dest,
      std::array<bitvec, kMaxVias> const& is_via,
      std::vector<std::uint16_t> const& dist_to_dest,
      hash_map<location_idx_t, std::vector<td_offset>> const& td_dist_to_dest,
      std::vector<std::uint16_t> const& lb,
      std::vector<via_stop> const& via_stops,
      day_idx_t base_day,
      clasz_mask_t allowed_claszes,
      bool require_bike_transport,
      bool require_car_transport,
      bool is_wheelchair,
      transfer_time_settings const& tts);

  algo_stats_t get_stats() const { return stats_; }

  void reset_arrivals() {}

  void next_start_time() {
    segment_day_.clear();
    arrival_times_.clear();
    num_transfers_until_segment_.clear();
    num_transfers_until_segment_.clear();
    came_from_.clear();
    to_location_.clear();
    closed_set_.zero_out();
    open_set_.clear();
    start_segments_.clear();
    potential_starts_.clear();
  }

  size_t cost_function(open_set_element el,
                       unixtime_t arrival_time,
                       uint32_t num_transfers_until_segment);

  void execute(unixtime_t start_time,
               uint8_t max_transfers,
               unixtime_t worst_time_at_dest,
               profile_idx_t prf_idx,
               pareto_set<journey>& journeys);

  void progress_neighbor_segment(tb::segment_idx_t current,
                                 tb::segment_idx_t neighbor,
                                 day_idx_t neighbor_day,
                                 bool is_same_trip);

  void expand_node(tb::segment_idx_t current);

  pair<location_idx_t, location_idx_t> get_segment_locations(
      tb::segment_idx_t seg);

  unixtime_t get_arrival_time(tb::segment_idx_t seg);

  unixtime_t get_departure_time(tb::segment_idx_t seg);

  void add_start(location_idx_t, unixtime_t);

  void reconstruct(query const& q, journey& j);

  timetable const& tt_;
  bitvec const& is_dest_;
  tb::query_state& state_;
  tb::query_stats stats_;
  std::vector<std::uint16_t> const& lower_bounds_;
  using bucket_fn = std::function<std::size_t(open_set_element)>;
  bucket_fn get_bucket_;

  // open_set: all potential next segments (next segment in current trip or
  // segments after transfer) as well as final footpaths closed_set: all already
  // covered segments came_from: predecessor segment for each segment in openSet
  // or closedSet arrival_times, num_transfers_until_segment: relevant for cost
  // function
  dial<open_set_element, bucket_fn> open_set_;
  bitvec_map<tb::segment_idx_t> closed_set_;
  hash_map<tb::segment_idx_t, tb::segment_idx_t> came_from_;
  hash_map<tb::segment_idx_t, unixtime_t> arrival_times_;
  hash_map<tb::segment_idx_t, day_idx_t> segment_day_;
  hash_map<tb::segment_idx_t, uint32_t> num_transfers_until_segment_;

  unixtime_t start_time_;
  size_t cost_upper_bound_;
  // 'start_segments' are the start segments which are actually viable, i.e.
  // don't take too much time. 'potential_starts' are the pairs of ALL start
  // segments and the corresponding earliest transport.
  vector<tb::segment_idx_t> start_segments_;
  vector<std::pair<tb::segment_idx_t, transport>> potential_starts_;
  tb::segment_idx_t end_segment_;  // final segment of found route
  std::map<tb::segment_idx_t, location_idx_t> nearest_target_location_;
  std::map<tb::segment_idx_t, location_idx_t> to_location_;
  bool journey_found_;
};

}  // namespace nigiri::routing::a_star
