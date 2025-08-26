#pragma once

#include <cinttypes>

#include "nigiri/routing/journey.h"
#include "nigiri/routing/pareto_set.h"
#include "nigiri/routing/query.h"
#include "nigiri/routing/tb/preprocess.h"
#include "nigiri/routing/tb/q_n.h"
#include "nigiri/routing/tb/reached.h"
#include "nigiri/routing/tb/settings.h"

namespace nigiri {
struct timetable;
}  // namespace nigiri

namespace nigiri::routing::tb {

struct queue_entry;

// a route that reaches the destination
struct route_dest {
  route_dest(std::uint16_t stop_idx, std::uint16_t time)
      : stop_idx_(stop_idx), time_(time) {}
  // the stop index at which the route reaches the target location
  std::uint16_t stop_idx_;
  // the time in it takes after exiting the route until the target location is
  // reached
  std::uint16_t time_;
};

struct query_start {
  query_start(location_idx_t const l, unixtime_t const t)
      : location_(l), time_(t) {}

  location_idx_t location_;
  unixtime_t time_;
};

struct query_state {
  query_state() = delete;
  query_state(timetable const& tt, tb_data const& tbd)
      : ts_{tbd}, r_{tt}, q_n_{r_} {
    route_dest_.reserve(128);
    t_min_.resize(kNumTransfersMax, unixtime_t::max());
    q_n_.segments_.reserve(10000);
    query_starts_.reserve(20);
    route_dest_.resize(tt.n_routes());
  }

  void reset(day_idx_t new_base) {
    std::fill(t_min_.begin(), t_min_.end(), unixtime_t::max());
    r_.reset();
    q_n_.reset(new_base);
    for (auto& inner_vec : route_dest_) {
      inner_vec.clear();
    }
  }

  // transfer set built by preprocessor
  tb_data const& tbd_;

  // routes that reach the target stop
  std::vector<std::vector<route_dest>> route_dest_;

  // reached stops per transport
  reached r_;

  // minimum arrival times per number of transfers
  std::vector<unixtime_t> t_min_;

  // queues of transport segments
  q_n q_n_;

  std::vector<query_start> query_starts_;
};

struct query_stats {
  bool lower_bound_pruning_{false};
  std::uint64_t n_segments_enqueued_{0U};
  std::uint64_t n_segments_pruned_{0U};
  std::uint64_t n_enqueue_prevented_by_reached_{0U};
  std::uint64_t n_journeys_found_{0U};
  std::uint64_t n_rounds_{0U};
  bool max_transfers_reached_{false};
};

template <bool UseLowerBounds>
struct query_engine {
  using algo_state_t = query_state;
  using algo_stats_t = query_stats;

  static constexpr bool kUseLowerBounds = UseLowerBounds;
  static constexpr auto const kUnreachable =
      std::numeric_limits<std::uint16_t>::max();

  query_engine(
      timetable const& tt,
      rt_timetable const* rtt,
      query_state& state,
      bitvec& is_dest,
      std::optional<std::array<bitvec, kMaxVias>> is_via,  // unsupported
      std::vector<std::uint16_t>& dist_to_dest,
      std::optional<hash_map<location_idx_t, std::vector<td_offset>>>
          td_dist_to_dest,  // unsupported
      std::vector<std::uint16_t>& lb,
      std::optional<std::vector<via_stop>> via_stops,  // unsupported
      day_idx_t const base,
      std::optional<clasz_mask_t> allowed_claszes,  // unsupported
      std::optional<bool> require_bike_transport,  // unsupported
      std::optional<bool> is_wheelchair,  // unsupported
      std::optional<transfer_time_settings> tts);  // unsupported

  algo_stats_t get_stats() const { return stats_; }

  algo_state_t& get_state() { return state_; }

  void reset_arrivals() {
    state_.r_.reset();
    std::fill(state_.t_min_.begin(), state_.t_min_.end(), unixtime_t::max());
  }

  void next_start_time() {
    state_.q_n_.reset(base_);
    state_.query_starts_.clear();
  }

  void add_start(location_idx_t, unixtime_t);

  void execute(unixtime_t const start_time,
               std::uint8_t const max_transfers,
               unixtime_t const worst_time_at_dest,
               profile_idx_t const,
               pareto_set<journey>& results);

  void reconstruct(query const& q, journey& j) const;

private:
  void handle_start_footpath(day_idx_t,
                             minutes_after_midnight_t,
                             footpath const);

  void seg_dest(unixtime_t const start_time,
                pareto_set<journey>& results,
                unixtime_t worst_time_at_dest,
                std::uint8_t const n,
                queue_entry& seg);

  void seg_prune(unixtime_t const worst_time_at_dest,
                 std::uint8_t const n,
                 queue_entry& seg);

  void seg_transfers(std::uint8_t const n, queue_idx_t const q_cur);

  struct journey_end {
    journey_end(queue_idx_t const seg_idx,
                route_dest const& le,
                location_idx_t const le_location,
                location_idx_t const last_location)
        : seg_idx_(seg_idx),
          le_(le),
          le_location_(le_location),
          last_location_(last_location) {}

    // the last transport segment of the journey
    queue_idx_t seg_idx_;
    // the l_entry for the destination of the journey
    route_dest le_;
    // the location idx of the l_entry
    location_idx_t le_location_;
    // the reconstructed destination of the journey
    location_idx_t last_location_;
  };

  std::optional<journey_end> reconstruct_journey_end(query const& q,
                                                     journey const& j) const;

  void add_final_footpath(query const& q,
                          journey& j,
                          journey_end const& je) const;

  void add_segment_leg(journey& j, queue_entry const& seg) const;

  // reconstruct the transfer from the given segment to the last journey leg
  // returns the stop idx at which the segment is exited
  std::optional<queue_entry> reconstruct_transfer(journey& j,
                                                  queue_entry const& seg_next,
                                                  std::uint8_t n) const;

  void add_initial_footpath(query const& q, journey& j) const;

  bool is_start_location(query const&, location_idx_t const) const;

  timetable const& tt_;
  rt_timetable const* rtt_;
  query_state& state_;
  bitvec& is_dest_;
  std::vector<std::uint16_t>& dist_to_dest_;
  std::vector<std::uint16_t>& lb_;
  day_idx_t const base_;
  query_stats stats_;
};

}  // namespace nigiri::routing::tb