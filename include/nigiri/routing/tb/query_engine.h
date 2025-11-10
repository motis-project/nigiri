#pragma once

#include "nigiri/routing/journey.h"
#include "nigiri/routing/pareto_set.h"
#include "nigiri/routing/query.h"
#include "nigiri/routing/tb/queue.h"
#include "nigiri/routing/tb/reached.h"
#include "nigiri/routing/tb/settings.h"

namespace nigiri {
struct timetable;
}

namespace nigiri::routing::tb {

struct queue_entry;
struct segment_info;

struct query_state {
  query_state(timetable const& tt, tb_data const& tbd)
      : tbd_{tbd}, r_{tt, tbd} {
    t_min_.fill(unixtime_t::max());
    q_n_.q_.reserve(10'000'000);
    end_reachable_.resize(tbd.segment_transfers_.size());
  }

  void reset() {
    utl::fill(parent_, queue_entry::kNoParent);
    r_.reset();
    q_n_.reset();
    end_reachable_.zero_out();
  }

  tb_data const& tbd_;

  // avx_reached r_;
  // queue<avx_reached> q_n_{r_};

  reached r_;
  queue<reached> q_n_{r_};

  // minimum arrival times per round
  std::array<unixtime_t, kMaxTransfers + 1U> t_min_;
  std::array<queue_idx_t, kMaxTransfers + 1U> parent_;

  bitvec_map<segment_idx_t> end_reachable_;
  hash_map<segment_idx_t, duration_t> dist_to_dest_;
};

struct query_stats {
  std::map<std::string, std::uint64_t> to_map() const {
    return {
        {"lower_bound_pruning", lower_bound_pruning_},
        {"n_segments_enqueued", n_segments_enqueued_},
        {"n_segments_pruned", n_segments_pruned_},
        {"n_enqueue_prevented_by_reached", n_enqueue_prevented_by_reached_},
        {"n_journeys_found", n_journeys_found_},
        {"n_rounds", n_rounds_},
        {"max_transfers_reached", max_transfers_reached_},
        {"max_pareto_set_size", max_pareto_set_size_},
    };
  }

  bool lower_bound_pruning_{false};
  std::uint64_t n_segments_enqueued_{0U};
  std::uint64_t n_segments_pruned_{0U};
  std::uint64_t n_enqueue_prevented_by_reached_{0U};
  std::uint64_t n_journeys_found_{0U};
  std::uint64_t n_rounds_{0U};
  std::uint64_t max_pareto_set_size_{0U};
  bool max_transfers_reached_{false};
};

template <bool UseLowerBounds>
struct query_engine {
  using algo_state_t = query_state;
  using algo_stats_t = query_stats;

  static constexpr bool kUseLowerBounds = UseLowerBounds;
  static constexpr auto const kUnreachable =
      std::numeric_limits<std::uint16_t>::max();

  query_engine(timetable const&,
               rt_timetable const*,
               query_state&,
               bitvec const& is_dest,
               std::array<bitvec, kMaxVias> const&,
               std::vector<std::uint16_t> const& dist_to_dest,
               hash_map<location_idx_t, std::vector<td_offset>> const&,
               std::vector<std::uint16_t> const& lb,
               std::vector<via_stop> const&,
               day_idx_t base,
               clasz_mask_t,
               bool,
               bool,
               bool,
               transfer_time_settings);

  algo_stats_t get_stats() const { return stats_; }

  algo_state_t& get_state() { return state_; }

  void reset_arrivals() {
    state_.r_.reset();
    utl::fill(state_.t_min_, unixtime_t::max());
    utl::fill(state_.parent_, queue_entry::kNoParent);
  }

  void next_start_time() { state_.q_n_.reset(); }

  void add_start(location_idx_t, unixtime_t);

  void execute(unixtime_t const start_time,
               std::uint8_t const max_transfers,
               unixtime_t const worst_time_at_dest,
               profile_idx_t const,
               pareto_set<journey>& results);

  void reconstruct(query const& q, journey& j) const;

private:
  void seg_dest(std::uint8_t k, queue_idx_t);
  void seg_prune(std::uint8_t k, queue_entry&);
  void seg_transfers(queue_idx_t, std::uint8_t k);

  segment_info seg(segment_idx_t, queue_entry const&) const;
  segment_info seg(segment_idx_t, day_idx_t) const;

  timetable const& tt_;
  query_state& state_;
  bitvec const& is_dest_;
  std::vector<std::uint16_t> const& dist_to_dest_;
  std::vector<std::uint16_t> const& lb_;
  day_idx_t base_;
  query_stats stats_;
};

}  // namespace nigiri::routing::tb