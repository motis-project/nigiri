#pragma once
#include "nigiri/common/dial.h"
#include "nigiri/routing/pareto_set.h"
#include "nigiri/routing/tb/tb_data.h"
#include "nigiri/rt/rt_timetable.h"

namespace nigiri::routing::tb::a_star {

inline unsigned int transfer_factor = 5;
unixtime_t get_time(segment_idx_t const& idx,
                    timetable const& tt,
                    event_type const& event,
                    tb_data const& tbd,
                    day_idx_t const& day_idx);
day_idx_t get_day(unixtime_t const& before,
                  segment_idx_t const& s_idx,
                  timetable const& tt,
                  tb_data const& tbd);
std::pair<std::vector<segment_idx_t>, bool> get_neighbours(
    segment_idx_t const& s_idx, tb_data const& tbd, day_idx_t const& day);

struct cost_func_t {
  explicit cost_func_t() = default;
  std::size_t operator()(
      std::pair<segment_idx_t, duration_t> const& element) const {
    if (element.second.count() < 0)
      throw std::runtime_error("Got negative costs");
    return static_cast<std::size_t>(element.second.count());
  }
};

struct a_star_stats {
  std::map<std::string, std::uint64_t> to_map() const {
    return {{"max_transfers_reached_", max_transfers_reached_},
            {"queued_segments_", queued_segments_},
            {"segments_taken_out_", segments_taken_out_},
            {"segments_ignored_", segments_ignored_},
            {"ever_requeued_", ever_requeued_}};
  }

  bool max_transfers_reached_ = false;
  int queued_segments_ = 0;
  int segments_taken_out_ = 0;
  int segments_ignored_ = 0;
  bool ever_requeued_ = false;
};

struct a_star_state {
  explicit a_star_state(tb_data const& tbd) : tbd_{tbd} {
    end_reachable_.resize(tbd_.segment_transfers_.size());
  }
  tb_data tbd_;
  bitvec_map<segment_idx_t> end_reachable_;
  hash_map<segment_idx_t, duration_t> dist_to_dest_;
};

struct tb_a_star {

  using algo_state_t = a_star_state;
  using algo_stats_t = a_star_stats;
  static constexpr bool kUseLowerBounds = true;
  static constexpr auto kUnreachable =
      std::numeric_limits<std::uint16_t>::max();

  explicit tb_a_star(unsigned int const& init_cap)
      : state_(tb_data()),
        queue_(1440 + (init_cap - 1) * transfer_factor, cost_func_t()) {
    pred_ = vector_map<segment_idx_t, segment_idx_t>(init_cap,
                                                     segment_idx_t::invalid());
    day_idx_ = vector_map<segment_idx_t, day_idx_t>(init_cap);
    transfers_ = vector_map<segment_idx_t, uint8_t>(
        init_cap, std::numeric_limits<uint8_t>::max());
    is_start_segment_ = bitvec_map<segment_idx_t>(init_cap);
    is_start_segment_.zero_out();
  }

  tb_a_star(timetable const& tt,
            rt_timetable const*,
            a_star_state& state,
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

  algo_stats_t get_stats() const { return algo_stats_; }
  void reset_arrivals() {}
  void next_start_time() {
    queue_.clear();
    is_start_segment_.zero_out();
    utl::fill(pred_, segment_idx_t::invalid());
    utl::fill(transfers_, std::numeric_limits<uint8_t>::max());
  }
  void add_start(location_idx_t const l, unixtime_t const t);
  segment_idx_t get_departure_segment(segment_idx_t const& s);
  void execute(unixtime_t const start_time,
               std::uint8_t const max_transfers,
               unixtime_t const worst_time_at_dest,
               profile_idx_t const profile_idx,
               pareto_set<journey>& results);
  void reconstruct(query const& q, journey& j);

  duration_t heuristic(segment_idx_t const& s);

  vector_map<segment_idx_t, segment_idx_t> pred_;
  vector_map<segment_idx_t, day_idx_t> day_idx_;
  timetable tt_;
  a_star_state state_;
  dial<std::pair<segment_idx_t, duration_t>, cost_func_t> queue_;
  segment_idx_t end_segment_ = segment_idx_t::invalid();
  vector_map<segment_idx_t, uint8_t> transfers_;
  std::vector<std::uint16_t> travel_time_lower_bound_;
  bitvec_map<segment_idx_t> is_start_segment_;
  algo_stats_t algo_stats_;
};

}  // namespace nigiri::routing::tb::a_star