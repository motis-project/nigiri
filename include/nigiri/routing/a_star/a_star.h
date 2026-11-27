#pragma once

#include "nigiri/common/dial.h"
#include "nigiri/routing/a_star/a_star_state.h"
#include "nigiri/routing/a_star/settings.h"
#include "nigiri/routing/journey.h"
#include "nigiri/routing/pareto_set.h"
#include "nigiri/routing/tb/tb_data.h"
#include "nigiri/timetable.h"
#include "nigiri/types.h"

namespace nigiri {
namespace routing {

using tb_data = tb::tb_data;
using segment_idx_t = tb::segment_idx_t;

struct a_star_stats {
  std::map<std::string, std::uint64_t> to_map() const {
    return {
        {"use_lower_bounds", use_lower_bounds_},
        {"n_segments_reached", n_segments_reached_},
        {"n_dest_segments_reached", n_dest_segments_reached_},
        {"max_transfers_reached", max_transfers_reached_},
        {"max_travel_time_reached", max_travel_time_reached_},
        {"no_journey_found", no_journey_found_},
    };
  }

  std::uint64_t n_segments_reached_{0ULL};
  std::uint64_t n_dest_segments_reached_{0ULL};
  bool use_lower_bounds_{true};
  bool max_transfers_reached_{false};
  bool max_travel_time_reached_{false};
  bool no_journey_found_{false};
};

template <bool UseLowerBounds>
struct a_star {
  using algo_state_t = a_star_state;
  using algo_stats_t = a_star_stats;
  static constexpr bool kUseLowerBounds = UseLowerBounds;

  static constexpr auto const kUnreachable =
      std::numeric_limits<std::uint16_t>::max();

  a_star(timetable const& tt,
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
         transfer_time_settings tts);

  algo_stats_t get_stats() const { return stats_; }

  algo_state_t& get_state() { return state_; }

  void reset_arrivals() { state_.reset(); }

  void next_start_time() { state_.reset(); }

  void add_start(location_idx_t, unixtime_t);

  void execute(unixtime_t const start_time,
               std::uint8_t const max_transfers,
               unixtime_t const worst_time_at_dest,
               profile_idx_t const,
               pareto_set<journey>& results);

  void reconstruct(query const& q, journey& j) const;

  delta event_day_idx_mam(transport t,
                          stop_idx_t const s_idx,
                          event_type const et) {
    auto const arr_time = tt_.event_mam(t.t_idx_, s_idx, et);
    return delta{static_cast<std::uint16_t>(arr_time.days_ + t.day_.v_),
                 arr_time.mam_};
  }

  delta day_idx_mam(day_idx_t const day,
                    minutes_after_midnight_t const mam) const {
    return delta{to_idx(day - base_), static_cast<std::uint16_t>(mam.count())};
  }

  delta day_idx_mam(unixtime_t const ut) const {
    auto const [d, t] = tt_.day_idx_mam(ut);
    return day_idx_mam(d, t);
  }

  unixtime_t to_unixtime(delta const& d) const {
    return tt_.to_unixtime(base_ + static_cast<day_idx_t>(d.days()),
                           static_cast<minutes_after_midnight_t>(d.mam()));
  }

private:
  timetable const& tt_;
  a_star_state& state_;
  bitvec const& is_dest_;
  std::vector<std::uint16_t> const& dist_to_dest_;
  std::vector<std::uint16_t> const& lb_;
  day_idx_t const base_;
  a_star_stats stats_;
};

}  // namespace routing
}  // namespace nigiri