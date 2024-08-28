#pragma once

#include <stack>

#include "nigiri/common/delta_t.h"
#include "nigiri/routing/limits.h"
#include "nigiri/routing/meat/compact_representation.h"
#include "nigiri/routing/meat/csa/profile.h"
#include "nigiri/routing/meat/decision_graph.h"
#include "nigiri/timetable.h"
#include "nigiri/types.h"

namespace nigiri::routing::meat::csa {

template <typename ProfileSet>
struct decision_graph_extractor {
public:
  explicit decision_graph_extractor(timetable const& tt,
                                    day_idx_t const& base,
                                    ProfileSet const& ps)
      : tt_(tt), base_(base), to_node_id_(tt_.n_locations(), -1) {
    is_enter_conn_relevant_.resize(ps.n_entry_idxs());
  }

private:
  std::vector<profile_entry const*> extract_relevant_entries(
      ProfileSet const& profile_set,
      location_idx_t source_stop,
      delta_t source_time,
      location_idx_t target_stop,
      delta_t max_delay) const;
  int as_int(day_idx_t const d) const { return static_cast<int>(d.v_); }
  date::sys_days base() const {
    return tt_.internal_interval_days().from_ + as_int(base_) * date::days{1};
  }
  unixtime_t to_unix(delta_t const t) const { return delta_to_unix(base(), t); }
  unixtime_t to_unix(meat_t const t) const { return delta_to_unix(base(), static_cast<delta_t>(t)); }
  std::pair<day_idx_t, minutes_after_midnight_t> split(delta_t const x) const {
    return split_day_mam(base_, x);
  }

public:
  decision_graph operator()(ProfileSet const& profile_set,
                            location_idx_t source_stop,
                            delta_t source_time,
                            location_idx_t target_stop,
                            delta_t max_delay) const;

private:
  timetable const& tt_;
  day_idx_t const& base_;
  mutable std::stack<profile_entry const*> stack_;
  mutable bitvec is_enter_conn_relevant_;
  mutable vector_map<location_idx_t, int> to_node_id_;
};

template <typename ProfileSet>
std::pair<decision_graph, delta_t> extract_small_sub_decision_graph(
    decision_graph_extractor<ProfileSet> const& e,
    ProfileSet const& profile_set,
    location_idx_t source_stop,
    delta_t source_time,
    location_idx_t target_stop,
    delta_t max_delay,
    int max_ride_count,
    int max_arrow_count) {
  if (max_ride_count == std::numeric_limits<int>::max() &&
      max_arrow_count == std::numeric_limits<int>::max())
    return {e(profile_set, source_stop, source_time, target_stop, max_delay),
            max_delay};
  else if (max_arrow_count == std::numeric_limits<int>::max()) {
    delta_t min_delay = 0;
    while (min_delay != max_delay) {
      delta_t mid_delay = (min_delay + max_delay + 1) / 2;
      auto g = e(profile_set, source_stop, source_time, target_stop, mid_delay);
      if (g.arc_count() <= max_ride_count) {
        min_delay = mid_delay;
      } else {
        max_delay = mid_delay - 1;
      }
    }
    return {e(profile_set, source_stop, source_time, target_stop, max_delay),
            max_delay};
  } else {
    delta_t min_delay = 0;
    while (min_delay != max_delay) {
      delta_t mid_delay = (min_delay + max_delay + 1) / 2;
      auto g = e(profile_set, source_stop, source_time, target_stop, mid_delay);
      if (g.arc_count() <= max_ride_count) {
        if (compact_representation(g).arrow_count() <= max_arrow_count)
          min_delay = mid_delay;
        else
          max_delay = mid_delay - 1;
      } else {
        max_delay = mid_delay - 1;
      }
    }
    return {e(profile_set, source_stop, source_time, target_stop, max_delay),
            max_delay};
  }
}
}  // namespace nigiri::routing::meat::csa