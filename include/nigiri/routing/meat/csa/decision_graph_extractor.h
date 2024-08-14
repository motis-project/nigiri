#pragma once

#include <stack>

#include "nigiri/common/delta_t.h"
#include "nigiri/routing/limits.h"
#include "nigiri/routing/meat/csa/profile.h"
#include "nigiri/routing/meat/decision_graph.h"
#include "nigiri/timetable.h"
#include "nigiri/types.h"

namespace nigiri::routing::meat::csa {

struct decision_graph_extractor {
private:
  timetable const& tt_;
  day_idx_t base_;
  mutable std::stack<profile_entry const*> stack_;
  // mutable std::vector<profile_entry*> queue_;
  mutable bitvec is_enter_conn_relevant_;
  mutable vector_map<location_idx_t, int> to_node_id_;

public:
  explicit decision_graph_extractor(timetable const& tt, day_idx_t base)
      : tt_(tt),
        base_(base),
        to_node_id_(tt_.n_locations(), -1) {
    is_enter_conn_relevant_.resize(tt_.fwd_connections_.size() *
                                   kMaxSearchDays);

    // size_t n_entry1 = 0;
    // size_t n_entry2 = 0;
    // for (auto& c : tt.fwd_connections_) {
    //   if (c.dep_stop_.in_allowed()) {
    //     ;
    //     n_entry1 +=
    //         tt.bitfields_[tt.transport_traffic_days_[c.transport_idx_]].count();
    //     n_entry2 += kMaxSearchDays;
    //   }
    // }
    // auto b = n_entry1 < n_entry2;
    // queue_ = std::vector<int>(b ? n_entry1 : n_entry2);
  }

private:
  std::vector<profile_entry const*> extract_relevant_entries(
      profile_set const& profile_set,
      location_idx_t source_stop,
      delta_t source_time,
      location_idx_t target_stop,
      delta_t max_delay) const;
  int as_int(day_idx_t const d) const { return static_cast<int>(d.v_); }
  date::sys_days base() const {
    return tt_.internal_interval_days().from_ + as_int(base_) * date::days{1};
  }
  unixtime_t to_unix(delta_t const t) const { return delta_to_unix(base(), t); }
  unixtime_t to_unix(meat_t const t) const { return delta_to_unix(base(), t); }
  std::pair<day_idx_t, minutes_after_midnight_t> split(delta_t const x) const {
    return split_day_mam(base_, x);
  }

public:
  decision_graph operator()(profile_set const& profile_set,
                            location_idx_t source_stop,
                            delta_t source_time,
                            location_idx_t target_stop,
                            delta_t max_delay) const;
};

std::pair<decision_graph, delta_t> extract_small_sub_decision_graph(
    decision_graph_extractor const& e,
    profile_set const& profile_set,
    location_idx_t source_stop,
    delta_t source_time,
    location_idx_t target_stop,
    delta_t max_delay,
    int max_ride_count,
    int max_arrow_count);
}  // namespace nigiri::routing::meat::csa