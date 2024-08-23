#include "nigiri/routing/meat/csa/profile.h"

#include "nigiri/routing/limits.h"

namespace nigiri::routing::meat::csa {

static_profile_set::static_profile_set(timetable const& tt)
    : fp_dis_to_target_(tt.n_locations(),
                        std::numeric_limits<meat_t>::infinity()),
      fp_to_target_reset_list_(tt.n_locations()),
      fp_to_target_reset_list_end_{0},
      stop_reset_list_(tt.n_locations()),
      stop_reset_list_end_{0},
      entry_begin_end_(tt.n_locations()) {
  vector_map<location_idx_t, val_t_con> max_entry(tt.n_locations(), 1);
  size_t n_entry = tt.n_locations();
  for (auto& c : tt.fwd_connections_) {
    if (!stop{c.dep_stop_}.in_allowed()) {
      continue;
    }
    auto n_c_in_tt =
        tt.bitfields_[tt.transport_traffic_days_[c.transport_idx_]].count();
    max_entry[stop{c.dep_stop_}.location_idx()] += n_c_in_tt;
    n_entry += n_c_in_tt;

    vector_map<location_idx_t, val_t_con> max_fp(tt.n_locations(), 0);
    for (auto const& fps : tt.locations_.footpaths_in_) {
      if (fps.empty()) {
        continue;
      }
      vector_map<location_idx_t, val_t_con> n_fp(tt.n_locations(), 0);
      for (auto const& fp : fps[stop{c.dep_stop_}.location_idx()]) {
        ++n_fp[fp.target()];
        if (n_fp[fp.target()] <= max_fp[fp.target()]) {
          continue;
        }
        ++max_fp[fp.target()];
        max_entry[fp.target()] += n_c_in_tt;
        n_entry += n_c_in_tt;
      }
    }
  }
  entry_ = std::vector<profile_entry>(n_entry);

  entry_begin_end_[location_idx_t{0}].first = 0;
  for (auto i = location_idx_t{1}; i < tt.n_locations(); ++i)
    entry_begin_end_[i].first =
        entry_begin_end_[i - 1].first + max_entry[i - 1];

  for (auto i = location_idx_t{0}; i < tt.n_locations(); ++i) {
    entry_begin_end_[i].second = entry_begin_end_[i].first + 1;
    entry_[entry_begin_end_[i].first] = {
        std::numeric_limits<delta_t>::max(),
        std::numeric_limits<meat_t>::infinity(),
        ride{connection_idx_t::invalid(), connection_idx_t::invalid()}};
  }
}

val_t_con static_profile_set::compute_entry_amount() const {
  val_t_con n = -entry_begin_end_.size();
  for (auto p : entry_begin_end_) n += p.second - p.first;
  return n;
}

}  // namespace nigiri::routing::meat::csa
