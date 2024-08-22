#include "nigiri/routing/meat/csa/profile.h"

#include "nigiri/routing/limits.h"

namespace nigiri::routing::meat::csa {

profile_set::profile_set(timetable const& tt)
    : fp_dis_to_target_(tt.n_locations(),
                        std::numeric_limits<meat_t>::infinity()),
      fp_to_target_reset_list_(tt.n_locations()),
      fp_to_target_reset_list_end_{0},
      stop_reset_list_(tt.n_locations()),
      stop_reset_list_end_{0},
      entry_begin_end_(tt.n_locations()) {
  vector_map<location_idx_t, val_t_con> max_entry1(tt.n_locations(), 1);
  size_t n_entry1 = tt.n_locations();
  vector_map<location_idx_t, val_t_con> max_entry2(tt.n_locations(), 1);
  size_t n_entry2 = tt.n_locations();
  for (auto& c : tt.fwd_connections_) {
    if (!stop{c.dep_stop_}.in_allowed()) {
      continue;
    }
    auto n_c_in_tt =
        tt.bitfields_[tt.transport_traffic_days_[c.transport_idx_]].count();
    max_entry1[stop{c.dep_stop_}.location_idx()] += n_c_in_tt;
    n_entry1 += n_c_in_tt;
    max_entry2[stop{c.dep_stop_}.location_idx()] += kMaxSearchDays;
    n_entry2 += kMaxSearchDays;

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
        max_entry1[fp.target()] += n_c_in_tt;
        n_entry1 += n_c_in_tt;
        max_entry2[fp.target()] += kMaxSearchDays;
        n_entry2 += kMaxSearchDays;
      }
    }
    // TODO
    // dafür muss fp_prf_idx_ fest sein (nicht const& sondern const[in
    // meat_profile_computer])
    // for (auto const& fp :
    //     tt.locations_.footpaths_in_[0 /* TODO fp_prf_idx_*/]
    //                                [stop{c.dep_stop_}.location_idx()]) {
    //  max_entry1[fp.target()] += n_c_in_tt;
    //  n_entry1 += n_c_in_tt;
    //  max_entry2[fp.target()] += kMaxSearchDays;
    //  n_entry2 += kMaxSearchDays;
    //}
  }
  auto b = n_entry1 < n_entry2;
  auto const& max_entry = b ? max_entry1 : max_entry2;
  entry_ = std::vector<profile_entry>(b ? n_entry1 : n_entry2);

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

val_t_con profile_set::compute_entry_amount() const {
  val_t_con n = -entry_begin_end_.size();
  for (auto p : entry_begin_end_) n += p.second - p.first;
  return n;
}

}  // namespace nigiri::routing::meat::csa
