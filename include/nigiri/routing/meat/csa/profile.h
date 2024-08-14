#pragma once

#include "nigiri/common/delta_t.h"
#include "nigiri/timetable.h"
#include "nigiri/types.h"

namespace nigiri::routing::meat::csa {

// using val_t_loc = location_idx_t::value_t;
using val_t_con = connection_idx_t::value_t;

struct profile_entry {
  delta_t dep_time_;
  meat_t meat_;
  ride ride_;
};

struct profile {
  std::vector<profile_entry>::const_iterator begin_, end_;

  std::reverse_iterator<std::vector<profile_entry>::const_iterator> begin()
      const {
    return std::reverse_iterator<std::vector<profile_entry>::const_iterator>{
        end_};
  }

  std::reverse_iterator<std::vector<profile_entry>::const_iterator> end()
      const {
    return std::reverse_iterator<std::vector<profile_entry>::const_iterator>{
        begin_};
  }
};

struct profile_set {
  profile_set(timetable const& tt);

  profile for_stop(location_idx_t stop_id) const {
    return {entry_.begin() + entry_begin_end_[stop_id].first,
            entry_.begin() + entry_begin_end_[stop_id].second};
  }

  void reset_stop(location_idx_t stop_id) {
    entry_begin_end_[stop_id].second = entry_begin_end_[stop_id].first + 1;
  }

  profile_entry early_stop_entry(location_idx_t stop_id) const {
    return entry_[entry_begin_end_[stop_id].second - 1];
  }

  void add_early_entry(location_idx_t stop_id, profile_entry e) {
    entry_[entry_begin_end_[stop_id].second++] = e;
  }

  void replace_early_entry(location_idx_t stop_id, profile_entry e) {
    entry_[entry_begin_end_[stop_id].second - 1] = e;
  }

  bool is_stop_empty(location_idx_t stop) const {
    return entry_begin_end_[stop].second == entry_begin_end_[stop].first + 1;
  }

  val_t_con compute_entry_amount() const;

  std::vector<profile_entry> entry_;
  vector_map<location_idx_t, std::pair<val_t_con, val_t_con>> entry_begin_end_;
};

}  // namespace nigiri::routing::meat::csa