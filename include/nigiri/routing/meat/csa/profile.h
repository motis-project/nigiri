#pragma once

#include <variant>

#include "nigiri/common/delta_t.h"
#include "nigiri/timetable.h"
#include "nigiri/types.h"

namespace nigiri::routing::meat::csa {

// using val_t_loc = location_idx_t::value_t;
using val_t_con = connection_idx_t::value_t;

struct walk {
location_idx_t from_;
footpath fp_;
};

struct profile_entry {
  delta_t dep_time_;
  meat_t meat_;
  std::variant<ride, walk> uses_;
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

// TODO make entry size dynamic ?
struct profile_set {
  profile_set(timetable const& tt);

  profile for_stop(location_idx_t stop_id) const {
    return {entry_.begin() + entry_begin_end_[stop_id].first,
            entry_.begin() + entry_begin_end_[stop_id].second};
  }

  void reset_stop(location_idx_t stop_id) {
    entry_begin_end_[stop_id].second = entry_begin_end_[stop_id].first + 1;
  }

  profile_entry const& early_stop_entry(location_idx_t stop_id) const {
    return entry_[entry_begin_end_[stop_id].second - 1];
  }

  void add_early_entry(location_idx_t stop_id, profile_entry const& e) {
    if (is_stop_empty(stop_id)) {
      stop_reset_list_[stop_reset_list_end_++] = stop_id;
    }
    entry_[entry_begin_end_[stop_id].second++] = e;
  }

  void replace_early_entry(location_idx_t stop_id, profile_entry const& e) {
    entry_[entry_begin_end_[stop_id].second - 1] = e;
  }

  bool is_stop_empty(location_idx_t stop) const {
    return entry_begin_end_[stop].second == entry_begin_end_[stop].first + 1;
  }

  val_t_con compute_entry_amount() const;

  void reset_fp_dis_to_target() {
    for (auto i = 0U; i < fp_to_target_reset_list_end_; ++i) {
      fp_dis_to_target_[fp_to_target_reset_list_[i]] =
          std::numeric_limits<meat_t>::infinity();
    }
    fp_to_target_reset_list_end_ = 0;
  }

  void reset_stop() {
    for (auto i = 0U; i < stop_reset_list_end_; ++i) {
      reset_stop(stop_reset_list_[i]);
    }
    stop_reset_list_end_ = 0;
  }
  void reset() {
    reset_stop();
    reset_fp_dis_to_target();
  }

  void set_fp_dis_to_target(location_idx_t l_idx, meat_t dis) {
    fp_dis_to_target_[l_idx] = dis;
    fp_to_target_reset_list_[fp_to_target_reset_list_end_++] = l_idx;
  }

  size_t n_entry_idxs() const { return entry_.size(); }

   auto global_index_of(
      std::reverse_iterator<std::vector<profile_entry>::const_iterator> const&
          rit) const {
    return rit.base() - entry_.begin() - 1;
  }

  vector_map<location_idx_t, meat_t> fp_dis_to_target_;
  std::vector<location_idx_t> fp_to_target_reset_list_;
  location_idx_t::value_t fp_to_target_reset_list_end_;
  std::vector<location_idx_t> stop_reset_list_;
  location_idx_t::value_t stop_reset_list_end_;
  vector_map<location_idx_t, std::pair<val_t_con, val_t_con>> entry_begin_end_;
  std::vector<profile_entry> entry_;
};

}  // namespace nigiri::routing::meat::csa