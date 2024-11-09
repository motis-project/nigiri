#pragma once

#include <variant>

#include "nigiri/common/delta_t.h"
#include "nigiri/routing/pareto_set.h"
#include "nigiri/timetable.h"
#include "nigiri/types.h"

namespace nigiri::routing::meat::raptor {

struct walk {
  location_idx_t from_;
  footpath fp_;
};

struct ride {
  ride() = default;
  ride(transport t, stop_idx_t const a, stop_idx_t const b)
      : t_{std::move(t)},
        stop_range_{std::min(a, b),
                    static_cast<stop_idx_t>(std::max(a, b) + 1U)} {}
  transport t_;
  interval<stop_idx_t> stop_range_;
};

struct profile_entry {
  bool dominates(profile_entry const& o) const {
    return dep_time_ >= o.dep_time_ && meat_ <= o.meat_;
  }

  friend bool operator<(profile_entry const& a, profile_entry const& b) {
    return a.dep_time_ < b.dep_time_;
  }

  delta_t dep_time_;
  meat_t meat_;
  std::variant<ride, walk> uses_;
};

struct profile {
  std::vector<profile_entry>::const_iterator begin_, end_;

  std::vector<profile_entry>::const_iterator begin() const { return begin_; }

  std::vector<profile_entry>::const_iterator end() const { return end_; }
};

struct profile_set {
  profile_set()
      : tt_{nullptr}, recompute_entry_amount_{true}, stop_reset_list_end_{0} {}

  void prepare_for_tt(timetable const& tt) {
    reset();
    entry_.resize(tt.n_locations(),
                  optional_sorted_pareto_set<profile_entry>{profile_entry{
                      std::numeric_limits<delta_t>::max(),
                      std::numeric_limits<meat_t>::infinity(),
                      walk{location_idx_t::invalid(), footpath()}}});
    entry_.shrink_to_fit();
    tt_ = &tt;
    l_start_idx_.resize(tt.n_locations(), 0);
    stop_reset_list_.resize(tt.n_locations());
    stop_reset_list_.shrink_to_fit();
  }

  profile for_unsorted_stop(location_idx_t stop_id) const {
    return {entry_[to_idx(stop_id)].begin(), entry_[to_idx(stop_id)].end()};
  }
  profile for_sorted_stop(location_idx_t stop_id) {
    auto& ps = entry_[to_idx(stop_id)];
    if (!ps.is_sorted()) {
      ps.sort();
      recompute_entry_amount_ = true;
    }
    return {ps.begin(), ps.end()};
  }
  auto for_stop_begin(location_idx_t stop_id, delta_t when) {
    auto& ps = entry_[to_idx(stop_id)];
    if (!ps.is_sorted()) {
      ps.sort();
      recompute_entry_amount_ = true;
    }
    return ps.lower_bound(
        profile_entry{when, std::numeric_limits<meat_t>::infinity(),
                      walk{location_idx_t::invalid(), footpath()}});
  }

  void reset_stop(location_idx_t stop_id) {
    entry_[to_idx(stop_id)].clear(
        profile_entry{std::numeric_limits<delta_t>::max(),
                      std::numeric_limits<meat_t>::infinity(),
                      walk{location_idx_t::invalid(), footpath()}});
    recompute_entry_amount_ = true;
  }

  profile_entry const& last_dep(location_idx_t stop_id) {
    auto& ps = entry_[to_idx(stop_id)];
    if (!ps.is_sorted()) {
      ps.sort();
      recompute_entry_amount_ = true;
    }
    return ps[ps.size() - 2];
  }

  bool add_entry(location_idx_t stop_id, profile_entry&& e) {
    if (is_stop_empty(stop_id)) {
      stop_reset_list_[stop_reset_list_end_++] = stop_id;
    }
    auto const [added, it1, it2] =
        entry_[to_idx(stop_id)].unsorted_add(std::move(e));
    recompute_entry_amount_ = added;
    return added;
  }

  bool is_stop_empty(location_idx_t stop) const {
    return entry_[to_idx(stop)].size() == 1;
  }

  size_t compute_entry_amount() const {
    size_t n_entries = 0;
    for (auto i = 0U; i < stop_reset_list_end_; ++i) {
      l_start_idx_[stop_reset_list_[i]] = n_entries;
      n_entries += entry_[to_idx(stop_reset_list_[i])].size();
    }
    recompute_entry_amount_ = false;
    return n_entries - stop_reset_list_end_;
  }

  void reset_stops() {
    for (auto i = 0U; i < stop_reset_list_end_; ++i) {
      reset_stop(stop_reset_list_[i]);
    }
    stop_reset_list_end_ = 0;
    recompute_entry_amount_ = true;
  }

  void reset() { reset_stops(); }

  auto n_entry_idxs() const {
    return compute_entry_amount() + stop_reset_list_end_;
  }

  auto global_index_of(
      std::vector<profile_entry>::const_iterator const& it) const {
    location_idx_t stop_idx;
    std::visit(
        utl::overloaded{
            [&](walk const& w) { stop_idx = w.from_; },
            [&](ride const& r) {
              auto const route_idx = tt_->transport_route_[r.t_.t_idx_];
              stop_idx =
                  stop{tt_->route_location_seq_[route_idx][r.stop_range_.from_]}
                      .location_idx();
            }},
        it->uses_);
    if (recompute_entry_amount_) {
      compute_entry_amount();
    }
    assert(&(*it) == &entry_[stop_idx.v_][static_cast<unsigned long>(
                         it - entry_[to_idx(stop_idx)].begin())]);
    return it - entry_[to_idx(stop_idx)].begin() +
           static_cast<long>(l_start_idx_[stop_idx]);
  }

  timetable const* tt_;
  mutable bool recompute_entry_amount_;
  mutable vector_map<location_idx_t, size_t> l_start_idx_;
  std::vector<location_idx_t> stop_reset_list_;
  location_idx_t::value_t stop_reset_list_end_;
  std::vector<optional_sorted_pareto_set<profile_entry>> entry_;
};

}  // namespace nigiri::routing::meat::raptor