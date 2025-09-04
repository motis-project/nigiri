#pragma once

#include <cmath>

#include "nigiri/common/delta_t.h"
#include "nigiri/routing/journey.h"
#include "nigiri/routing/raptor/raptor_state.h"
#include "nigiri/routing/transfer_time_settings.h"
#include "nigiri/timetable.h"
#include "nigiri/types.h"

namespace nigiri::routing {

struct mcraptor_label {

  bool dominates(mcraptor_label const& other_label) const {
    return this->arr_t_ < other_label.arr_t_;
  }
  bool dominates_or_equals(mcraptor_label const& other_label) const {
    return this->arr_t_ <= other_label.arr_t_;
  }

  delta_t arr_t_{kInvalidDelta<direction::kForward>};
  location_idx_t trip_l_{};
  location_idx_t fp_l_{};
  route_idx_t routeIdx_{};
  transport trip_;

};

struct mcraptor_bag {
  using bag_iterator = vector<mcraptor_label>::iterator;

  bag_iterator begin() {
    return labels_.begin();
  }
  bag_iterator end(){
    return labels_.end();
  }

  bool dominates(mcraptor_label const& other_label) const {
    return std::any_of(labels_.begin(),
                       labels_.end(),
                       [&](mcraptor_label bag_label) {
                         return bag_label.dominates(other_label);
    });
  }
  bool dominates_or_equals(mcraptor_label const& other_label) const {
    return std::any_of(labels_.begin(),
                       labels_.end(),
                       [&](mcraptor_label bag_label) {
                         return bag_label.dominates_or_equals(other_label);
    });
  }

  void add(mcraptor_label new_label) {
    labels_.erase(std::remove_if(
                      labels_.begin(),
                      labels_.end(),
                      [&](mcraptor_label bag_label) {
                        return new_label.dominates_or_equals(bag_label);
                      }), labels_.end());
    labels_.emplace_back(new_label);
  }
  mcraptor_label get(size_t index) const {
    if (index >= labels_.size()) {
      return mcraptor_label();
    }
    return labels_[index];
  }

  void reset() {
    labels_.clear();
  }
  bool empty() const {
    return labels_.empty();
  }

  void remove_invalid_trips() {
    labels_.erase(std::remove_if(
                      labels_.begin(),
                      labels_.end(),
                      [](mcraptor_label bag_label) {
                        return !bag_label.trip_.is_valid();
                      }), labels_.end());
  }

  vector<mcraptor_label> labels_;
};

}  // namespace nigiri::routing
