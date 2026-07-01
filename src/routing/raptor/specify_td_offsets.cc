#include "nigiri/routing/raptor/reconstruct.h"

#include "nigiri/routing/journey.h"

namespace nigiri::routing {

using namespace nigiri;

void specify_td_offsets(query const& q, journey& j) {
  auto const& td_first_mile = q.td_start_;
  auto const& td_last_mile = q.td_dest_;

  if (!j.legs_.empty() &&
      std::holds_alternative<offset>(j.legs_.front().uses_)) {
    auto& o = std::get<offset>(j.legs_.front().uses_);
    if (auto const td_offsets = td_first_mile.find(o.target_);
        td_offsets != end(td_first_mile)) {
      if (auto const ret = get_td_duration<direction::kForward>(
              td_offsets->second, j.legs_.front().dep_time_)) {
        j.legs_.front().arr_time_ =
            j.legs_.front().dep_time_ + ret->second.duration_;
        o.duration_ = ret->second.duration_;
      }
    }
  }

  if (j.legs_.size() > 1 &&
      std::holds_alternative<offset>(j.legs_.back().uses_)) {
    auto& o = std::get<offset>(j.legs_.back().uses_);
    if (auto const td_offsets = td_last_mile.find(o.target_);
        td_offsets != end(td_last_mile)) {
      if (auto const ret = get_td_duration<direction::kBackward>(
              td_offsets->second, j.legs_.back().arr_time_)) {
        j.legs_.back().dep_time_ =
            j.legs_.back().arr_time_ - ret->second.duration_;
        o.duration_ = ret->second.duration_;
      }
    }
  }
}

}  // namespace nigiri::routing