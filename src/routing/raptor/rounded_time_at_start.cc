#include "nigiri/routing/raptor/reconstruct.h"

#include "nigiri/routing/journey.h"
#include "nigiri/routing/query.h"
#include "nigiri/rt/frun.h"
#include "nigiri/timetable.h"

namespace nigiri::routing {

template <direction SearchDir>
void correct_rounded_time_at_start(timetable const& tt,
                                   rt_timetable const* rtt,
                                   query const& q,
                                   journey& j) {
  if constexpr (SearchDir == direction::kForward) {
    if (q.start_match_mode_ == location_match_mode::kIntermodal) {
      if (j.legs_.size() >= 2 && holds_alternative<offset>(j.legs_[0].uses_) &&
          holds_alternative<journey::run_enter_exit>(j.legs_[1].uses_)) {
        auto const rounding_error = j.legs_[1].dep_time_ - j.legs_[0].arr_time_;
        j.legs_[0].dep_time_ += rounding_error;
        j.legs_[0].arr_time_ += rounding_error;
        j.start_time_ += rounding_error;
      }
    } else {  // start at station
      if (!j.legs_.empty() &&
          holds_alternative<journey::run_enter_exit>(j.legs_[0].uses_)) {
        auto const ree = get<journey::run_enter_exit>(j.legs_[0].uses_);
        auto const fr = rt::frun{tt, rtt, ree.r_};
        auto const t_actual = fr[ree.stop_range_.from_].time(event_type::kDep);
        j.legs_[0].dep_time_ = t_actual;
        j.start_time_ = t_actual;
      }
    }
  } else {  // SearchDir == direction::kBackward
    if (q.start_match_mode_ == location_match_mode::kIntermodal) {
      if (j.legs_.size() >= 2 &&
          holds_alternative<offset>(rbegin(j.legs_)[0].uses_) &&
          holds_alternative<journey::run_enter_exit>(
              rbegin(j.legs_)[1].uses_)) {
        auto const rounding_error =
            rbegin(j.legs_)[0].dep_time_ - rbegin(j.legs_)[1].arr_time_;
        rbegin(j.legs_)[0].dep_time_ -= rounding_error;
        rbegin(j.legs_)[0].arr_time_ -= rounding_error;
        j.start_time_ -= rounding_error;
      }
    } else {  // start at station
      if (!j.legs_.empty() && holds_alternative<journey::run_enter_exit>(
                                  rbegin(j.legs_)[0].uses_)) {
        auto const ree = get<journey::run_enter_exit>(rbegin(j.legs_)[0].uses_);
        auto const fr = rt::frun{tt, rtt, ree.r_};
        auto const t_actual =
            fr[ree.stop_range_.to_ - 1].time(event_type::kArr);
        j.legs_.back().arr_time_ = t_actual;
        j.start_time_ = t_actual;
      }
    }
  }
}

}  // namespace nigiri::routing