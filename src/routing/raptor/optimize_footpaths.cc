#include "nigiri/routing/raptor/reconstruct.h"

#include "nigiri/routing/journey.h"

namespace nigiri::routing {

void optimize_footpaths(timetable const& tt,
                        rt_timetable const* rtt,
                        query const& q,
                        raptor_state const& raptor_state,
                        journey& j,
                        date::sys_days const base,
                        day_idx_t const base_day_idx) {
  // optimize start

  // optimize end

  // optimize transfers
  for (auto i = 0U; i != j.legs_.size(); ++i) {
    auto& leg = j.legs_[i];
    if (!holds_alternative<journey::run_enter_exit>(leg.uses_)) {
      continue;
    }
    if (i + 2 < j.legs_.size() &&
        holds_alternative<journey::run_enter_exit>(j.legs_[i + 2].uses_)) {
      auto& transfer_leg = j.legs_[i + 2];
      auto fp_dur_best = get<footpath>(j.legs_[i + 1].uses_).duration();
      if (rtt != nullptr) {

      } else {
      }
    }
  }
}

}  // namespace nigiri::routing