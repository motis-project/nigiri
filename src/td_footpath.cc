#include "nigiri/td_footpath.h"

#include "utl/helpers/algorithm.h"

#include "nigiri/timetable.h"

namespace nigiri {

void expand_td_footpaths_to_virts(
    timetable const& tt,
    std::map<location_idx_t, std::vector<td_footpath>>& fps) {
  auto const& loc = tt.locations_;
  auto expanded = std::map<location_idx_t, std::vector<td_footpath>>{};
  for (auto const& [l, l_fps] : fps) {
    auto const add = [&](location_idx_t const to, unixtime_t const valid_from,
                         duration_t const d) {
      loc.for_each_member_pair(
          l, to, [&](location_idx_t const f, location_idx_t const t) {
            expanded[f].push_back({t, valid_from, d});
          });
    };
    // Every location of l replaces its static walks, also without td walks.
    expanded[l];
    loc.for_each_virt(l, [&](location_idx_t const v) { expanded[v]; });
    if (loc.transfer_time_[l] != kNoTransferAllowed) {
      add(l, kNull, duration_t{loc.transfer_time_[l]});
    }
    for (auto const& fp : l_fps) {
      add(fp.target_, fp.valid_from_, fp.duration_);
    }
  }
  for (auto& [_, x] : expanded) {
    utl::sort(x);
  }
  fps = std::move(expanded);
}

}  // namespace nigiri
