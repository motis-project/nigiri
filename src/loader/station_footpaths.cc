#include "nigiri/loader/station_footpaths.h"

#include <cmath>
#include <algorithm>
#include <tuple>

#include "utl/erase_if.h"
#include "utl/helpers/algorithm.h"

#include "nigiri/logging.h"
#include "nigiri/timetable.h"

namespace nigiri::loader {

void remove_dominated(std::vector<station_footpath>& fps) {
  utl::sort(fps, [](station_footpath const& a, station_footpath const& b) {
    return std::tuple{a.target_, a.duration_, a.restriction_.flags_,
                      a.restriction_.fixed_} <
           std::tuple{b.target_, b.duration_, b.restriction_.flags_,
                      b.restriction_.fixed_};
  });
  auto is_dominated = std::vector<bool>(fps.size(), false);
  for (auto i = 0U; i != fps.size(); ++i) {
    for (auto j = i + 1U; j != fps.size() && fps[j].target_ == fps[i].target_;
         ++j) {
      if (is_dominated[i] || is_dominated[j]) {
        continue;
      }
      if (fps[i].dominates(fps[j])) {
        is_dominated[j] = true;
      } else if (fps[j].dominates(fps[i])) {
        is_dominated[i] = true;
      }
    }
  }
  auto i = 0U;
  utl::erase_if(fps,
                [&](station_footpath const&) { return is_dominated[i++]; });
}

station_footpaths_t get_station_footpaths(
    timetable const& tt, std::vector<station_route> const& routes) {
  auto fps = station_footpaths_t{};
  if (routes.empty()) {
    return fps;
  }

  auto n_footpaths = 0U;
  for (auto prf = profile_idx_t{0U}; prf != kNProfiles; ++prf) {
    auto const& p = kStationProfiles[prf];
    if (!p.has_value()) {
      continue;
    }

    auto& prf_fps = fps[prf];
    prf_fps.resize(tt.n_locations());
    for (auto const& r : routes) {
      if (r.is_entrance_route() ||
          (r.restriction_.flags_ & p->forbidden_) != 0U) {
        continue;
      }
      auto const seconds =
          static_cast<double>(r.walk_seconds_) * p->walk_factor_ +
          static_cast<double>(r.fixed_seconds_);
      auto const minutes = std::ceil(seconds / 60.0);
      if (minutes > static_cast<double>(footpath::kMax.count())) {
        continue;
      }
      prf_fps[r.from_].push_back(station_footpath{
          .target_ = r.to_,
          .duration_ = duration_t{static_cast<duration_t::rep>(minutes)},
          .restriction_ = r.restriction_});
    }
    for (auto& l_fps : prf_fps) {
      remove_dominated(l_fps);
      n_footpaths += l_fps.size();
    }
  }

  log(log_lvl::info, "loader.station_graph", "{} station footpaths",
      n_footpaths);
  return fps;
}

}  // namespace nigiri::loader
