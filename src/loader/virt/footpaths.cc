#include "nigiri/loader/virt/footpaths.h"

#include <algorithm>
#include <tuple>
#include <vector>

#include "utl/erase_duplicates.h"
#include "utl/helpers/algorithm.h"

#include "nigiri/loader/build_footpaths.h"
#include "nigiri/logging.h"
#include "nigiri/timetable.h"

namespace nigiri::loader {

// The footpaths of a profile: the shortest per target, shortest first (the
// target breaks ties so the built timetable is reproducible), and their mirror.
void write_profile(timetable& tt,
                   profile_idx_t const prf,
                   vector_map<location_idx_t, std::vector<footpath>>& fps) {
  auto const by_duration = [](footpath const a, footpath const b) {
    return std::tie(a.duration_, a.target_) < std::tie(b.duration_, b.target_);
  };
  auto& out = tt.locations_.footpaths_out_[prf];
  auto in = vector_map<location_idx_t, std::vector<footpath>>{};
  in.resize(tt.n_locations());
  out.clear();
  for (auto l = location_idx_t{0U}; l != tt.n_locations(); ++l) {
    utl::erase_duplicates(
        fps[l],
        [](footpath const a, footpath const b) {
          return std::tie(a.target_, a.duration_) <
                 std::tie(b.target_, b.duration_);
        },
        [](footpath const a, footpath const b) {
          return a.target_ == b.target_;
        });
    utl::sort(fps[l], by_duration);
    for (auto const fp : fps[l]) {
      in[fp.target()].emplace_back(l, fp.duration());
    }
    out.emplace_back(fps[l]);
    fps[l] = {};
  }
  auto& in_fps = tt.locations_.footpaths_in_[prf];
  in_fps.clear();
  for (auto l = location_idx_t{0U}; l != tt.n_locations(); ++l) {
    utl::sort(in[l], by_duration);
    in_fps.emplace_back(in[l]);
    in[l] = {};
  }
}

// Base footpaths apply to all virtual locations of their stops, except where
// a virtual location has its own footpath (forbidden: none at all).
void expand_to_virts(
    timetable& tt,
    profile_idx_t const prf,
    auto const& fallback_fps,
    mutable_fws_multimap<location_idx_t, footpath> const& qualified_rule_fps) {
  auto const& loc = tt.locations_;
  auto const has_virt_fp = [&](location_idx_t const a, location_idx_t const b) {
    return to_idx(a) < qualified_rule_fps.size() &&
           utl::any_of(
               qualified_rule_fps[a],
               [&](footpath const fp) { return fp.target() == b; });
  };
  auto fps = vector_map<location_idx_t, std::vector<footpath>>{};
  fps.resize(tt.n_locations());
  for (auto l = location_idx_t{0U}; l != fallback_fps.size(); ++l) {
    if (loc.is_virt(l)) {
      continue;
    }
    auto const add = [&](location_idx_t const to, duration_t const d) {
      loc.for_each_member_pair(
          l, to, [&](location_idx_t const a, location_idx_t const b) {
            if (!has_virt_fp(a, b)) {
              fps[a].emplace_back(b, d);
            }
          });
    };
    if (loc.transfer_time_[l] != kNoTransferAllowed) {
      add(l, loc.transfer_time_[l]);
    }
    for (auto const fp : fallback_fps[l]) {
      add(fp.target(), fp.duration());
    }
  }

  for (auto l = location_idx_t{0U}; l != qualified_rule_fps.size(); ++l) {
    for (auto const fp : qualified_rule_fps[l]) {
      if (fp.duration() != footpath::kBan) {
        fps[l].push_back(fp);
      }
    }
  }

  write_profile(tt, prf, fps);
}

void expand_profile_to_virts(timetable& tt, profile_idx_t const prf) {
  expand_to_virts(tt, prf, tt.locations_.footpaths_out_[prf], {});
}

void write_profiles(
    timetable& tt,
    mutable_fws_multimap<location_idx_t, footpath> const& fallback_fps,
    mutable_fws_multimap<location_idx_t, footpath> const& qualified_rule_fps) {
  auto const timer = scoped_timer{"loader.footpath.profiles"};

  expand_to_virts(tt, kDefaultProfile, fallback_fps, qualified_rule_fps);

  for (auto prf = profile_idx_t{1U}; prf != kNProfiles; ++prf) {
    if (!tt.locations_.footpaths_out_[prf].empty()) {
      expand_profile_to_virts(tt, prf);
    }
  }
}

}  // namespace nigiri::loader
