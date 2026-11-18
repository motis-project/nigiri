#include "nigiri/loader/virt/footpaths.h"

#include <cassert>
#include <algorithm>
#include <optional>
#include <span>
#include <tuple>
#include <vector>

#include "utl/erase_duplicates.h"
#include "utl/helpers/algorithm.h"

#include "nigiri/loader/build_footpaths.h"
#include "nigiri/logging.h"
#include "nigiri/timetable.h"

namespace nigiri::loader {

// Footpath lists [unrestricted | restricted] (each part sorted by duration)
// + the records of the restricted parts, outgoing and incoming.
void write_profile(
    timetable& tt,
    profile_idx_t const prf,
    vector_map<location_idx_t, std::vector<footpath>>& fps,
    vector_map<location_idx_t, std::vector<station_footpath>>& restricted) {
  auto const by_duration = [](footpath const a, footpath const b) {
    return std::tie(a.duration_, a.target_) < std::tie(b.duration_, b.target_);
  };
  auto const by_restricted_duration = [](station_footpath const& a,
                                         station_footpath const& b) {
    return std::tie(a.duration_, a.target_) < std::tie(b.duration_, b.target_);
  };

  auto& loc = tt.locations_;
  auto& out = loc.footpaths_out_[prf];
  auto& in_fps = loc.footpaths_in_[prf];
  auto& records_out = loc.footpath_restrictions_out_[prf];
  auto& records_in = loc.footpath_restrictions_in_[prf];
  out.clear();
  in_fps.clear();
  records_out.clear();
  records_in.clear();

  auto const has_restricted =
      utl::any_of(restricted, [](auto&& x) { return !x.empty(); });
  auto in = vector_map<location_idx_t, std::vector<footpath>>{};
  auto in_restricted =
      vector_map<location_idx_t, std::vector<station_footpath>>{};
  in.resize(tt.n_locations());
  if (has_restricted) {
    in_restricted.resize(tt.n_locations());
  }

  auto bucket = std::vector<footpath>{};
  auto bucket_records = std::vector<footpath_restriction>{};
  auto const write = [&](auto& lists, auto& records,
                         std::vector<footpath>& plain,
                         std::vector<station_footpath>* r) {
    utl::sort(plain, by_duration);
    bucket = plain;
    bucket_records.clear();
    if (r != nullptr) {
      utl::sort(*r, by_restricted_duration);
      for (auto const& x : *r) {
        bucket.push_back(footpath{x.target_, x.duration_});
        bucket_records.push_back(x.restriction_);
      }
    }
    lists.emplace_back(bucket);
    if (has_restricted) {
      records.emplace_back(bucket_records);
    }
  };

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
    for (auto const fp : fps[l]) {
      in[fp.target()].emplace_back(l, fp.duration());
    }
    auto* r = static_cast<std::vector<station_footpath>*>(nullptr);
    if (has_restricted && l < restricted.size()) {
      r = &restricted[l];
      remove_dominated(*r);
      for (auto const& x : *r) {
        in_restricted[x.target_].push_back(
            station_footpath{.target_ = l,
                             .duration_ = x.duration_,
                             .restriction_ = x.restriction_});
      }
    }
    write(out, records_out, fps[l], r);
    fps[l] = {};
    if (r != nullptr) {
      *r = {};
    }
  }
  for (auto l = location_idx_t{0U}; l != tt.n_locations(); ++l) {
    write(in_fps, records_in, in[l],
          has_restricted ? &in_restricted[l] : nullptr);
    in[l] = {};
  }
}

void expand_to_virts(
    timetable& tt,
    profile_idx_t const prf,
    auto const& fallback_fps,
    mutable_fws_multimap<location_idx_t, footpath> const& qualified_rule_fps,
    vector_map<location_idx_t, std::vector<station_footpath>> const&
        station_fps,
    vector_map<location_idx_t, std::vector<station_footpath>> const& coverage,
    rule_pairs_t const& rule_pairs) {
  auto const& loc = tt.locations_;
  auto const get_virt_fp =
      [&](location_idx_t const a,
          location_idx_t const b) -> std::optional<duration_t> {
    if (to_idx(a) >= qualified_rule_fps.size()) {
      return std::nullopt;
    }
    for (auto const fp : qualified_rule_fps[a]) {
      if (fp.target() == b) {
        return fp.duration();
      }
    }
    return std::nullopt;
  };
  auto const get_station_fps =
      [&](location_idx_t const l) -> std::span<station_footpath const> {
    return l < station_fps.size() ? std::span{station_fps[l]}
                                  : std::span<station_footpath const>{};
  };
  // Covered pairs (station graph route in the default profile): no other
  // walks, also if no route is usable in this profile.
  auto const is_covered = [&](location_idx_t const from,
                              location_idx_t const to) {
    auto const s = from < coverage.size() ? std::span{coverage[from]}
                                          : std::span<station_footpath const>{};
    auto const it =
        std::lower_bound(begin(s), end(s), to,
                         [](station_footpath const& fp,
                            location_idx_t const x) { return fp.target_ < x; });
    return it != end(s) && it->target_ == to;
  };
  auto const get_fallback =
      [&](location_idx_t const from,
          location_idx_t const to) -> std::optional<duration_t> {
    for (auto const fp : fallback_fps[from]) {
      if (fp.target() == to) {
        return fp.duration();
      }
    }
    return std::nullopt;
  };

  auto fps = vector_map<location_idx_t, std::vector<footpath>>{};
  fps.resize(tt.n_locations());
  auto restricted = vector_map<location_idx_t, std::vector<station_footpath>>{};
  if (!station_fps.empty()) {
    restricted.resize(tt.n_locations());
  }
  for (auto l = location_idx_t{0U}; l != fallback_fps.size(); ++l) {
    if (loc.is_virt(l)) {
      continue;
    }
    auto const add = [&](location_idx_t const to, duration_t const d) {
      loc.for_each_member_pair(
          l, to, [&](location_idx_t const a, location_idx_t const b) {
            if (!get_virt_fp(a, b).has_value()) {
              fps[a].emplace_back(b, d);
            }
          });
    };
    if (loc.transfer_time_[l] != kNoTransferAllowed) {
      add(l, loc.transfer_time_[l]);
    }
    for (auto const fp : fallback_fps[l]) {
      if (!is_covered(l, fp.target())) {
        add(fp.target(), fp.duration());
      }
    }

    // Station graph routes replace all walks between their stops. Floor:
    // default profile = the walk after transfer times (missing: forbidden by
    // a rule), other profiles = the transfer times (routed alternatives
    // outside station graphs: none, like routed footpaths). Transfer rules
    // are authoritative: their duration replaces the route durations.
    auto const s = get_station_fps(l);
    for (auto it = begin(s); it != end(s);) {
      auto const to = it->target_;
      auto const group_end = std::find_if(
          it, end(s),
          [&](station_footpath const& x) { return x.target_ != to; });
      auto const group = std::span{it, group_end};
      it = group_end;

      auto const floor =
          prf == kDefaultProfile ? get_fallback(l, to)
          : is_covered(l, to)
              ? std::optional{max_with_transfer_times(tt, l, to, duration_t{0})}
              : std::optional{duration_t{0}};
      if (!floor.has_value()) {
        continue;
      }
      auto const is_ruled =
          prf == kDefaultProfile && rule_pairs.contains(pair{l, to});
      loc.for_each_member_pair(
          l, to, [&](location_idx_t const a, location_idx_t const b) {
            auto const virt_fp = get_virt_fp(a, b);
            if (virt_fp == footpath::kBan) {
              return;
            }
            auto const rule = virt_fp.has_value() ? virt_fp
                              : is_ruled          ? floor
                                                  : std::nullopt;
            for (auto const& fp : group) {
              restricted[a].push_back(station_footpath{
                  .target_ = b,
                  .duration_ = rule.value_or(std::max(fp.duration_, *floor)),
                  .restriction_ = fp.restriction_});
            }
          });
    }
  }

  for (auto l = location_idx_t{0U}; l != qualified_rule_fps.size(); ++l) {
    for (auto const fp : qualified_rule_fps[l]) {
      if (fp.duration() != footpath::kBan &&
          !is_covered(loc.get_base_idx(l), loc.get_base_idx(fp.target()))) {
        fps[l].push_back(fp);
      }
    }
  }

  write_profile(tt, prf, fps, restricted);
}

void expand_profile_to_virts(timetable& tt, profile_idx_t const prf) {
  expand_to_virts(tt, prf, tt.locations_.footpaths_out_[prf], {}, {}, {}, {});
}

void write_profiles(
    timetable& tt,
    mutable_fws_multimap<location_idx_t, footpath> const& fallback_fps,
    mutable_fws_multimap<location_idx_t, footpath> const& qualified_rule_fps,
    station_footpaths_t const& station_fps,
    rule_pairs_t const& rule_pairs) {
  auto const timer = scoped_timer{"loader.footpath.profiles"};

  expand_to_virts(tt, kDefaultProfile, fallback_fps, qualified_rule_fps,
                  station_fps[kDefaultProfile], station_fps[kDefaultProfile],
                  rule_pairs);

  for (auto prf = profile_idx_t{1U}; prf != kNProfiles; ++prf) {
    if (!tt.locations_.footpaths_out_[prf].empty()) {
      expand_to_virts(tt, prf, tt.locations_.footpaths_out_[prf], {},
                      station_fps[prf], station_fps[kDefaultProfile], {});
    }
  }
}

}  // namespace nigiri::loader
