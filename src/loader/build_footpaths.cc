#include "nigiri/loader/build_footpaths.h"

#include <cassert>
#include <algorithm>
#include <map>
#include <optional>
#include <span>
#include <tuple>
#include <vector>

#include "geo/latlng.h"

#include "utl/equal_ranges_linear.h"
#include "utl/erase_duplicates.h"
#include "utl/erase_if.h"
#include "utl/helpers/algorithm.h"

#include "nigiri/loader/build_lb_graph.h"
#include "nigiri/loader/link_nearby_stations.h"
#include "nigiri/loader/merge_duplicates.h"
#include "nigiri/loader/transfer_rules.h"
#include "nigiri/constants.h"
#include "nigiri/logging.h"
#include "nigiri/types.h"

namespace nigiri::loader {

void add_equivalence_footpaths(timetable& tt,
                               std::uint16_t const max_footpath_length) {
  auto const max_duration =
      duration_t{static_cast<duration_t::rep>(std::min<std::uint32_t>(
          max_footpath_length,
          static_cast<std::uint32_t>(footpath::kMaxDuration.count())))};

  auto const add_if_not_exists = [](auto bucket, footpath const fp) {
    if (utl::none_of(bucket, [&](footpath const x) {
          return x.target() == fp.target();
        })) {
      bucket.push_back(fp);
    }
  };

  for (auto l = location_idx_t{0U}; l != tt.n_locations(); ++l) {
    if (tt.locations_.equivalences_[l].empty()) {
      continue;
    }
    auto const& pos = tt.locations_.coordinates_[l];
    auto const dist_lng_degrees = geo::approx_distance_lng_degrees(pos);
    for (auto const eq : tt.locations_.equivalences_[l]) {
      if (eq == l) {
        continue;
      }
      auto const dist = std::sqrt(geo::approx_squared_distance(
          pos, tt.locations_.coordinates_[eq], dist_lng_degrees));
      auto const minutes = std::max(2.0, std::ceil((dist / kWalkSpeed) / 60.0));
      if (minutes > static_cast<double>(max_duration.count())) {
        continue;
      }
      auto const duration = max_with_transfer_times(
          tt, l, eq, duration_t{static_cast<duration_t::rep>(minutes)});

      add_if_not_exists(tt.locations_.preprocessing_footpaths_out_[l],
                        {eq, duration});
      add_if_not_exists(tt.locations_.preprocessing_footpaths_out_[eq],
                        {l, duration});
    }
  }
}

struct rule_index {
  explicit rule_index(
      mutable_fws_multimap<location_idx_t, footpath>&& footpaths)
      : footpaths_{std::move(footpaths)} {
    for (auto fps : footpaths_) {
      utl::sort(fps, [](footpath const a, footpath const b) {
        return a.target() < b.target();
      });
    }
  }

  bool contains(location_idx_t const from, location_idx_t const to) const {
    auto const b = footpaths_[from];
    auto const it = std::lower_bound(
        begin(b), end(b), to, [](footpath const fp, location_idx_t const t) {
          return fp.target() < t;
        });
    return it != end(b) && it->target() == to;
  }

  mutable_fws_multimap<location_idx_t, footpath> footpaths_;
};

void collect_members(timetable const& tt,
                     location_idx_t const l,
                     std::vector<location_idx_t>& out) {
  out.assign({l});
  tt.locations_.for_each_virt(
      l, [&](location_idx_t const c) { out.push_back(c); });
}

void add_walk_hubs(timetable& tt,
                   rule_index const& rule_fps,
                   mutable_fws_multimap<location_idx_t, footpath>& walk) {
  auto l_and_virts = std::vector<location_idx_t>{};
  auto targets = std::vector<location_idx_t>{};
  auto egress = std::vector<location_idx_t>{};

  auto const is_footpath_allowed = [&](location_idx_t const from,
                                       location_idx_t const to) {
    return !rule_fps.contains(from, to);
  };

  for (auto l = location_idx_t{0U}; l != tt.n_locations(); ++l) {
    collect_members(tt, l, l_and_virts);

    // Targets grouped by duration:
    auto by_duration = std::map<duration_t, std::vector<location_idx_t>>{};

    if (l_and_virts.size() != 1U &&
        tt.locations_.transfer_time_[l] != kNoTransferAllowed) {
      by_duration[to_fp_duration(tt.locations_.transfer_time_[l])].push_back(l);
    }

    for (auto const& fp : tt.locations_.preprocessing_footpaths_out_[l]) {
      // Self-transfer -> covered by transfer time => skip.
      if (fp.target() == l) {
        continue;
      }

      // Covered by transfer rule => skip.
      if (rule_fps.contains(l, fp.target())) {
        continue;
      }

      // Neither side has virts -> skip.
      collect_members(tt, fp.target(), targets);
      if (l_and_virts.size() == 1U && targets.size() == 1U) {
        continue;
      }

      // Store duration -> target
      by_duration[fp.duration()].push_back(fp.target());
    }

    for (auto const& [d, stops] : by_duration) {
      egress.clear();

      for (auto const to : stops) {
        collect_members(tt, to, targets);

        // Marks slow pairs.
        auto coverage = hub_coverage{};
        for (auto const m : l_and_virts) {
          if (to == l && to_fp_duration(tt.locations_.transfer_time_[m]) > d) {
            coverage.mark_slow(m, m);
          }

          for (auto const r : rule_fps.footpaths_[m]) {
            if (r.duration() > d && tt.base(r.target()) == to) {
              coverage.mark_slow(m, r.target());
            }
          }
        }

        // No slow pairs.
        // Pool target with others that have the same duration.
        if (coverage.slow_from_.empty()) {
          egress.insert(end(egress), begin(targets), end(targets));
          continue;
        }

        // Target with slow pairs:
        // build its own hubs; slow pairs become footpaths unless ruled.
        add_hubs_or_footpaths(l_and_virts, targets, d, coverage,
                              is_footpath_allowed, tt, walk);
      }

      // Build hub for pairs that follow (base -> base) duration d.
      utl::erase_duplicates(egress);
      add_hubs_or_footpaths(l_and_virts, egress, d, hub_coverage{},
                            is_footpath_allowed, tt, walk);
    }
  }
}

bool is_hub_covered(timetable const& tt,
                    location_idx_t const from,
                    location_idx_t const to,
                    duration_t const max = footpath::kMaxDuration) {
  constexpr auto const p = kDefaultProfile;
  auto const& loc = tt.locations_;
  return utl::any_of(loc.hub_in_by_loc_[p][from], [&](hub_idx_t const h) {
    auto const out = loc.hub_out_[p][h];
    assert(std::is_sorted(begin(out), end(out)));
    return loc.hub_time_[p][h] <= max &&
           std::binary_search(begin(out), end(out), to);
  });
}

void override_footpaths_with_rules(timetable& tt, rule_index const& rule_fps) {
  for (auto l = location_idx_t{0U}; l != tt.n_locations(); ++l) {
    auto bucket = tt.locations_.preprocessing_footpaths_out_[l];

    // Remove every footpath covered by rube hubs / rule footpaths.
    utl::erase_if(bucket, [&](footpath const fp) {
      return rule_fps.contains(l, fp.target()) ||
             is_hub_covered(tt, l, fp.target());
    });

    // Add rule footpaths (skip forbidden transfers).
    for (auto const fp : rule_fps.footpaths_[l]) {
      if (fp.duration() != footpath::kMaxDuration) {
        bucket.push_back(fp);
      }
    }
  }
}

void write_layer(timetable& tt,
                 profile_idx_t const prf,
                 vector_map<location_idx_t, std::vector<footpath>>& out) {
  // shortest duration first, the order sort_footpaths() left the preprocessing
  // layers in; the target breaks ties so the built timetable is reproducible
  auto const by_duration = [](footpath const a, footpath const b) {
    return std::tie(a.duration_, a.target_) < std::tie(b.duration_, b.target_);
  };

  auto& loc = tt.locations_;
  auto in = vector_map<location_idx_t, std::vector<footpath>>{};
  in.resize(tt.n_locations());
  loc.footpaths_out_[prf].clear();
  for (auto l = location_idx_t{0U}; l != tt.n_locations(); ++l) {
    auto& fps = out[l];

    // Sort by target; keeps the shortest duration per target.
    utl::erase_duplicates(
        fps,
        [](footpath const a, footpath const b) {
          return std::tie(a.target_, a.duration_) <
                 std::tie(b.target_, b.duration_);
        },
        [](footpath const a, footpath const b) {
          return a.target_ == b.target_;
        });
    utl::sort(fps, by_duration);  // re-sort by duration

    loc.footpaths_out_[prf].emplace_back(fps);

    // Mirror out to in.
    for (auto const fp : fps) {
      in[fp.target()].emplace_back(l, fp.duration());
    }

    fps = {};
  }

  loc.footpaths_in_[prf].clear();
  for (auto l = location_idx_t{0U}; l != tt.n_locations(); ++l) {
    utl::sort(in[l], by_duration);
    loc.footpaths_in_[prf].emplace_back(in[l]);
    in[l] = {};
  }
}

void write_footpaths(timetable& tt) {
  auto out = vector_map<location_idx_t, std::vector<footpath>>{};
  out.resize(tt.n_locations());
  auto n_pruned = std::size_t{0U};
  for (auto l = location_idx_t{0U}; l != tt.n_locations(); ++l) {
    for (auto const fp : tt.locations_.preprocessing_footpaths_out_[l]) {
      if (fp.target() == l) {
        continue;
      }
      if (is_hub_covered(tt, l, fp.target(), fp.duration())) {
        ++n_pruned;
        continue;
      }
      out[l].push_back(fp);
    }
  }
  tt.locations_.preprocessing_footpaths_out_.clear();
  write_layer(tt, kDefaultProfile, out);

  log(log_lvl::info, "loader.footpath", "hub-covered footpaths dropped: {}",
      n_pruned);
}

void write_default_profile(timetable& tt) {
  auto const timer = scoped_timer{"loader.footpath.default_profile"};

  auto& loc = tt.locations_;
  loc.hub_in_[kDefaultProfile].clear();
  loc.hub_out_[kDefaultProfile].clear();
  loc.hub_time_[kDefaultProfile].clear();

  auto const& rules = tt.transfer_rules_.rules_;
  auto rule_footpaths = mutable_fws_multimap<location_idx_t, footpath>{};
  utl::equal_ranges_linear(
      rules,
      [](transfer_rule const& a, transfer_rule const& b) {
        // Group per feed to keep most_specific map small.
        return a.src_ == b.src_;
      },
      [&](auto const from, auto const to) {
        auto const feed_rules = interval{
            transfer_rule_idx_t{static_cast<std::size_t>(from - begin(rules))},
            transfer_rule_idx_t{static_cast<std::size_t>(to - begin(rules))}};
        auto const most_specific = get_most_specific(tt, feed_rules);
        add_rule_hubs(tt, most_specific, rule_footpaths);
      });

  // Resize to prevent out-of-bounds access.
  rule_footpaths[location_idx_t{tt.n_locations() - 1U}];

  index_hubs(tt);

  auto const rule_fps = rule_index{std::move(rule_footpaths)};
  override_footpaths_with_rules(tt, rule_fps);

  auto walk_footpaths = mutable_fws_multimap<location_idx_t, footpath>{};
  add_walk_hubs(tt, rule_fps, walk_footpaths);
  for (auto l = location_idx_t{0U}; l != walk_footpaths.size(); ++l) {
    for (auto const fp : walk_footpaths[l]) {
      loc.preprocessing_footpaths_out_[l].emplace_back(fp);
    }
  }
  index_hubs(tt);

  write_footpaths(tt);
  build_lb_graph<direction::kForward>(tt, kDefaultProfile);
  build_lb_graph<direction::kBackward>(tt, kDefaultProfile);
}

void build_footpaths(timetable& tt, finalize_options const opt) {
  {
    auto const timer = scoped_timer{"loader.footpath.beelines"};
    link_nearby_stations(tt);
    add_equivalence_footpaths(tt, opt.max_footpath_length_);
  }

  if (opt.merge_dupes_intra_src_ || opt.merge_dupes_inter_src_) {
    merge_duplicates(tt, opt.merge_threshold_, opt.merge_dupes_intra_src_,
                     opt.merge_dupes_inter_src_, opt.merge_stats_dir_,
                     opt.src_tags_);
  }

  write_default_profile(tt);
}

}  // namespace nigiri::loader
