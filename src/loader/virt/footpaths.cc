#include "nigiri/loader/virt/footpaths.h"

#include <algorithm>
#include <tuple>
#include <vector>

#include "utl/equal_ranges_linear.h"
#include "utl/erase_duplicates.h"
#include "utl/erase_if.h"
#include "utl/helpers/algorithm.h"

#include "nigiri/loader/build_footpaths.h"
#include "nigiri/loader/build_lb_graph.h"
#include "nigiri/loader/transfer_rules.h"
#include "nigiri/for_each_meta.h"
#include "nigiri/logging.h"
#include "nigiri/timetable.h"
#include "nigiri/transfer_rule_sides.h"

namespace nigiri::loader {

rule_index::rule_index(
    mutable_fws_multimap<location_idx_t, footpath>&& footpaths)
    : footpaths_{std::move(footpaths)} {
  for (auto fps : footpaths_) {
    utl::sort(fps, [](footpath const a, footpath const b) {
      return a.target() < b.target();
    });
  }
}

std::optional<duration_t> rule_index::get(location_idx_t const from,
                                          location_idx_t const to) const {
  auto const b = footpaths_[from];
  auto const it = std::lower_bound(
      begin(b), end(b), to, [](footpath const fp, location_idx_t const t) {
        return fp.target() < t;
      });
  return it != end(b) && it->target() == to ? std::optional{it->duration()}
                                            : std::nullopt;
}

hash_map<transfer_pair, transfer_rule_idx_t> get_most_specific(
    timetable const& tt, interval<transfer_rule_idx_t> const rules) {
  auto most_specific = hash_map<transfer_pair, transfer_rule_idx_t>{};
  for (auto const rule : rules) {
    auto const& r = tt.transfer_rules_.rules_[rule];
    if (r.is_qualified()) {
      continue;
    }
    // Unqualified rules map to the stop and all descendants.
    routing::for_each_meta(
        tt, nullptr, routing::location_match_mode::kOnlyChildren, r.from_stop_,
        [&](location_idx_t const from) {
          routing::for_each_meta(
              tt, nullptr, routing::location_match_mode::kOnlyChildren,
              r.to_stop_, [&](location_idx_t const to) {
                if (from == to) {
                  return;
                }
                auto const [it, is_new] =
                    most_specific.emplace(transfer_pair{from, to}, rule);
                if (!is_new) {
                  it->second = get_more_specific(tt, it->second, rule);
                }
              });
        });
  }
  return most_specific;
}

// One footpath per stop pair a rule wins, with the rule's duration.
void add_rule_footpaths(
    timetable const& tt,
    hash_map<transfer_pair, transfer_rule_idx_t> const& most_specific,
    mutable_fws_multimap<location_idx_t, footpath>& footpaths) {
  auto const& rules = tt.transfer_rules_.rules_;
  for (auto const& [from_to, rule] : most_specific) {
    // Same base location, its transfer time: the walks of virtual locations.
    auto const d = rules[rule].duration_;
    auto const base = tt.base(from_to.from_);
    if (base == tt.base(from_to.to_) &&
        d == duration_t{tt.locations_.transfer_time_[base]}) {
      continue;
    }
    footpaths[from_to.from_].emplace_back(from_to.to_, d);
  }
}

mutable_fws_multimap<location_idx_t, footpath> get_rule_footpaths(
    timetable const& tt) {
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
        add_rule_footpaths(tt, most_specific, rule_footpaths);
      });

  // Resize to prevent out-of-bounds access.
  rule_footpaths[location_idx_t{tt.n_locations() - 1U}];
  return rule_footpaths;
}

rule_index get_rule_index(timetable const& tt) {
  return rule_index{get_rule_footpaths(tt)};
}

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
  build_lb_graph<direction::kForward>(tt, prf);
  build_lb_graph<direction::kBackward>(tt, prf);
}

// A walk between two stops (or the stop's transfer time) connects every
// location of the one with every location of the other - except pairs a rule
// decides.
void expand_to_virts(timetable& tt,
                     profile_idx_t const prf,
                     auto const& walks,
                     rule_index const* rule_fps) {
  auto const& loc = tt.locations_;
  auto fps = vector_map<location_idx_t, std::vector<footpath>>{};
  fps.resize(tt.n_locations());
  for (auto l = location_idx_t{0U}; l != walks.size(); ++l) {
    if (loc.is_virt(l)) {
      continue;
    }
    auto const add = [&](location_idx_t const to, duration_t const d) {
      loc.for_each_member_pair(
          l, to, [&](location_idx_t const f, location_idx_t const t) {
            if (rule_fps == nullptr || !rule_fps->get(f, t).has_value()) {
              fps[f].emplace_back(t, d);
            }
          });
    };
    if (loc.transfer_time_[l] != kNoTransferAllowed) {
      add(l, duration_t{loc.transfer_time_[l]});
    }
    for (auto const fp : walks[l]) {
      add(fp.target(), fp.duration());
    }
  }

  if (rule_fps != nullptr) {
    // Add rule footpaths (skip forbidden transfers).
    for (auto l = location_idx_t{0U}; l != tt.n_locations(); ++l) {
      for (auto const fp : rule_fps->footpaths_[l]) {
        if (fp.target() != l && fp.duration() != footpath::kBan) {
          fps[l].push_back(fp);
        }
      }
    }
  }
  write_profile(tt, prf, fps);
}

void write_default_profile(
    timetable& tt,
    mutable_fws_multimap<location_idx_t, footpath> const& walks,
    mutable_fws_multimap<location_idx_t, footpath> const& durations) {
  auto const timer = scoped_timer{"loader.footpath.default_profile"};

  auto rule_footpaths = get_rule_footpaths(tt);
  for (auto l = location_idx_t{0U}; l != durations.size(); ++l) {
    auto bucket = rule_footpaths[l];
    for (auto const fp : durations[l]) {
      utl::erase_if(
          bucket, [&](footpath const x) { return x.target() == fp.target(); });
      bucket.push_back(fp);
    }
  }

  auto const rule_fps = rule_index{std::move(rule_footpaths)};
  expand_to_virts(tt, kDefaultProfile, walks, &rule_fps);
}

void expand_profile_to_virts(timetable& tt, profile_idx_t const prf) {
  expand_to_virts(tt, prf, tt.locations_.footpaths_out_[prf], nullptr);
}

}  // namespace nigiri::loader
