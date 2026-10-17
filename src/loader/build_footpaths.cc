#include "nigiri/loader/build_footpaths.h"

#include <cassert>
#include <cmath>
#include <algorithm>
#include <iterator>
#include <limits>
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
#include "utl/lookup.h"
#include "utl/parallel_for.h"

#include "nigiri/loader/build_lb_graph.h"
#include "nigiri/loader/init_finish.h"
#include "nigiri/loader/link_nearby_stations.h"
#include "nigiri/loader/merge_duplicates.h"
#include "nigiri/loader/transfer_rules.h"
#include "nigiri/loader/virtual_locations.h"
#include "nigiri/common/dial.h"
#include "nigiri/constants.h"
#include "nigiri/logging.h"
#include "nigiri/routing/dijkstra.h"
#include "nigiri/stop.h"
#include "nigiri/types.h"

namespace nigiri::loader {

namespace {

// Stop pairs named by an unqualified transfer rule (a station for each of its
// stops) with the longest duration of their rules (0 for a ban). A qualified
// rule binds only its trips, at their virtual locations.
hash_map<transfer_pair, duration_t> get_rule_walks(timetable const& tt) {
  auto const stops = [&](location_idx_t const l) {
    auto ls = std::vector<location_idx_t>{l};
    for (auto const c : tt.locations_.children_[l]) {
      if (!tt.locations_.is_virt(c)) {
        ls.push_back(c);
      }
    }
    return ls;
  };

  auto walks = hash_map<transfer_pair, duration_t>{};
  for (auto const& r : tt.transfer_rules_.rules_) {
    if (r.from_stop_ == r.to_stop_ || r.is_qualified(event_type::kArr) ||
        r.is_qualified(event_type::kDep)) {
      continue;
    }
    auto const d =
        r.duration_ == footpath::kMaxDuration ? duration_t{0} : r.duration_;
    for (auto const from : stops(r.from_stop_)) {
      for (auto const to : stops(r.to_stop_)) {
        if (from != to) {
          auto& x = walks[{from, to}];
          x = std::max(x, d);
        }
      }
    }
  }
  return walks;
}

// The walk alone, before transfer times (see get_beeline_walk).
std::optional<duration_t> get_beeline_minutes(
    timetable const& tt,
    hash_map<transfer_pair, duration_t> const& rule_walks,
    std::uint16_t const max_footpath_length,
    location_idx_t const a,
    location_idx_t const b) {
  auto const& eq = tt.locations_.equivalences_;
  auto const rule = utl::lookup(rule_walks, transfer_pair{a, b});
  if (!rule.has_value() && utl::find(eq[a], b) == end(eq[a]) &&
      utl::find(eq[b], a) == end(eq[b])) {
    return std::nullopt;
  }

  auto const& pos = tt.locations_.coordinates_[a];
  auto const dist = std::sqrt(
      geo::approx_squared_distance(pos, tt.locations_.coordinates_[b],
                                   geo::approx_distance_lng_degrees(pos)));
  auto const minutes =
      std::max({2.0, std::ceil((dist / kWalkSpeed) / 60.0),
                static_cast<double>(rule.value_or(duration_t{0}).count())});
  // A walk named by transfers.txt is not capped by max_footpath_length - only
  // by what a footpath can store (kMaxDuration means forbidden).
  auto const max =
      rule.has_value()
          ? footpath::kMaxDuration.count() - 1
          : std::min<int>(max_footpath_length, footpath::kMaxDuration.count());
  if (minutes > max) {
    return std::nullopt;
  }
  return duration_t{static_cast<duration_t::rep>(minutes)};
}

// The beeline fill-in walk from a to b, nullopt if there is none: between
// equivalent stops (at most max_footpath_length) and between stops named by a
// rule (not faster than the rule, not capped).
std::optional<duration_t> get_beeline_walk(
    timetable const& tt,
    hash_map<transfer_pair, duration_t> const& rule_walks,
    std::uint16_t const max_footpath_length,
    location_idx_t const a,
    location_idx_t const b) {
  auto const walk =
      get_beeline_minutes(tt, rule_walks, max_footpath_length, a, b);
  return walk.has_value()
             ? std::optional{max_with_transfer_times(tt, a, b, *walk)}
             : std::nullopt;
}

void add_if_not_exists(auto bucket, footpath const fp) {
  if (utl::none_of(bucket, [&](footpath const x) {
        return x.target() == fp.target();
      })) {
    bucket.push_back(fp);
  }
}

void add_equivalence_footpaths(
    timetable& tt,
    hash_map<transfer_pair, duration_t> const& rule_walks,
    std::uint16_t const max_footpath_length) {
  auto const add = [&](location_idx_t const a, location_idx_t const b) {
    auto const walk =
        get_beeline_minutes(tt, rule_walks, max_footpath_length, a, b);
    if (walk.has_value()) {
      add_if_not_exists(tt.locations_.preprocessing_footpaths_out_[a],
                        {b, *walk});
    }
  };

  for (auto l = location_idx_t{0U}; l != tt.n_locations(); ++l) {
    for (auto const eq : tt.locations_.equivalences_[l]) {
      if (eq != l) {
        add(l, eq);
        add(eq, l);
      }
    }
  }
}

// A footpath is one change: a walk A -> B -> C needs its own footpath. Adds
// the chains of the walks so far up to max_footpath_length walking minutes -
// transfer times come afterwards (walking through a stop is no change there).
void add_transitive_walks(timetable& tt,
                          std::uint16_t const max_footpath_length,
                          std::vector<std::size_t> const& n_loaded) {
  constexpr auto const kUnreached = std::numeric_limits<std::uint16_t>::max();
  struct search {
    std::vector<std::uint16_t> dist_;
    std::vector<location_idx_t> reached_;
    dial<routing::label, routing::get_bucket> pq_;
  };

  auto& fps = tt.locations_.preprocessing_footpaths_out_;
  auto const n = tt.n_locations();
  auto const max = static_cast<std::uint16_t>(std::min<unsigned>(
      max_footpath_length, footpath::kMaxDuration.count() - 1U));
  fps[location_idx_t{n - 1U}];  // resize
  auto chains = std::vector<std::vector<footpath>>(n);
  utl::parallel_for_run_threadlocal<search>(
      n, [&](search& s, std::size_t const i) {
        auto const from = location_idx_t{i};
        if (tt.locations_.is_virt(from) || fps[from].empty()) {
          return;
        }
        s.dist_.resize(n, kUnreached);
        s.pq_.n_buckets(static_cast<std::uint16_t>(max + 1U));
        s.pq_.clear();  // a default constructed dial is uninitialized
        s.dist_[i] = 0U;
        s.reached_.push_back(from);
        s.pq_.push(routing::label{from, 0U});
        while (!s.pq_.empty()) {
          auto const l = s.pq_.top();
          s.pq_.pop();
          if (s.dist_[to_idx(l.l_)] < l.d_) {
            continue;
          }
          for (auto const fp : fps[l.l_]) {
            auto const d = l.d_ + fp.duration().count();
            auto& dist = s.dist_[to_idx(fp.target())];
            if (d <= max && d < dist) {
              if (dist == kUnreached) {
                s.reached_.push_back(fp.target());
              }
              dist = static_cast<std::uint16_t>(d);
              s.pq_.push(routing::label{fp.target(), dist});
            }
          }
        }
        for (auto const l : s.reached_) {
          if (l != from) {
            chains[i].emplace_back(l, duration_t{s.dist_[to_idx(l)]});
          }
          s.dist_[to_idx(l)] = kUnreached;
        }
        s.reached_.clear();
      });

  // Keep the shortest walk per pair (footpaths the loader brought stay).
  for (auto l = location_idx_t{0U}; l != n; ++l) {
    auto bucket = fps[l];
    for (auto const c : chains[to_idx(l)]) {
      auto const it = utl::find_if(
          bucket, [&](footpath const x) { return x.target() == c.target(); });
      if (it == end(bucket)) {
        bucket.push_back(c);
      } else if (static_cast<std::size_t>(std::distance(begin(bucket), it)) >=
                     n_loaded[to_idx(l)] &&
                 c.duration() < it->duration()) {
        *it = c;
      }
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

// Virtual locations walk like their stop: calls fn(f, t) for every pair of
// the locations of stop a (a and its virtual locations) and of stop b.
void for_each_member_pair(timetable const& tt,
                          location_idx_t const a,
                          location_idx_t const b,
                          auto&& fn) {
  auto const from = [&](location_idx_t const f) {
    auto const to = [&](location_idx_t const t) {
      if (f != t) {
        fn(f, t);
      }
    };
    to(b);
    tt.locations_.for_each_virt(b, to);
  };
  from(a);
  tt.locations_.for_each_virt(a, from);
}

// A walk between two stops (or the stop's transfer time) connects every
// location of the one with every location of the other - except pairs a rule
// decides. The walks between the stops themselves exist already.
void add_virt_walks(timetable const& tt,
                    rule_index const& rule_fps,
                    mutable_fws_multimap<location_idx_t, footpath>& walks) {
  auto const& loc = tt.locations_;
  for (auto l = location_idx_t{0U}; l != tt.n_locations(); ++l) {
    auto const add = [&](location_idx_t const to, duration_t const d) {
      for_each_member_pair(
          tt, l, to, [&](location_idx_t const f, location_idx_t const t) {
            if ((f != l || t != to) && !rule_fps.contains(f, t)) {
              walks[f].emplace_back(t, d);
            }
          });
    };
    if (loc.transfer_time_[l] != kNoTransferAllowed) {
      add(l, to_fp_duration(loc.transfer_time_[l]));
    }
    for (auto const& fp : loc.preprocessing_footpaths_out_[l]) {
      add(fp.target(), fp.duration());
    }
  }
}

}  // namespace

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

void write_footpaths(timetable& tt, rule_index const& rule_fps) {
  auto& loc = tt.locations_;
  auto fps = vector_map<location_idx_t, std::vector<footpath>>{};
  fps.resize(tt.n_locations());
  for (auto l = location_idx_t{0U}; l != tt.n_locations(); ++l) {
    auto const add = [&](footpath const fp) {
      if (fp.target() != l) {
        fps[l].push_back(fp);
      }
    };
    for (auto const fp : loc.preprocessing_footpaths_out_[l]) {
      add(fp);
    }

    // Add rule footpaths (skip forbidden transfers).
    for (auto const fp : rule_fps.footpaths_[l]) {
      if (fp.duration() != footpath::kMaxDuration) {
        add(fp);
      }
    }
  }
  loc.preprocessing_footpaths_out_ = {};
  write_profile(tt, kDefaultProfile, fps);
}

void write_default_profile(
    timetable& tt,
    mutable_fws_multimap<location_idx_t, footpath> const& durations) {
  auto const timer = scoped_timer{"loader.footpath.default_profile"};

  auto& loc = tt.locations_;
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
  for (auto l = location_idx_t{0U}; l != durations.size(); ++l) {
    auto bucket = rule_footpaths[l];
    for (auto const fp : durations[l]) {
      utl::erase_if(
          bucket, [&](footpath const x) { return x.target() == fp.target(); });
      bucket.push_back(fp);
    }
  }

  // Remove every walk a rule decides.
  auto const rule_fps = rule_index{std::move(rule_footpaths)};
  for (auto l = location_idx_t{0U}; l != tt.n_locations(); ++l) {
    utl::erase_if(loc.preprocessing_footpaths_out_[l], [&](footpath const fp) {
      return rule_fps.contains(l, fp.target());
    });
  }

  auto walk_footpaths = mutable_fws_multimap<location_idx_t, footpath>{};
  add_virt_walks(tt, rule_fps, walk_footpaths);
  for (auto l = location_idx_t{0U}; l != walk_footpaths.size(); ++l) {
    for (auto const fp : walk_footpaths[l]) {
      loc.preprocessing_footpaths_out_[l].emplace_back(fp);
    }
  }

  write_footpaths(tt, rule_fps);
  build_lb_graph<direction::kForward>(tt, kDefaultProfile);
  build_lb_graph<direction::kBackward>(tt, kDefaultProfile);
}

void expand_profile_to_virts(timetable& tt, profile_idx_t const prf) {
  auto const& loc = tt.locations_;
  auto fps = vector_map<location_idx_t, std::vector<footpath>>{};
  fps.resize(tt.n_locations());
  for (auto l = location_idx_t{0U}; l != loc.footpaths_out_[prf].size(); ++l) {
    auto const add = [&](location_idx_t const to, duration_t const d) {
      for_each_member_pair(tt, l, to,
                           [&](location_idx_t const f, location_idx_t const t) {
                             fps[f].emplace_back(t, d);
                           });
    };
    if (loc.is_virt(l)) {
      continue;
    }
    if (loc.transfer_time_[l] != kNoTransferAllowed) {
      add(l, to_fp_duration(loc.transfer_time_[l]));
    }
    for (auto const fp : loc.footpaths_out_[prf][l]) {
      add(fp.target(), fp.duration());
    }
  }
  write_profile(tt, prf, fps);
  build_lb_graph<direction::kForward>(tt, prf);
  build_lb_graph<direction::kBackward>(tt, prf);
}

void expand_td_footpaths_to_virts(
    timetable const& tt,
    std::map<location_idx_t, std::vector<td_footpath>>& fps) {
  auto const& loc = tt.locations_;
  auto expanded = std::map<location_idx_t, std::vector<td_footpath>>{};
  for (auto const& [l, l_fps] : fps) {
    auto const add = [&](location_idx_t const to, unixtime_t const valid_from,
                         duration_t const d) {
      for_each_member_pair(tt, l, to,
                           [&](location_idx_t const f, location_idx_t const t) {
                             expanded[f].push_back({t, valid_from, d});
                           });
    };
    // Every location of l replaces its static walks, also without td walks.
    expanded[l];
    loc.for_each_virt(l, [&](location_idx_t const v) { expanded[v]; });
    if (loc.transfer_time_[l] != kNoTransferAllowed) {
      add(l, kNull, to_fp_duration(loc.transfer_time_[l]));
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

namespace {

void apply_transfer_rules(timetable& tt, shapes_storage* shapes) {
  // the walks grow with the virtual locations (see register_location)
  auto walks = tt.locations_.preprocessing_footpaths_out_;
  write_default_profile(tt);
  tt.locations_.preprocessing_footpaths_out_ = std::move(walks);
  auto durations = mutable_fws_multimap<location_idx_t, footpath>{};
  if (!add_virtual_locations(tt, shapes, durations)) {
    tt.locations_.preprocessing_footpaths_out_ = {};
    return;
  }
  auto const n = tt.n_locations();
  write_default_profile(tt, durations);

  // Routed walks of other profiles (computed before) reach the virtual
  // locations as well, importance counts at the root.
  for (auto prf = profile_idx_t{1U}; prf != kNProfiles; ++prf) {
    if (!tt.locations_.footpaths_out_[prf].empty()) {
      expand_profile_to_virts(tt, prf);
    }
  }
  tt.location_location_groups_.resize(n);
  if (!tt.location_areas_.empty()) {
    tt.location_areas_.resize(n);
  }
}

}  // namespace

void build_footpaths(timetable& tt,
                     finalize_options const opt,
                     shapes_storage* shapes) {
  if (opt.walks_) {
    link_nearby_stations(tt);  // the equivalences
  } else {
    auto const timer = scoped_timer{"loader.footpath.beelines"};

    // Walks between nearby and equivalent stops and their chains, in walking
    // minutes; footpaths the loader brought keep their durations.
    auto& fps = tt.locations_.preprocessing_footpaths_out_;
    fps[location_idx_t{tt.n_locations() - 1U}];  // resize
    auto n_loaded = std::vector<std::size_t>(tt.n_locations());
    for (auto l = location_idx_t{0U}; l != tt.n_locations(); ++l) {
      n_loaded[to_idx(l)] = fps[l].size();
    }
    auto const rule_walks = get_rule_walks(tt);
    link_nearby_stations(tt);
    add_equivalence_footpaths(tt, rule_walks, opt.max_footpath_length_);
    add_transitive_walks(tt, opt.max_footpath_length_, n_loaded);

    // Then transfer times, and a stop pair named by a rule never walks faster
    // than the rule.
    for (auto l = location_idx_t{0U}; l != tt.n_locations(); ++l) {
      auto bucket = fps[l];
      for (auto i = static_cast<std::uint32_t>(n_loaded[to_idx(l)]);
           i < bucket.size(); ++i) {
        auto const t = bucket[i].target();
        auto const walk = std::max(bucket[i].duration(),
                                   utl::lookup(rule_walks, transfer_pair{l, t})
                                       .value_or(duration_t{0}));
        bucket[i] = footpath{t, max_with_transfer_times(tt, l, t, walk)};
      }
    }
    for (auto const& [p, _] : rule_walks) {
      if (auto const walk = get_beeline_walk(
              tt, rule_walks, opt.max_footpath_length_, p.from_, p.to_);
          walk.has_value()) {
        add_if_not_exists(fps[p.from_], {p.to_, *walk});
      }
    }
  }

  if (opt.merge_dupes_intra_src_ || opt.merge_dupes_inter_src_) {
    merge_duplicates(tt, opt.merge_threshold_, opt.merge_dupes_intra_src_,
                     opt.merge_dupes_inter_src_, opt.merge_stats_dir_,
                     opt.src_tags_);
  }

  if (opt.walks_) {
    opt.walks_(tt);
  }
  if (opt.apply_transfer_rules_) {
    apply_transfer_rules(tt, shapes);
  } else {
    write_default_profile(tt);
  }
}

}  // namespace nigiri::loader
