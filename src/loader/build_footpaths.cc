#include "nigiri/loader/build_footpaths.h"

#include <cassert>
#include <map>
#include <optional>
#include <span>
#include <string>
#include <tuple>
#include <vector>

#include "geo/latlng.h"

#include "utl/enumerate.h"
#include "utl/erase_duplicates.h"
#include "utl/helpers/algorithm.h"
#include "utl/zip.h"

#include "nigiri/loader/build_lb_graph.h"
#include "nigiri/loader/link_nearby_stations.h"
#include "nigiri/loader/merge_duplicates.h"
#include "nigiri/loader/transfer_rules.h"
#include "nigiri/constants.h"
#include "nigiri/logging.h"
#include "nigiri/types.h"

namespace nigiri::loader {

// Returns the duration a walk between a and b takes at the very least, or
// nullopt if the two are so far apart that walking is out of the question
// (wormhole: a data error, not a transfer).
std::optional<u8_minutes> adjust_to_walk_speed(timetable const& tt,
                                               location_idx_t const a,
                                               location_idx_t const b,
                                               u8_minutes const duration) {
  constexpr auto const kMaxWalkDistance =
      std::numeric_limits<u8_minutes::rep>::max() * 60.0 * kWalkSpeed;

  auto const distance = geo::distance(tt.locations_.coordinates_[a],
                                      tt.locations_.coordinates_[b]);
  if (distance > kMaxWalkDistance) {
    log(log_lvl::error, "loader.footpath.adjust",
        "dropping footpath {} -> {}: {:.1f} km apart, not walkable",
        tt.locations_.ids_[a].view(), tt.locations_.ids_[b].view(),
        distance / 1000.0);
    return std::nullopt;
  }

  return u8_minutes{
      std::max(static_cast<duration_t::rep>(duration.count()),
               static_cast<duration_t::rep>(distance / kWalkSpeed / 60))};
}

// Walking transfers between equivalent stops (e.g. GTFS same-name / nearby /
// parent-child stops, HRDF meta stations): the loaders only collect the
// equivalences, the beeline footpaths are derived here for pairs without a
// footpath from the input data. With street routing, these are later replaced
// by routed footpaths (except where a transfer rule fixes the duration).
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
      bucket.emplace_back(fp);
    }
  };

  for (auto l = location_idx_t{0U}; l != tt.n_locations(); ++l) {
    if (tt.locations_.equivalences_[l].empty()) {
      continue;
    }
    auto const& pos = tt.locations_.coordinates_[l];
    auto const dist_lng_degrees = geo::approx_distance_lng_degrees(pos);
    for (auto const eq : tt.locations_.equivalences_[l]) {
      if (eq == l) {  // get_metas() contains the location itself
        continue;
      }
      auto const dist = std::sqrt(geo::approx_squared_distance(
          pos, tt.locations_.coordinates_[eq], dist_lng_degrees));
      // compare before narrowing: a stop group sits at (0,0), thousands of
      // kilometers away, and its walk does not fit into duration_t
      auto const minutes = std::max(2.0, std::ceil((dist / kWalkSpeed) / 60.0));
      if (minutes > static_cast<double>(max_duration.count())) {
        continue;
      }
      auto const duration = duration_t{static_cast<duration_t::rep>(minutes)};

      add_if_not_exists(tt.locations_.preprocessing_footpaths_out_[l],
                        {eq, duration});
      add_if_not_exists(tt.locations_.preprocessing_footpaths_out_[eq],
                        {l, duration});
    }
  }
}

// Sorted rule targets per location and the bases they sit at. A walk asks
// only "does this member have a rule into that stop", which is a binary
// search in the second list; the first is needed for the few that answer yes.
struct rule_index {
  explicit rule_index(timetable& tt) : tt_{tt} {
    auto const n = static_cast<std::size_t>(cista::to_idx(tt.n_locations()));
    auto const n_rules = std::min(
        static_cast<std::size_t>(tt.locations_.transfer_rule_fps_.size()), n);
    bases_.resize(n);
    for (auto l = location_idx_t{0U}; l != location_idx_t{n_rules}; ++l) {
      utl::sort(tt.locations_.transfer_rule_fps_[l],
                [](footpath const a, footpath const b) {
                  return a.target() < b.target();
                });
      auto& b = bases_[to_idx(l)];
      for (auto const fp : tt.locations_.transfer_rule_fps_[l]) {
        b.push_back(base_of(tt, fp.target()));
      }
      utl::erase_duplicates(b);
    }
  }

  // does `from` have any rule into `stop` or one of its virtual locations?
  bool any_at(location_idx_t const from, location_idx_t const stop) const {
    auto const& b = bases_[to_idx(from)];
    return std::binary_search(begin(b), end(b), stop);
  }

  // ... one that takes longer than a walk of `d`? Only such a rule can be
  // undercut by a hub handing out `d`: a faster one is written as a cell and
  // wins the minimum the routing takes anyway.
  bool any_slower_at(location_idx_t const from,
                     location_idx_t const stop,
                     duration_t const d) const {
    return any_at(from, stop) &&
           utl::any_of(
               tt_.locations_.transfer_rule_fps_[from], [&](footpath const r) {
                 return r.duration() > d && base_of(tt_, r.target()) == stop;
               });
  }

  bool ruled(location_idx_t const from, location_idx_t const to) const {
    if (to_idx(from) >= tt_.locations_.transfer_rule_fps_.size()) {
      return false;
    }
    auto const b = tt_.locations_.transfer_rule_fps_[from];
    auto const it = std::lower_bound(
        begin(b), end(b), to, [](footpath const fp, location_idx_t const t) {
          return fp.target() < t;
        });
    return it != end(b) && it->target() == to;
  }

  timetable& tt_;
  std::vector<std::vector<location_idx_t>> bases_;
};

// The members of a stop for walking purposes: itself and its virtual
// locations, which own no footpaths of their own.
void collect_members(timetable const& tt,
                     location_idx_t const l,
                     std::vector<location_idx_t>& out) {
  out.assign({l});
  for (auto const c : tt.locations_.children_[l]) {
    if (tt.locations_.types_[c] == location_type::kVirt) {
      out.push_back(c);
    }
  }
}

// The walks of the stops with virtual locations, as hubs. One hub stands for
// every pair of its two lists at one weight, so the footpaths of a stop that
// share a duration share a hub: the ingress is the same either way and the
// egress is their union. Pairs a rule speaks about must stay out of the lists
// - the same split the rule hubs use - and rectangles too small to pay for a
// hub become ordinary footpaths instead. This runs before the footpath lists
// are written, so the durations are adjusted here exactly as write_footpaths
// would, and it is the only place that decides.
template <typename Fps, typename AddExtra>
void build_walk_hubs_impl(timetable& tt,
                          bool const adjust_footpaths,
                          Fps&& fps_of,
                          AddExtra&& add_extra,
                          vecvec<hub_idx_t, location_idx_t>& walk_hub_in,
                          vecvec<hub_idx_t, location_idx_t>& walk_hub_out,
                          vector_map<hub_idx_t, duration_t>& walk_hub_time) {
  auto const idx = rule_index{tt};
  auto const has_rule = [&](location_idx_t const l, footpath const fp) {
    return to_idx(l) < tt.locations_.transfer_rule_fps_.size() &&
           utl::any_of(
               tt.locations_.transfer_rule_fps_[l],
               [&](footpath const r) { return r.target() == fp.target(); });
  };

  auto extra = mutable_fws_multimap<location_idx_t, footpath>{};
  auto members = std::vector<location_idx_t>{};
  auto targets = std::vector<location_idx_t>{};
  auto egress = std::vector<location_idx_t>{};

  auto& w_in = walk_hub_in;
  auto& w_out = walk_hub_out;
  auto& w_time = walk_hub_time;
  auto const emit = [&](std::vector<location_idx_t> const& ingress,
                        std::vector<location_idx_t> const& eg,
                        duration_t const d) {
    if (ingress.empty() || eg.empty()) {
      return;
    }
    if (ingress.size() * eg.size() <= ingress.size() + eg.size()) {
      for (auto const m : ingress) {  // a hub would not even be smaller
        for (auto const t : eg) {
          if (m != t && !idx.ruled(m, t)) {
            extra[m].emplace_back(t, d);
          }
        }
      }
      return;
    }
    w_in.emplace_back(ingress);
    w_out.emplace_back(eg);
    w_time.push_back(d);
  };

  for (auto l = location_idx_t{0U}; l != tt.n_locations(); ++l) {
    collect_members(tt, l, members);

    // group the footpaths by the duration they will end up with
    auto by_duration = std::map<duration_t, std::vector<location_idx_t>>{};
    for (auto const& fp : fps_of(l)) {
      if (fp.target() == l) {
        continue;
      }
      if (idx.ruled(l, fp.target())) {
        continue;  // a rule states this pair; it is not a walk and must not
                   // shape a rectangle - the layer may already contain it
                   // (street routing merges the rules before this runs)
      }
      auto d = fp.duration();
      if (adjust_footpaths && !has_rule(l, fp)) {
        auto const adjusted = adjust_to_walk_speed(tt, l, fp.target(), d);
        if (!adjusted.has_value()) {
          continue;  // dropped as unwalkable, by write_footpaths too
        }
        d = duration_t{adjusted->count()};
      }
      collect_members(tt, fp.target(), targets);
      if (members.size() == 1U && targets.size() == 1U) {
        continue;  // the footpath itself is the whole rectangle
      }
      by_duration[d].push_back(fp.target());
    }

    for (auto const& [d, stops] : by_duration) {
      // Targets no slower rule speaks about can share one hub, whatever stop
      // they belong to: same ingress, same weight. A rule faster than the walk
      // is no obstacle, its cell wins the minimum the routing takes. A target
      // some member has a slower rule into keeps its own pair of hubs -
      // merging it would make every member of the group restricted at every
      // stop of the group, and the pairs that fall out of the split would
      // have to be written one by one.
      egress.clear();
      for (auto const t_stop : stops) {
        collect_members(tt, t_stop, targets);

        auto split = hub_split{};
        for (auto const m : members) {
          if (!idx.any_slower_at(m, t_stop, d)) {
            continue;
          }
          for (auto const r : tt.locations_.transfer_rule_fps_[m]) {
            if (r.duration() > d && base_of(tt, r.target()) == t_stop) {
              split.mark(m, r.target());
            }
          }
        }
        if (split.slow_from_.empty()) {
          egress.insert(end(egress), begin(targets), end(targets));
          continue;
        }

        emit_split_hubs(members, targets, d, split, emit);
        for (auto const m : split.slow_from_) {
          for (auto const t : split.slow_to_) {
            if (m != t && !idx.ruled(m, t)) {
              extra[m].emplace_back(t, d);
            }
          }
        }
      }

      utl::erase_duplicates(egress);
      if (!egress.empty()) {
        emit(members, egress, d);
      }
    }
  }

  for (auto l = location_idx_t{0U}; l != tt.n_locations(); ++l) {
    for (auto const& fp : extra[l]) {
      add_extra(l, fp);
    }
  }
}

void build_walk_hubs(timetable& tt,
                     bool const adjust_footpaths,
                     vecvec<hub_idx_t, location_idx_t>& walk_hub_in,
                     vecvec<hub_idx_t, location_idx_t>& walk_hub_out,
                     vector_map<hub_idx_t, duration_t>& walk_hub_time) {
  build_walk_hubs_impl(
      tt, adjust_footpaths,
      [&](location_idx_t const l) -> decltype(auto) {
        return tt.locations_.preprocessing_footpaths_out_[l];
      },
      [&](location_idx_t const l, footpath const fp) {
        tt.locations_.preprocessing_footpaths_out_[l].emplace_back(fp);
      },
      walk_hub_in, walk_hub_out, walk_hub_time);
}

// Overwrite/insert the directed transfer edges emitted from transfer rules.
// They are authoritative: any generic footpath between the same pair is
// replaced and the duration survives the walk speed adjustment - a rule fixes
// the transfer time, which may be shorter or longer than the walking time. A
// pair a rule hub speaks for loses its footpath instead: a hub never
// overrides, the routing takes the minimum of hub and footpath, so the hub
// has to be the only statement for the pair.
void apply_transfer_rules(timetable& tt) {
  auto const& hub_in = tt.locations_.hub_in_[kDefaultProfile];
  auto const& hub_out = tt.locations_.hub_out_[kDefaultProfile];
  auto hubs_of = std::vector<std::vector<hub_idx_t>>(tt.n_locations());
  for (auto h = hub_idx_t{0U}; h != hub_idx_t{hub_in.size()}; ++h) {
    for (auto const l : hub_in[h]) {
      hubs_of[to_idx(l)].push_back(h);
    }
  }
  auto const hub_covers = [&](location_idx_t const from,
                              location_idx_t const to) {
    return utl::any_of(hubs_of[to_idx(from)], [&](hub_idx_t const h) {
      auto const o = hub_out[h];  // sorted, see write_transfer_rules
      return std::binary_search(begin(o), end(o), to);
    });
  };

  auto const n = std::min(
      static_cast<std::size_t>(tt.locations_.transfer_rule_fps_.size()),
      static_cast<std::size_t>(tt.n_locations()));
  for (auto l = location_idx_t{0U}; l != tt.n_locations(); ++l) {
    auto bucket = tt.locations_.preprocessing_footpaths_out_[l];
    // Self loops are dropped when the layer is written, so that is what a
    // walk becomes here when a hub speaks for its pair.
    for (auto& existing : bucket) {
      if (existing.target() != l && hub_covers(l, existing.target())) {
        existing = footpath{l, duration_t{0}};
      }
    }
    if (to_idx(l) >= n) {
      continue;
    }
    for (auto const fp : tt.locations_.transfer_rule_fps_[l]) {
      if (fp.duration() == footpath::kMaxDuration) {
        // transfers.txt type 3: the pair is not a transfer at all. Written as a
        // very long footpath the routing can still take it - a trip leaving
        // that many hours later would turn the ban into a journey. The walk
        // between the two has to go as well, or it survives underneath the
        // ban - it becomes a self loop like a hub-covered walk above.
        for (auto& existing : bucket) {
          if (existing.target() == fp.target()) {
            existing = footpath{l, duration_t{0}};
          }
        }
        continue;
      }
      auto replaced = false;
      for (auto& existing : bucket) {  // duplicate targets: replace them all
        if (existing.target() == fp.target()) {
          existing = fp;
          replaced = true;
        }
      }
      if (!replaced) {
        bucket.emplace_back(fp);
      }
    }
  }
}

// Writes the default profile footpaths: the outgoing footpaths as collected
// (deduplicated, sorted by target so that consumers can set-operate them
// against other target-sorted sequences), the incoming footpaths as their
// mirror.
void write_footpaths(timetable& tt, bool const adjust_footpaths) {
  auto const has_rule = [&](location_idx_t const l, footpath const fp) {
    return to_idx(l) < tt.locations_.transfer_rule_fps_.size() &&
           utl::any_of(
               tt.locations_.transfer_rule_fps_[l],
               [&](footpath const r) { return r.target() == fp.target(); });
  };

  // shortest duration first, the order sort_footpaths() left the preprocessing
  // layers in; the target breaks ties so the built timetable is reproducible
  auto const by_duration = [](footpath const a, footpath const b) {
    return std::tie(a.duration_, a.target_) < std::tie(b.duration_, b.target_);
  };

  auto fps = std::vector<footpath>{};
  auto fps_in = mutable_fws_multimap<location_idx_t, footpath>{};
  for (auto l = location_idx_t{0U}; l != tt.n_locations(); ++l) {
    fps.clear();
    for (auto fp : tt.locations_.preprocessing_footpaths_out_[l]) {
      if (fp.target() == l) {
        continue;
      }
      if (adjust_footpaths) {
        auto const adjusted =
            adjust_to_walk_speed(tt, l, fp.target(), fp.duration());
        if (!adjusted.has_value()) {
          continue;
        }
        if (!has_rule(l, fp)) {
          fp = footpath{fp.target(), *adjusted};
        }
      }
      fps.push_back(fp);
    }

    utl::erase_duplicates(
        fps,
        [](footpath const a, footpath const b) {
          return std::tie(a.target_, a.duration_) <
                 std::tie(b.target_, b.duration_);
        },
        [](footpath const a, footpath const b) {
          return a.target_ == b.target_;
        });  // also sorts; keeps the shortest duration per target

    utl::sort(fps, by_duration);
    tt.locations_.footpaths_out_[kDefaultProfile].emplace_back(fps);
    for (auto const fp : fps) {
      fps_in[fp.target()].emplace_back(l, fp.duration());
    }
  }

  for (auto l = location_idx_t{0U}; l != tt.n_locations(); ++l) {
    fps.clear();
    for (auto const fp : fps_in[l]) {
      fps.push_back(fp);
    }
    utl::sort(fps, by_duration);
    tt.locations_.footpaths_in_[kDefaultProfile].emplace_back(fps);
  }

  tt.locations_.preprocessing_footpaths_out_.clear();
}

// Builds the transfer hub member lists: one unrestricted and one restricted
// hub per stop that has virtual locations. A hub always delivers at the stop's
// transfer time, so nothing slower than that may become derivable through it.
//
//   hub           from                         to
//   ------------  ---------------------------  ----------------------
//   unrestricted  nothing slow starts here     all members
//   restricted    something slow starts here   nothing slow ends here
//   (neither)     slower than the stop itself  -
//
// A member is the stop itself or one of its virtual locations. The last row
// bars sources only: every hub holds its own sources in its to list too, so
// such a member would derive its own cell at the stop's transfer time - the
// exact value its own rule overrides. Nothing bars it from a to list, because
// a transfer costs what the rule for that pair says, and its own rule speaks
// only for the pair with itself.
//
// is_rebuild: the timetable is finished and only its walks changed (see
// rebuild_default_profile). The rule-derived hubs [0, n_rule_hubs_) depend on
// the rules and change times alone, so they are kept as they are and only the
// walk hubs behind them are replaced.
void build_hubs(timetable& tt,
                vecvec<hub_idx_t, location_idx_t> const& walk_hub_in,
                vecvec<hub_idx_t, location_idx_t> const& walk_hub_out,
                vector_map<hub_idx_t, duration_t> const& walk_hub_time,
                bool const is_rebuild = false) {
  auto const n = tt.n_locations();

  // Which members a slower transfer starts at / leads to. Taken from the
  // rules rather than from the footpaths, because the footpath layer is not
  // always the loader's to write - and because the loader decided what to
  // leave out from exactly this, so both sides have to read the same source.
  auto split = hub_split{};
  {
    auto const n_rules = std::min(
        static_cast<std::size_t>(tt.locations_.transfer_rule_fps_.size()),
        static_cast<std::size_t>(n));
    for (auto l = location_idx_t{0U}; l != location_idx_t{n_rules}; ++l) {
      auto const base = base_of(tt, l);
      for (auto const fp : tt.locations_.transfer_rule_fps_[l]) {
        if (base_of(tt, fp.target()) == base &&
            fp.duration() > tt.locations_.transfer_time_[base]) {
          split.mark(l, fp.target());
        }
      }
    }
  }

  auto in = vecvec<hub_idx_t, location_idx_t>{};
  auto out = vecvec<hub_idx_t, location_idx_t>{};
  auto time = vector_map<hub_idx_t, duration_t>{};
  auto in_by_loc = mutable_fws_multimap<location_idx_t, hub_idx_t>{};
  auto out_by_loc = mutable_fws_multimap<location_idx_t, hub_idx_t>{};

  auto const add_hub = [&](std::span<location_idx_t const> const ingress,
                           std::span<location_idx_t const> const egress,
                           duration_t const d) {
    if (ingress.empty() || egress.empty()) {
      return;
    }

    auto const h = hub_idx_t{in.size()};

    in.emplace_back(ingress);
    out.emplace_back(egress);
    time.push_back(d);

    for (auto const l : ingress) {
      in_by_loc[l].push_back(h);
    }
    for (auto const l : egress) {
      out_by_loc[l].push_back(h);
    }
  };

  // hubs the loader already emitted for constant-valued rule cross products
  // - on a rebuild every rule-derived one, the per-stop hubs below included
  auto const n_kept =
      is_rebuild ? hub_idx_t{tt.locations_.n_rule_hubs_}
                 : hub_idx_t{tt.locations_.hub_in_[kDefaultProfile].size()};
  for (auto h = hub_idx_t{0U}; h != n_kept; ++h) {
    auto const i = tt.locations_.hub_in_[kDefaultProfile][h];
    auto const o = tt.locations_.hub_out_[kDefaultProfile][h];
    add_hub({i.data(), i.size()}, {o.data(), o.size()},
            tt.locations_.hub_time_[kDefaultProfile][h]);
  }

  auto has_virts = std::vector<bool>(n, false);
  for (auto l = location_idx_t{0U}; l != location_idx_t{n}; ++l) {
    if (tt.locations_.types_[l] == location_type::kVirt) {
      has_virts[to_idx(tt.locations_.parents_[l])] = true;
    }
  }

  auto members = std::vector<location_idx_t>{};
  auto sources = std::vector<location_idx_t>{};
  auto const emit = [&](std::vector<location_idx_t> const& ingress,
                        std::vector<location_idx_t> const& egress,
                        duration_t const d) {
    add_hub({ingress.data(), ingress.size()}, {egress.data(), egress.size()},
            d);
  };
  for (auto base = location_idx_t{0U}; base != location_idx_t{n}; ++base) {
    if (is_rebuild || !has_virts[to_idx(base)]) {
      continue;
    }

    members.assign({base});
    for (auto const c : tt.locations_.children_[base]) {
      if (tt.locations_.types_[c] == location_type::kVirt) {
        members.push_back(c);
      }
    }

    // Only transfers between members count: the hubs never derive a pair
    // targeting a non-member, so only those can be undercut. Real stops can
    // carry same-stop edges to non-member children (e.g. equivalence
    // beelines) that must not flip the classification the loader elided
    // against.
    auto const d = tt.locations_.transfer_time_[base];
    if (d == kNoTransfer) {
      continue;  // nobody changes here, so nothing is derived at "d"
    }
    // a member slower than the stop itself feeds no hub: every hub holds its
    // own sources in its target list too, so it would derive its own cell at
    // the stop's time - the exact value its own rule overrides
    sources.clear();
    for (auto const m : members) {
      if (m == base || tt.locations_.transfer_time_[m] <= d) {
        sources.push_back(m);
      }
    }
    emit_split_hubs(sources, members, duration_t{d}, split, emit);
  }

  // the walks come last: everything before them is rule-derived
  tt.locations_.n_rule_hubs_ = static_cast<std::uint32_t>(in.size());
  for (auto h = hub_idx_t{0U}; h != hub_idx_t{walk_hub_in.size()}; ++h) {
    auto const w_i = walk_hub_in[h];
    auto const w_o = walk_hub_out[h];
    add_hub({w_i.data(), w_i.size()}, {w_o.data(), w_o.size()},
            walk_hub_time[h]);
  }

  tt.locations_.hub_in_[kDefaultProfile] = std::move(in);
  tt.locations_.hub_out_[kDefaultProfile] = std::move(out);
  tt.locations_.hub_time_[kDefaultProfile] = std::move(time);
  tt.locations_.hub_in_by_loc_[kDefaultProfile].clear();
  tt.locations_.hub_out_by_loc_[kDefaultProfile].clear();
  for (auto l = location_idx_t{0U}; l != location_idx_t{n}; ++l) {
    tt.locations_.hub_in_by_loc_[kDefaultProfile].emplace_back(in_by_loc[l]);
    tt.locations_.hub_out_by_loc_[kDefaultProfile].emplace_back(out_by_loc[l]);
  }
}

// The hubs derive most of the foot layer a second time: measured on CH, 3.75M
// of the 4.59M stored edges are exactly the pair a hub already hands out, at
// the same weight. The routing takes the minimum of both, so keeping the
// footpath cannot change an answer - it only makes the footpath phase walk
// five times more edges than it has to. Drop the ones a hub covers at the same
// weight or better and let the hub stand for them.
void prune_hub_covered_footpaths(timetable& tt) {
  constexpr auto const p = kDefaultProfile;
  auto const n = static_cast<std::size_t>(cista::to_idx(tt.n_locations()));
  auto const& hub_out = tt.locations_.hub_out_[p];
  auto const& hub_time = tt.locations_.hub_time_[p];
  auto const& hubs_of = tt.locations_.hub_in_by_loc_[p];
  // does a hub hand out this pair at the footpath's weight or better? Every
  // producer of a hub list keeps it sorted (write_transfer_rules sorts, the
  // stop and walk hubs list members in index order), so one binary search
  // per hub of the source answers it - without spelling out the pairs, of
  // which a stop with hundreds of virtual locations has tens of thousands.
  auto const covered = [&](location_idx_t const l, footpath const fp) {
    return to_idx(l) < hubs_of.size() &&
           utl::any_of(
               hubs_of[l],
               [&](hub_idx_t const h) {
                 auto const o = hub_out[h];
                 assert(std::is_sorted(begin(o), end(o)));
                 return hub_time[h] <= fp.duration() &&
                        std::binary_search(begin(o), end(o), fp.target());
               });
  };

  auto out = std::vector<std::vector<footpath>>(n);
  auto n_pruned = std::size_t{0U}, n_kept = std::size_t{0U};
  for (auto l = location_idx_t{0U}; l != tt.n_locations(); ++l) {
    for (auto const fp : tt.locations_.footpaths_out_[p][l]) {
      if (covered(l, fp)) {
        ++n_pruned;  // the hub already delivers this pair, at least as fast
        continue;
      }
      out[to_idx(l)].push_back(fp);
      ++n_kept;
    }
  }

  auto fps_out = vecvec<location_idx_t, footpath>{};
  auto fps_in = mutable_fws_multimap<location_idx_t, footpath>{};
  for (auto l = location_idx_t{0U}; l != tt.n_locations(); ++l) {
    auto const& fps = out[to_idx(l)];
    fps_out.emplace_back(fps);
    for (auto const fp : fps) {
      fps_in[fp.target()].emplace_back(l, fp.duration());
    }
  }
  tt.locations_.footpaths_out_[p] = std::move(fps_out);
  tt.locations_.footpaths_in_[p].clear();
  for (auto l = location_idx_t{0U}; l != tt.n_locations(); ++l) {
    tt.locations_.footpaths_in_[p].emplace_back(fps_in[l]);
  }

  log(log_lvl::info, "loader.footpath",
      "hub-covered footpaths: {} dropped, {} kept", n_pruned, n_kept);
}

void rebuild_default_profile(
    timetable& tt,
    vector_map<location_idx_t, std::vector<footpath>> const& walks) {
  constexpr auto const p = kDefaultProfile;
  auto const no_hubs = vecvec<hub_idx_t, location_idx_t>{};

  // The old walk hubs go first: apply_transfer_rules drops every walk a hub
  // speaks for, which may only be said of the rule-derived ones.
  build_hubs(tt, no_hubs, no_hubs, {}, true);

  auto& pending = tt.locations_.preprocessing_footpaths_out_;
  pending.clear();
  for (auto l = location_idx_t{0U}; l != tt.n_locations(); ++l) {
    auto bucket = pending.emplace_back();
    if (to_idx(l) < walks.size()) {
      for (auto const fp : walks[l]) {
        bucket.push_back(fp);
      }
    }
  }

  // the same steps build_footpaths takes, without the walk speed adjustment:
  // these durations are not estimates
  apply_transfer_rules(tt);
  auto walk_hub_in = vecvec<hub_idx_t, location_idx_t>{};
  auto walk_hub_out = vecvec<hub_idx_t, location_idx_t>{};
  auto walk_hub_time = vector_map<hub_idx_t, duration_t>{};
  build_walk_hubs(tt, false, walk_hub_in, walk_hub_out, walk_hub_time);
  tt.locations_.footpaths_out_[p].clear();
  tt.locations_.footpaths_in_[p].clear();
  write_footpaths(tt, false);
  build_hubs(tt, walk_hub_in, walk_hub_out, walk_hub_time, true);
  prune_hub_covered_footpaths(tt);

  build_lb_graph<direction::kForward>(tt, p);
  build_lb_graph<direction::kBackward>(tt, p);
}

// Track locations (HRDF) have no position of their own and are not listed as
// equivalent stations, so nothing derives their transfers: connect every
// track with its station and with the other tracks of that station at the
// station's transfer time.
void link_tracks_with_station(timetable& tt) {
  auto const is_track = [&](location_idx_t const l) {
    return tt.locations_.types_[l] == location_type::kGeneratedTrack;
  };
  for (auto const [i, children] : utl::enumerate(tt.locations_.children_)) {
    auto const parent = location_idx_t{i};
    auto const t = tt.locations_.walk_transfer_time(parent);
    for (auto const child : children) {
      if (!is_track(child)) {
        continue;
      }
      tt.locations_.preprocessing_footpaths_out_[parent].emplace_back(child, t);
      tt.locations_.preprocessing_footpaths_out_[child].emplace_back(parent, t);
      for (auto const other : children) {
        if (other != child && is_track(other)) {
          tt.locations_.preprocessing_footpaths_out_[child].emplace_back(other,
                                                                         t);
        }
      }
    }
  }
}

void build_footpaths(timetable& tt, finalize_options const opt) {
  // Covers locations created after the last transfers.txt was read (virtual
  // locations, locations from feeds without transfers.txt).
  tt.locations_.sync_base_transfer_time();

  // tt.bin is always a complete timetable on its own: the walks are beelines
  // here, and whoever computes a routed layer afterwards writes it into
  // tt_ext.bin instead of taking anything away from this one.
  {
    auto const timer = scoped_timer{"loader.footpath.beelines"};
    link_tracks_with_station(tt);
    link_nearby_stations(tt);
    add_equivalence_footpaths(tt, opt.max_footpath_length_);
  }

  if (opt.merge_dupes_intra_src_ || opt.merge_dupes_inter_src_) {
    merge_duplicates(tt, opt.merge_threshold_, opt.merge_dupes_intra_src_,
                     opt.merge_dupes_inter_src_, opt.merge_stats_dir_,
                     opt.src_tags_);
  }

  // The rules first: their cells join the layer and the walks a rule hub
  // speaks for leave it, so the walk hubs below see the final walks. Then
  // the walks of every stop with virtual locations become hubs instead of
  // one copy per virtual location.
  {
    auto const timer = scoped_timer{"loader.footpath.rules"};
    apply_transfer_rules(tt);
  }
  auto walk_hub_in = vecvec<hub_idx_t, location_idx_t>{};
  auto walk_hub_out = vecvec<hub_idx_t, location_idx_t>{};
  auto walk_hub_time = vector_map<hub_idx_t, duration_t>{};
  {
    auto const timer = scoped_timer{"loader.footpath.walk_hubs"};
    build_walk_hubs(tt, opt.adjust_footpaths_, walk_hub_in, walk_hub_out,
                    walk_hub_time);
  }
  {
    auto const timer = scoped_timer{"loader.footpath.write"};
    write_footpaths(tt, opt.adjust_footpaths_);
  }
  {
    auto const timer = scoped_timer{"loader.footpath.hubs"};
    build_hubs(tt, walk_hub_in, walk_hub_out, walk_hub_time);
  }
  // The pruned edges are exactly the ones the hubs hand out anyway, so this
  // is the same transfer relation with the duplicates left out.
  auto const timer = scoped_timer{"loader.footpath.prune"};
  prune_hub_covered_footpaths(tt);
}

}  // namespace nigiri::loader
