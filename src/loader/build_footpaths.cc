#include "nigiri/loader/build_footpaths.h"

#include <cstdlib>
#include <map>
#include <optional>
#include <string>
#include <span>
#include <vector>

#include "geo/latlng.h"

#include "utl/erase_duplicates.h"
#include "utl/helpers/algorithm.h"
#include "utl/zip.h"

#include "nigiri/loader/link_nearby_stations.h"
#include "nigiri/loader/merge_duplicates.h"
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

bool is_generated(location_type const t) {
  return t == location_type::kGeneratedTrack || t == location_type::kVirt;
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
      auto const duration = duration_t{
          std::max(2, static_cast<int>(std::ceil((dist / kWalkSpeed) / 60.0)))};

      if (duration > max_duration) {
        continue;
      }

      add_if_not_exists(tt.locations_.preprocessing_footpaths_out_[l],
                        {eq, duration});
      add_if_not_exists(tt.locations_.preprocessing_footpaths_out_[eq],
                        {l, duration});
    }
  }
}

// Generated children (HRD track locations, virtual locations) have no
// position of their own: they sit exactly where their parent sits. Every
// transfer of the parent is therefore a transfer of the child at the same
// duration - be it a beeline or a transfers.txt row - so they are copied over
// instead of being recomputed per child. Self loops must not be copied: for a
// virtual location the copy would land between it and its own stop and
// undercut the rule matrix (materialized deviations plus the defaults derived
// by that stop's hubs).
void copy_footpaths_to_generated_children(timetable& tt) {
  auto fp_out = mutable_fws_multimap<location_idx_t, footpath>{};
  for (auto l = location_idx_t{0U};
       l != tt.locations_.preprocessing_footpaths_out_.size(); ++l) {
    for (auto const& fp : tt.locations_.preprocessing_footpaths_out_[l]) {
      if (fp.target() == l) {
        continue;  // a self loop is not a transfer, and propagating it would
                   // connect the children at an unrelated duration
      }
      for (auto const& neighbor_child : tt.locations_.children_[fp.target()]) {
        if (!is_generated(tt.locations_.types_[neighbor_child])) {
          continue;
        }
        fp_out[l].emplace_back(neighbor_child, fp.duration());
        for (auto const& child : tt.locations_.children_[l]) {
          if (is_generated(tt.locations_.types_[child])) {
            fp_out[child].emplace_back(neighbor_child, fp.duration());
          }
        }
      }

      for (auto const& child : tt.locations_.children_[l]) {
        if (is_generated(tt.locations_.types_[child])) {
          fp_out[child].emplace_back(fp.target(), fp.duration());
        }
      }
    }
  }

  for (auto l = location_idx_t{0U};
       l != tt.locations_.preprocessing_footpaths_out_.size(); ++l) {
    for (auto const& fp : fp_out[l]) {
      tt.locations_.preprocessing_footpaths_out_[l].emplace_back(fp);
    }
  }
}





// Sorted rule targets per location and the bases they sit at. A walk asks
// only "does this member have a rule into that stop", which is a binary
// search in the second list; the first is needed for the few that answer yes.
struct rule_index {
  explicit rule_index(timetable& tt) : tt_{tt} {
    auto const n = static_cast<std::size_t>(cista::to_idx(tt.n_locations()));
    auto const n_rules =
        std::min(static_cast<std::size_t>(tt.locations_.transfer_rule_fps_.size()), n);
    bases_.resize(n);
    for (auto l = location_idx_t{0U}; l != location_idx_t{n_rules}; ++l) {
      utl::sort(tt.locations_.transfer_rule_fps_[l],
                [](footpath const a, footpath const b) {
                  return a.target() < b.target();
                });
      auto& b = bases_[to_idx(l)];
      for (auto const fp : tt.locations_.transfer_rule_fps_[l]) {
        b.push_back(base_of(fp.target()));
      }
      utl::erase_duplicates(b);
    }
  }

  location_idx_t base_of(location_idx_t const l) const {
    return tt_.locations_.types_[l] == location_type::kVirt
               ? tt_.locations_.parents_[l]
               : l;
  }

  // does `from` have any rule into `stop` or one of its virtual locations?
  bool any_at(location_idx_t const from, location_idx_t const stop) const {
    auto const& b = bases_[to_idx(from)];
    return std::binary_search(begin(b), end(b), stop);
  }

  bool ruled(location_idx_t const from, location_idx_t const to) const {
    if (to_idx(from) >= tt_.locations_.transfer_rule_fps_.size()) {
      return false;
    }
    auto const b = tt_.locations_.transfer_rule_fps_[from];
    auto const it = std::lower_bound(
        begin(b), end(b), to,
        [](footpath const fp, location_idx_t const t) { return fp.target() < t; });
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

// The walking transfers of a stop with virtual locations are carried by hubs:
// one per footpath, ingress = the stop and its virtual locations, egress = the
// target and its own. A hub hands its duration to every pair of the two lists,
// so pairs a rule speaks about have to be kept out of it - the same split the
// rule hubs use: sources with an exception move to a second hub that reaches
// only the targets no exception names, and what neither covers is written as
// an ordinary footpath. This first pass writes those, before the footpath
// lists are finalised; build_hubs then emits the hubs themselves.
// The walks of the stops with virtual locations, as hubs. One hub stands for
// every pair of its two lists at one weight, so the footpaths of a stop that
// share a duration share a hub: the ingress is the same either way and the
// egress is their union. Pairs a rule speaks about must stay out of the lists
// - the same split the rule hubs use - and rectangles too small to pay for a
// hub become ordinary footpaths instead. This runs before the footpath lists
// are written, so the durations are adjusted here exactly as write_footpaths
// would, and it is the only place that decides.
constexpr auto const kMinHubPairs = std::size_t{4U};

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
           utl::any_of(tt.locations_.transfer_rule_fps_[l],
                       [&](footpath const r) { return r.target() == fp.target(); });
  };

  auto extra = mutable_fws_multimap<location_idx_t, footpath>{};
  auto members = std::vector<location_idx_t>{};
  auto targets = std::vector<location_idx_t>{};
  auto unrestricted = std::vector<location_idx_t>{};
  auto restricted = std::vector<location_idx_t>{};
  auto egress = std::vector<location_idx_t>{};
  auto named = hash_set<location_idx_t>{};

  auto& w_in = walk_hub_in;
  auto& w_out = walk_hub_out;
  auto& w_time = walk_hub_time;
  auto const emit = [&](std::vector<location_idx_t> const& ingress,
                        std::vector<location_idx_t> const& eg,
                        duration_t const d) {
    if (ingress.empty() || eg.empty()) {
      return;
    }
    if (ingress.size() * eg.size() <= kMinHubPairs) {
      for (auto const m : ingress) {  // cheaper as footpaths than as a hub
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
      // Targets no rule speaks about can share one hub, whatever stop they
      // belong to: same ingress, same weight. A target some member has a rule
      // into keeps its own pair of hubs - merging it would make every member
      // of the group restricted at every stop of the group, and the pairs that
      // fall out of the split would have to be written one by one.
      egress.clear();
      for (auto const t_stop : stops) {
        collect_members(tt, t_stop, targets);

        restricted.clear();
        for (auto const m : members) {
          if (idx.any_at(m, t_stop)) {
            restricted.push_back(m);
          }
        }
        if (restricted.empty()) {
          egress.insert(end(egress), begin(targets), end(targets));
          continue;
        }

        unrestricted.clear();
        for (auto const m : members) {
          if (!idx.any_at(m, t_stop)) {
            unrestricted.push_back(m);
          }
        }
        named.clear();
        for (auto const m : restricted) {
          for (auto const r : tt.locations_.transfer_rule_fps_[m]) {
            if (idx.base_of(r.target()) == t_stop) {
              named.insert(r.target());
            }
          }
        }
        if (!unrestricted.empty()) {
          emit(unrestricted, targets, d);
        }
        auto clean = std::vector<location_idx_t>{};
        for (auto const t : targets) {
          if (!named.contains(t)) {
            clean.push_back(t);
          }
        }
        if (!clean.empty()) {
          emit(restricted, clean, d);
        }
        for (auto const m : restricted) {
          for (auto const t : named) {
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



// The rule cells the loader had to write out, one footpath each, are what the
// routing walks per marked location - on Switzerland 1.6 million of them. A
// hub can carry them too, but only where they form a complete rectangle of
// one duration: a hub hands its weight to every pair of its lists, so a hub
// over a partial rectangle would invent cells the data does not state. Sources
// that name exactly the same targets at exactly the same duration are such a
// rectangle, so they are grouped and handed to add_hub; the pairs are then
// left out of the footpath lists.
hash_set<std::pair<location_idx_t, location_idx_t>> hub_ify_rule_cells(
    timetable& tt) {
  struct group {
    std::vector<location_idx_t> targets_, sources_;
  };
  auto by_key = hash_map<cista::hash_t, std::vector<group>>{};

  auto targets = std::vector<location_idx_t>{};
  auto const n_rules = std::min(
      static_cast<std::size_t>(tt.locations_.transfer_rule_fps_.size()),
      static_cast<std::size_t>(cista::to_idx(tt.n_locations())));
  for (auto m = location_idx_t{0U}; m != location_idx_t{n_rules}; ++m) {
    auto by_duration = std::map<duration_t, std::vector<location_idx_t>>{};
    for (auto const fp : tt.locations_.transfer_rule_fps_[m]) {
      by_duration[fp.duration()].push_back(fp.target());
    }
    for (auto& [d, t] : by_duration) {
      targets = t;
      utl::sort(targets);
      auto h = cista::hash_combine(cista::BASE_HASH,
                                   static_cast<std::uint64_t>(d.count()));
      for (auto const x : targets) {
        h = cista::hash_combine(h, to_idx(x));
      }
      auto& candidates = by_key[h];
      auto found = false;
      for (auto& g : candidates) {
        if (g.targets_ == targets) {
          g.sources_.push_back(m);
          found = true;
          break;
        }
      }
      if (!found) {
        candidates.push_back(group{.targets_ = targets, .sources_ = {m}});
      }
    }
  }

  // A rule does not only state its own edge, it overrides any walk between the
  // same pair. A hub can carry the first but not the second, so a cell whose
  // pair already has a footpath stays a footpath too - otherwise the beeline
  // survives underneath the hub and undercuts the rule.
  auto fp_targets = hash_map<location_idx_t, std::vector<location_idx_t>>{};
  for (auto m = location_idx_t{0U}; m != location_idx_t{n_rules}; ++m) {
    if (tt.locations_.transfer_rule_fps_[m].empty()) {
      continue;
    }
    auto& t = fp_targets[m];
    for (auto const& fp : tt.locations_.preprocessing_footpaths_out_[m]) {
      t.push_back(fp.target());
    }
    utl::sort(t);
  }
  auto const walks_there = [&](location_idx_t const m, location_idx_t const t) {
    auto const it = fp_targets.find(m);
    return it != end(fp_targets) &&
           std::binary_search(begin(it->second), end(it->second), t);
  };

  auto hubbed = hash_set<std::pair<location_idx_t, location_idx_t>>{};
  for (auto const& [h, candidates] : by_key) {
    for (auto const& g : candidates) {
      auto const n_pairs = g.sources_.size() * g.targets_.size();
      if (n_pairs <= kMinHubPairs ||
          n_pairs <= g.sources_.size() + g.targets_.size()) {
        continue;  // cheaper as the footpaths it already is
      }
      // every source states every target at this duration, and the duration is
      // the one the hub hands out, so the hub states exactly these cells
      auto const d = duration_t{[&]() {
        for (auto const fp : tt.locations_.transfer_rule_fps_[g.sources_[0]]) {
          if (fp.target() == g.targets_[0]) {
            return fp.duration().count();
          }
        }
        return duration_t::rep{0};
      }()};
      tt.locations_.hub_in_[kDefaultProfile].emplace_back(g.sources_);
      tt.locations_.hub_out_[kDefaultProfile].emplace_back(g.targets_);
      tt.locations_.hub_time_[kDefaultProfile].push_back(d);
      for (auto const src : g.sources_) {
        for (auto const t : g.targets_) {
          if (!walks_there(src, t)) {
            hubbed.emplace(src, t);
          }
        }
      }
    }
  }
  if (std::getenv("NIGIRI_HUB_STATS") != nullptr) {
    auto total = std::uint64_t{0};
    for (auto m = location_idx_t{0U}; m != location_idx_t{n_rules}; ++m) {
      total += tt.locations_.transfer_rule_fps_[m].size();
    }
    log(log_lvl::info, "rule.hubs",
        "{} rule cells: {} carried by hubs, {} left as footpaths ({:.1f}%)",
        total, hubbed.size(), total - hubbed.size(),
        100.0 * static_cast<double>(total - hubbed.size()) /
            static_cast<double>(std::max(total, std::uint64_t{1})));
  }
  return hubbed;
}

// Overwrite/insert the directed transfer edges emitted from transfer rules.
// They are authoritative: any generic footpath between the same pair is
// replaced and the duration survives the walk speed adjustment - a rule fixes
// the transfer time, which may be shorter or longer than the walking time.
void apply_transfer_rules(
    timetable& tt,
    hash_set<std::pair<location_idx_t, location_idx_t>> const& hubbed) {
  auto const n = std::min(
      static_cast<std::size_t>(tt.locations_.transfer_rule_fps_.size()),
      static_cast<std::size_t>(tt.n_locations()));
  for (auto l = location_idx_t{0U}; l != location_idx_t{n}; ++l) {
    for (auto const fp : tt.locations_.transfer_rule_fps_[l]) {
      if (hubbed.contains(std::pair{l, fp.target()})) {
        continue;  // a hub states this cell now
      }
      auto bucket = tt.locations_.preprocessing_footpaths_out_[l];
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

    tt.locations_.footpaths_out_[kDefaultProfile].emplace_back(fps);
    for (auto const fp : fps) {
      fps_in[fp.target()].emplace_back(l, fp.duration());
    }
  }

  for (auto l = location_idx_t{0U}; l != tt.n_locations(); ++l) {
    tt.locations_.footpaths_in_[kDefaultProfile].emplace_back(fps_in[l]);
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
void build_hubs(timetable& tt,
                vecvec<hub_idx_t, location_idx_t> const& walk_hub_in,
                vecvec<hub_idx_t, location_idx_t> const& walk_hub_out,
                vector_map<hub_idx_t, duration_t> const& walk_hub_time) {
  auto const n = tt.n_locations();

  // Which members a slower transfer starts at / leads to. Taken from the
  // rules rather than from the footpaths, because the footpath layer is not
  // always the loader's to write - and because the loader decided what to
  // leave out from exactly this, so both sides have to read the same source.
  auto slow_from = hash_set<location_idx_t>{};
  auto slow_to = hash_set<location_idx_t>{};
  {
    auto const base_of = [&](location_idx_t const l) {
      return tt.locations_.types_[l] == location_type::kVirt
                 ? tt.locations_.parents_[l]
                 : l;
    };
    auto const n_rules = std::min(
        static_cast<std::size_t>(tt.locations_.transfer_rule_fps_.size()),
        static_cast<std::size_t>(n));
    for (auto l = location_idx_t{0U}; l != location_idx_t{n_rules}; ++l) {
      auto const base = base_of(l);
      for (auto const fp : tt.locations_.transfer_rule_fps_[l]) {
        if (base_of(fp.target()) == base &&
            fp.duration() > tt.locations_.transfer_time_[base]) {
          slow_from.insert(l);
          slow_to.insert(fp.target());
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
  for (auto const [in, out, d] :
       utl::zip(tt.locations_.hub_in_[kDefaultProfile],
                tt.locations_.hub_out_[kDefaultProfile],
                tt.locations_.hub_time_[kDefaultProfile])) {
    add_hub(in, out, d);
  }

  auto has_virts = std::vector<bool>(n, false);
  for (auto l = location_idx_t{0U}; l != location_idx_t{n}; ++l) {
    if (tt.locations_.types_[l] == location_type::kVirt) {
      has_virts[to_idx(tt.locations_.parents_[l])] = true;
    }
  }

  auto members = std::vector<location_idx_t>{};
  auto unrestricted_in = std::vector<location_idx_t>{};
  auto unrestricted_out = std::vector<location_idx_t>{};
  auto restricted_in = std::vector<location_idx_t>{};
  auto restricted_out = std::vector<location_idx_t>{};
  for (auto base = location_idx_t{0U}; base != location_idx_t{n}; ++base) {
    if (!has_virts[to_idx(base)]) {
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
    auto const is_slow = [&](location_idx_t const m) {
      return m != base && tt.locations_.transfer_time_[m] > d;
    };

    unrestricted_in.clear();
    unrestricted_out.clear();
    restricted_in.clear();
    restricted_out.clear();
    for (auto const m : members) {
      if (!is_slow(m) && !slow_from.contains(m)) {
        unrestricted_in.emplace_back(m);
      } else if (!is_slow(m)) {
        restricted_in.emplace_back(m);
      }
      unrestricted_out.emplace_back(m);
      if (!slow_to.contains(m)) {
        restricted_out.emplace_back(m);
      }
    }

    add_hub(unrestricted_in, unrestricted_out, d);
    add_hub(restricted_in, restricted_out, d);
  }

  // the walks come last: everything before them is rule-derived and therefore
  // the same in every profile, so a profile with its own walking layer can
  // take that prefix and append its own
  tt.locations_.n_rule_hubs_ = static_cast<std::uint32_t>(in.size());
  for (auto h = hub_idx_t{0U}; h != hub_idx_t{walk_hub_in.size()}; ++h) {
    auto const w_i = walk_hub_in[h];
    auto const w_o = walk_hub_out[h];
    add_hub({w_i.data(), w_i.size()}, {w_o.data(), w_o.size()},
            walk_hub_time[h]);
  }

  if (std::getenv("NIGIRI_HUB_STATS") != nullptr) {
    auto in_e = std::uint64_t{0}, out_e = std::uint64_t{0}, pairs = std::uint64_t{0};
    auto max_at = std::size_t{0};
    for (auto h = hub_idx_t{0U}; h != hub_idx_t{in.size()}; ++h) {
      in_e += in[h].size();
      out_e += out[h].size();
      pairs += static_cast<std::uint64_t>(in[h].size()) * out[h].size();
    }
    for (auto l = location_idx_t{0U}; l != location_idx_t{n}; ++l) {
      max_at = std::max(max_at, in_by_loc[l].size());
    }
    log(log_lvl::info, "hub.stats",
        "hubs={} in={} out={} pairs={} worst location in {} hubs", in.size(),
        in_e, out_e, pairs, max_at);
  }

  tt.locations_.hub_in_[kDefaultProfile] = std::move(in);
  tt.locations_.hub_out_[kDefaultProfile] = std::move(out);
  tt.locations_.hub_time_[kDefaultProfile] = std::move(time);
  for (auto l = location_idx_t{0U}; l != location_idx_t{n}; ++l) {
    tt.locations_.hub_in_by_loc_[kDefaultProfile].emplace_back(in_by_loc[l]);
    tt.locations_.hub_out_by_loc_[kDefaultProfile].emplace_back(out_by_loc[l]);
  }
}


// Temporary (NIGIRI_VIRT_MERGE): how many virtual locations are behaviourally
// the same node? Two can only be one if every transfer they take part in
// matches - the cells written for them in both directions, and the hubs they
// sit in, since two members with identical cells can still differ by being in
// an unrestricted and a restricted hub. References to themselves are
// normalised so that a pair differing only in naming itself still matches.
void measure_virt_merge(timetable const& tt) {
  auto const n = cista::to_idx(tt.n_locations());
  auto in_cells = std::vector<std::vector<std::pair<location_idx_t, duration_t>>>(n);
  auto const n_rules = std::min(
      static_cast<std::size_t>(tt.locations_.transfer_rule_fps_.size()),
      static_cast<std::size_t>(n));
  for (auto m = location_idx_t{0U}; m != location_idx_t{n_rules}; ++m) {
    for (auto const fp : tt.locations_.transfer_rule_fps_[m]) {
      in_cells[to_idx(fp.target())].emplace_back(m, fp.duration());
    }
  }

  auto by_base = std::map<location_idx_t, std::vector<location_idx_t>>{};
  for (auto l = location_idx_t{0U}; l != tt.n_locations(); ++l) {
    if (tt.locations_.types_[l] == location_type::kVirt) {
      by_base[tt.locations_.parents_[l]].push_back(l);
    }
  }

  auto const key_of = [&](location_idx_t const v) {
    // its own change time is an observable of its own: a rule matching one
    // trip on both sides lands here, not in the cells
    auto h = cista::hash_combine(
        cista::BASE_HASH,
        static_cast<std::uint64_t>(tt.locations_.transfer_time_[v].count()));
    auto out = std::vector<std::pair<std::uint32_t, std::int32_t>>{};
    if (to_idx(v) < n_rules) {
      for (auto const fp : tt.locations_.transfer_rule_fps_[v]) {
        out.emplace_back(fp.target() == v ? 0xFFFFFFFFU : to_idx(fp.target()),
                         fp.duration().count());
      }
    }
    utl::sort(out);
    for (auto const& [t, d] : out) {
      h = cista::hash_combine(cista::hash_combine(h, t),
                              static_cast<std::uint64_t>(d));
    }
    auto in = std::vector<std::pair<std::uint32_t, std::int32_t>>{};
    for (auto const& [src, d] : in_cells[to_idx(v)]) {
      in.emplace_back(src == v ? 0xFFFFFFFFU : to_idx(src), d.count());
    }
    utl::sort(in);
    for (auto const& [t, d] : in) {
      h = cista::hash_combine(cista::hash_combine(h, t),
                              static_cast<std::uint64_t>(d));
    }
    for (auto const x : tt.locations_.hub_in_by_loc_[kDefaultProfile][v]) {
      h = cista::hash_combine(h, to_idx(x) * 2U);
    }
    for (auto const x : tt.locations_.hub_out_by_loc_[kDefaultProfile][v]) {
      h = cista::hash_combine(h, to_idx(x) * 2U + 1U);
    }
    return h;
  };

  auto total = std::size_t{0}, classes = std::size_t{0};
  auto biggest = std::vector<std::tuple<std::size_t, std::size_t, location_idx_t>>{};
  for (auto const& [base, virts] : by_base) {
    auto keys = hash_set<cista::hash_t>{};
    for (auto const v : virts) {
      keys.insert(key_of(v));
    }
    total += virts.size();
    classes += keys.size();
    biggest.emplace_back(virts.size(), keys.size(), base);
  }
  utl::sort(biggest, [](auto const& a, auto const& b) {
    return std::get<0>(a) > std::get<0>(b);
  });
  auto top = std::string{};
  for (auto i = 0U; i != std::min<std::size_t>(8U, biggest.size()); ++i) {
    auto const& [nv, nc, base] = biggest[i];
    top += fmt::format("{}->{} ", nv, nc);
  }
  log(log_lvl::info, "virt.merge",
      "{} virtual locations in {} bases: {} behaviour classes ({:.1f}% "
      "mergeable) | biggest bases: {}",
      total, by_base.size(), classes,
      100.0 * static_cast<double>(total - classes) /
          static_cast<double>(std::max(total, std::size_t{1})),
      top);
}


// Temporary (NIGIRI_EDGE_DUMP=<from-id-substr>,<to-id-substr>): every edge the
// routing can take between the members of two stops, and where it comes from.
void dump_edges(timetable const& tt, std::string const& spec) {
  auto const comma = spec.find(',');
  auto const from_s = spec.substr(0, comma), to_s = spec.substr(comma + 1);
  auto find = [&](std::string const& needle) {
    for (auto l = location_idx_t{0U}; l != tt.n_locations(); ++l) {
      if (tt.locations_.ids_[l].view().find(needle) != std::string_view::npos) {
        return l;
      }
    }
    return location_idx_t::invalid();
  };
  auto const a = find(from_s), b = find(to_s);
  if (a == location_idx_t::invalid() || b == location_idx_t::invalid()) {
    log(log_lvl::info, "edge.dump", "stop not found");
    return;
  }
  auto const members = [&](location_idx_t const l) {
    auto v = std::vector<location_idx_t>{l};
    for (auto const c : tt.locations_.children_[l]) {
      if (tt.locations_.types_[c] == location_type::kVirt) {
        v.push_back(c);
      }
    }
    return v;
  };
  auto const ma = members(a), mb = members(b);
  log(log_lvl::info, "edge.dump", "{} ({} members) -> {} ({} members)",
      tt.locations_.ids_[a].view(), ma.size(), tt.locations_.ids_[b].view(),
      mb.size());

  for (auto const m : ma) {
    for (auto const t : mb) {
      auto rule = duration_t{-1};
      if (to_idx(m) < tt.locations_.transfer_rule_fps_.size()) {
        for (auto const fp : tt.locations_.transfer_rule_fps_[m]) {
          if (fp.target() == t) {
            rule = fp.duration();
          }
        }
      }
      auto fp_d = duration_t{-1};
      for (auto const& fp : tt.locations_.footpaths_out_[kDefaultProfile][m]) {
        if (fp.target() == t) {
          fp_d = fp.duration();
        }
      }
      auto hub_d = duration_t{-1};
      auto hub_id = -1;
      for (auto const h : tt.locations_.hub_in_by_loc_[kDefaultProfile][m]) {
        for (auto const x : tt.locations_.hub_out_[kDefaultProfile][h]) {
          if (x == t &&
              (hub_d.count() < 0 || tt.locations_.hub_time_[kDefaultProfile][h] < hub_d)) {
            hub_d = tt.locations_.hub_time_[kDefaultProfile][h];
            hub_id = static_cast<int>(to_idx(h));
          }
        }
      }
      if (rule.count() < 0 && fp_d.count() < 0 && hub_d.count() < 0) {
        continue;
      }
      // only the interesting ones: something reaches it faster than a rule says
      auto const best = std::min(fp_d.count() < 0 ? 9999 : fp_d.count(),
                                 hub_d.count() < 0 ? 9999 : hub_d.count());
      if (rule.count() > 0 && best < rule.count()) {
        log(log_lvl::info, "edge.dump",
            "  UNDERCUT {} -> {}: rule={} but footpath={} hub={} (hub #{})",
            to_idx(m), to_idx(t), rule.count(), fp_d.count(), hub_d.count(),
            hub_id);
      }
    }
  }
}

// Reference mode: replaces every hub by the pairs it stands for. A hub hands
// its weight to each pair without overriding anything - the routing takes the
// minimum - so the expansion is min-merged into the footpaths rather than
// written as rule cells, which would be authoritative. The result routes like
// the hubs it replaces and lets the compressed timetable be checked against an
// uncompressed one.
void expand_hubs_into_footpaths(timetable& tt) {
  auto const n = static_cast<std::size_t>(cista::to_idx(tt.n_locations()));
  auto out = std::vector<std::vector<footpath>>(n);
  for (auto l = location_idx_t{0U}; l != tt.n_locations(); ++l) {
    for (auto const fp : tt.locations_.footpaths_out_[kDefaultProfile][l]) {
      out[to_idx(l)].push_back(fp);
    }
  }

  auto n_pairs = std::size_t{0U};
  for (auto h = hub_idx_t{0U};
       h != hub_idx_t{tt.locations_.hub_time_[kDefaultProfile].size()}; ++h) {
    auto const d = tt.locations_.hub_time_[kDefaultProfile][h];
    for (auto const u : tt.locations_.hub_in_[kDefaultProfile][h]) {
      for (auto const v : tt.locations_.hub_out_[kDefaultProfile][h]) {
        if (u != v) {
          out[to_idx(u)].push_back(footpath{v, d});
          ++n_pairs;
        }
      }
    }
  }

  auto fps_out = vecvec<location_idx_t, footpath>{};
  auto fps_in = mutable_fws_multimap<location_idx_t, footpath>{};
  for (auto l = location_idx_t{0U}; l != tt.n_locations(); ++l) {
    auto& fps = out[to_idx(l)];
    utl::erase_duplicates(
        fps,
        [](footpath const a, footpath const b) {
          return std::tie(a.target_, a.duration_) <
                 std::tie(b.target_, b.duration_);
        },
        [](footpath const a, footpath const b) {
          return a.target_ == b.target_;
        });  // also sorts; keeps the shortest duration per target
    fps_out.emplace_back(fps);
    for (auto const fp : fps) {
      fps_in[fp.target()].emplace_back(l, fp.duration());
    }
  }
  tt.locations_.footpaths_out_[kDefaultProfile] = std::move(fps_out);
  tt.locations_.footpaths_in_[kDefaultProfile].clear();
  for (auto l = location_idx_t{0U}; l != tt.n_locations(); ++l) {
    tt.locations_.footpaths_in_[kDefaultProfile].emplace_back(fps_in[l]);
  }

  // The hubs stay. Dropping them is not the same timetable: the routing
  // reaches a hub-derived pair through expand_hubs, which delivers labels the
  // footpath phase does not produce even when every pair it stands for is
  // stored - a hub-free copy loses journeys and cannot be reconstructed
  // (measured: 259 of 2000 queries). What this mode is for is checking that
  // the hubs derive exactly the stored pairs, and that holds with them in
  // place: every derived transfer is now also a footpath, so any pair a hub
  // got wrong would change a journey.

  log(log_lvl::info, "loader.footpath",
      "reference mode: {} hub pairs expanded into footpaths", n_pairs);
}

// The hubs derive most of the foot layer a second time: measured on CH, 3.75M
// of the 4.59M stored edges are exactly the pair a hub already hands out, at
// the same weight. The routing takes the minimum of both, so keeping the
// footpath cannot change an answer - it only makes the footpath phase walk
// five times more edges than it has to. Drop the ones a hub covers at the same
// weight or better and let the hub stand for them (which is what the shared
// path already does at write time, see hub_ify_rule_cells).
void prune_hub_covered_footpaths(timetable& tt) {
  constexpr auto const p = kDefaultProfile;
  auto const n = static_cast<std::size_t>(cista::to_idx(tt.n_locations()));
  auto const key = [](location_idx_t const a, location_idx_t const b) {
    return (static_cast<std::uint64_t>(to_idx(a)) << 32) | to_idx(b);
  };

  auto derived = hash_map<std::uint64_t, int>{};
  for (auto h = hub_idx_t{0U}; h != hub_idx_t{tt.locations_.hub_time_[p].size()};
       ++h) {
    auto const w = static_cast<int>(tt.locations_.hub_time_[p][h].count());
    for (auto const u : tt.locations_.hub_in_[p][h]) {
      for (auto const v : tt.locations_.hub_out_[p][h]) {
        if (u == v) {
          continue;
        }
        auto const [it, ins] = derived.emplace(key(u, v), w);
        if (!ins) {
          it->second = std::min(it->second, w);
        }
      }
    }
  }

  auto out = std::vector<std::vector<footpath>>(n);
  auto n_pruned = std::size_t{0U}, n_kept = std::size_t{0U};
  for (auto l = location_idx_t{0U}; l != tt.n_locations(); ++l) {
    for (auto const fp : tt.locations_.footpaths_out_[p][l]) {
      auto const it = derived.find(key(l, fp.target()));
      if (it != end(derived) &&
          it->second <= static_cast<int>(fp.duration().count())) {
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

void build_footpaths(timetable& tt, finalize_options const opt) {
  // Covers locations created after the last transfers.txt was read (virtual
  // locations, locations from feeds without transfers.txt).
  tt.locations_.sync_base_transfer_time();

  auto walk_hub_in = vecvec<hub_idx_t, location_idx_t>{};
  auto walk_hub_out = vecvec<hub_idx_t, location_idx_t>{};
  auto walk_hub_time = vector_map<hub_idx_t, duration_t>{};
  link_nearby_stations(tt, opt.beeline_footpaths_);
  if (opt.beeline_footpaths_) {
    add_equivalence_footpaths(tt, opt.max_footpath_length_);
  }

  if (opt.merge_dupes_intra_src_ || opt.merge_dupes_inter_src_) {
    for (auto l = location_idx_t{0U}; l != tt.n_locations(); ++l) {
      if (tt.locations_.src_[l] == source_idx_t{source_idx_t::invalid()}) {
        continue;
      }
      for (auto e : tt.locations_.equivalences_[l]) {
        if (tt.locations_.src_[e] == source_idx_t{source_idx_t::invalid()} ||
            (!opt.merge_dupes_intra_src_ &&
             tt.locations_.src_[l] == tt.locations_.src_[e]) ||
            (!opt.merge_dupes_inter_src_ &&
             tt.locations_.src_[l] != tt.locations_.src_[e])) {
          continue;
        }

        find_duplicates(tt, l, e);
      }
    }
  }

  // Reference mode: every transfer is stored as an ordinary footpath - no
  // shared walks, no hubs, no elision (see transfer_rules.cc). Combined with
  // NIGIRI_NO_VIRT_MERGE it is the uncompressed timetable the tricks are
  // supposed to be equivalent to.
  // Reference mode keeps the shared walks: the hubs are what carries a virtual
  // location's walks, and they are expanded into ordinary footpaths at the end
  // (expand_hubs_into_footpaths). Copying to the children instead would miss
  // every pair created after copy_footpaths_to_generated_children ran.
  auto const materialize = std::getenv("NIGIRI_MATERIALIZE") != nullptr;
  auto const no_share = std::getenv("NIGIRI_NO_SHARE") != nullptr;
  auto const no_hubs = std::getenv("NIGIRI_NO_WALK_HUBS") != nullptr;
  // Reference mode keeps the walks on the parent and lets the hubs carry
  // them; NIGIRI_SHARE_CHILD_FP is the A/B switch for that path, which is
  // otherwise only exercised by NIGIRI_MATERIALIZE.
  auto const share =
      !no_share &&
      (std::getenv("NIGIRI_SHARE_CHILD_FP") != nullptr || materialize);
  if (opt.beeline_footpaths_ && !share) {
    copy_footpaths_to_generated_children(tt);
    // Otherwise whoever computes the footpath layer merges them in - keeping
    // them out here saves storing every rule cell twice more (as an outgoing
    // footpath and again in the incoming mirror).
    apply_transfer_rules(tt, {});
  }
  if (share) {
    // The rules themselves are ordinary footpaths - only the walks of the
    // virtual locations are left to the hubs.
    build_walk_hubs(tt, opt.adjust_footpaths_, walk_hub_in, walk_hub_out,
                    walk_hub_time);
    apply_transfer_rules(tt, hub_ify_rule_cells(tt));
  }
  write_footpaths(tt, opt.adjust_footpaths_);
  if (!no_hubs) {
    build_hubs(tt, walk_hub_in, walk_hub_out, walk_hub_time);
  }
  if (materialize) {
    expand_hubs_into_footpaths(tt);
  }
  // On by default: the pruned edges are exactly the ones the hubs hand out
  // anyway, so this is the same transfer relation with the duplicates left
  // out. NIGIRI_NO_PRUNE_HUB_FP keeps them for A/B. Reference mode is exempt -
  // it expands the hubs into footpaths on purpose.
  if (!no_hubs && !materialize &&
      std::getenv("NIGIRI_NO_PRUNE_HUB_FP") == nullptr) {
    prune_hub_covered_footpaths(tt);
  }
  if (auto const* spec = std::getenv("NIGIRI_EDGE_DUMP"); spec != nullptr) {
    dump_edges(tt, spec);
  }
  if (std::getenv("NIGIRI_VIRT_MERGE") != nullptr) {
    measure_virt_merge(tt);
  }
  if (std::getenv("NIGIRI_SIZE_REPORT") != nullptr) {
    fmt::print("locations={}\n", tt.n_locations());
    for (auto p = profile_idx_t{0U}; p != kNProfiles; ++p) {
      fmt::print(
          "prf={} fp_out={}/{} fp_in={}/{} hubs={} hub_in={}/{} "
          "hub_out={}/{} in_by_loc={}/{} out_by_loc={}/{}\n",
          p, tt.locations_.footpaths_out_[p].size(),
          tt.locations_.footpaths_out_[p].data_.size(),
          tt.locations_.footpaths_in_[p].size(),
          tt.locations_.footpaths_in_[p].data_.size(),
          tt.locations_.hub_time_[p].size(), tt.locations_.hub_in_[p].size(),
          tt.locations_.hub_in_[p].data_.size(),
          tt.locations_.hub_out_[p].size(),
          tt.locations_.hub_out_[p].data_.size(),
          tt.locations_.hub_in_by_loc_[p].size(),
          tt.locations_.hub_in_by_loc_[p].data_.size(),
          tt.locations_.hub_out_by_loc_[p].size(),
          tt.locations_.hub_out_by_loc_[p].data_.size());
    }
  }
}

}  // namespace nigiri::loader
