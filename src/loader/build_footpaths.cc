#include "nigiri/loader/build_footpaths.h"

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
void build_hubs(timetable& tt) {
  auto const n = tt.n_locations();

  // Which members a slower transfer starts at / leads to. Taken from the
  // rules rather than from the footpaths, because the footpath layer is not
  // always the loader's to write - and because the loader decided what to
  // leave out from exactly this, so both sides have to read the same source.
  auto slow_from = hash_set<location_idx_t>{};
  auto slow_to = hash_set<location_idx_t>{};
  {
    auto const n_rules = std::min(
        static_cast<std::size_t>(tt.locations_.transfer_rule_fps_.size()),
        static_cast<std::size_t>(n));
    for (auto l = location_idx_t{0U}; l != location_idx_t{n_rules}; ++l) {
      auto const base = base_of(tt, l);
      for (auto const fp : tt.locations_.transfer_rule_fps_[l]) {
        if (base_of(tt, fp.target()) == base &&
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
  for (auto const [hub_ingress, hub_egress, hub_time] :
       utl::zip(tt.locations_.hub_in_[kDefaultProfile],
                tt.locations_.hub_out_[kDefaultProfile],
                tt.locations_.hub_time_[kDefaultProfile])) {
    add_hub(hub_ingress, hub_egress, hub_time);
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
    if (d == kNoTransfer) {
      continue;  // nobody changes here, so nothing is derived at "d"
    }
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

  tt.locations_.n_rule_hubs_ = static_cast<std::uint32_t>(in.size());

  tt.locations_.hub_in_[kDefaultProfile] = std::move(in);
  tt.locations_.hub_out_[kDefaultProfile] = std::move(out);
  tt.locations_.hub_time_[kDefaultProfile] = std::move(time);
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
  auto const key = [](location_idx_t const a, location_idx_t const b) {
    return (static_cast<std::uint64_t>(to_idx(a)) << 32) | to_idx(b);
  };

  auto derived = hash_map<std::uint64_t, int>{};
  for (auto h = hub_idx_t{0U};
       h != hub_idx_t{tt.locations_.hub_time_[p].size()}; ++h) {
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
  link_tracks_with_station(tt);
  link_nearby_stations(tt);
  add_equivalence_footpaths(tt, opt.max_footpath_length_);

  if (opt.merge_dupes_intra_src_ || opt.merge_dupes_inter_src_) {
    merge_duplicates(tt, opt.merge_threshold_, opt.merge_dupes_intra_src_,
                     opt.merge_dupes_inter_src_, opt.merge_stats_dir_,
                     opt.src_tags_);
  }

  copy_footpaths_to_generated_children(tt);
  // Whoever computes the footpath layer merges the rules in - keeping them
  // out of the copies above saves storing every rule cell twice more (as an
  // outgoing footpath and again in the incoming mirror).
  apply_transfer_rules(tt);
  write_footpaths(tt, opt.adjust_footpaths_);
  build_hubs(tt);
  // The pruned edges are exactly the ones the hubs hand out anyway, so this
  // is the same transfer relation with the duplicates left out.
  prune_hub_covered_footpaths(tt);
}

}  // namespace nigiri::loader
