#include "nigiri/loader/build_footpaths.h"

#include <optional>
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

// Overwrite/insert the directed transfer edges emitted from transfer rules.
// They are authoritative: any generic footpath between the same pair is
// replaced and the duration survives the walk speed adjustment - a rule fixes
// the transfer time, which may be shorter or longer than the walking time.
void apply_transfer_rules(timetable& tt) {
  auto const n = std::min(
      static_cast<std::size_t>(tt.locations_.transfer_rule_fps_.size()),
      static_cast<std::size_t>(tt.n_locations()));
  for (auto l = location_idx_t{0U}; l != location_idx_t{n}; ++l) {
    for (auto const fp : tt.locations_.transfer_rule_fps_[l]) {
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
void build_hubs(timetable& tt) {
  auto const n = tt.n_locations();
  auto const& fps_out = tt.locations_.footpaths_out_[kDefaultProfile];
  auto const& fps_in = tt.locations_.footpaths_in_[kDefaultProfile];

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
       utl::zip(tt.locations_.hub_in_, tt.locations_.hub_out_,
                tt.locations_.hub_time_)) {
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
    auto const is_member = [&](location_idx_t const t) {
      return t == base || (tt.locations_.parents_[t] == base &&
                           tt.locations_.types_[t] == location_type::kVirt);
    };
    auto const is_slow = [&](location_idx_t const m) {
      return m != base && tt.locations_.transfer_time_[m] > d;
    };
    auto const has_slow_transfer = [&](auto const& all_fps,
                                       location_idx_t const m) {
      return utl::any_of(all_fps[m], [&](footpath const fp) {
        return is_member(fp.target()) && fp.duration() > d;
      });
    };

    unrestricted_in.clear();
    unrestricted_out.clear();
    restricted_in.clear();
    restricted_out.clear();
    for (auto const m : members) {
      auto const slow_to = has_slow_transfer(fps_in, m);
      if (!is_slow(m) && !has_slow_transfer(fps_out, m)) {
        unrestricted_in.emplace_back(m);
      } else if (!is_slow(m)) {
        restricted_in.emplace_back(m);
      }
      unrestricted_out.emplace_back(m);
      if (!slow_to) {
        restricted_out.emplace_back(m);
      }
    }

    add_hub(unrestricted_in, unrestricted_out, d);
    add_hub(restricted_in, restricted_out, d);
  }

  tt.locations_.hub_in_ = std::move(in);
  tt.locations_.hub_out_ = std::move(out);
  tt.locations_.hub_time_ = std::move(time);
  for (auto l = location_idx_t{0U}; l != location_idx_t{n}; ++l) {
    tt.locations_.hub_in_by_loc_.emplace_back(in_by_loc[l]);
    tt.locations_.hub_out_by_loc_.emplace_back(out_by_loc[l]);
  }
}

void build_footpaths(timetable& tt, finalize_options const opt) {
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

  if (opt.beeline_footpaths_) {
    copy_footpaths_to_generated_children(tt);
  }
  apply_transfer_rules(tt);
  write_footpaths(tt, opt.adjust_footpaths_);
  build_hubs(tt);
}

}  // namespace nigiri::loader
