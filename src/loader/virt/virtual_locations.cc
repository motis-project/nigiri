#include "nigiri/loader/virtual_locations.h"

#include <algorithm>
#include <optional>
#include <vector>

#include "utl/verify.h"

#include "nigiri/loader/build_footpaths.h"
#include "nigiri/loader/import_context.h"
#include "nigiri/loader/register.h"
#include "nigiri/loader/virt/footpaths.h"
#include "nigiri/loader/virt/merge_locations.h"
#include "nigiri/loader/virt/rewrite_routes.h"
#include "nigiri/loader/virt/route_stop_pairs.h"
#include "nigiri/loader/virt/signatures.h"
#include "nigiri/loader/virt/split_routes.h"
#include "nigiri/loader/virt/transfer_durations.h"
#include "nigiri/footpath.h"
#include "nigiri/logging.h"
#include "nigiri/shapes_storage.h"
#include "nigiri/stop.h"
#include "nigiri/timetable.h"

namespace nigiri::loader {

using stop_seq_t = std::vector<stop::value_type>;

duration_t to_duration(duration_t const d) {
  return d >= kNever ? footpath::kBan : d;
}

vector_map<route_stop_idx_t, location_idx_t> add_virts(
    timetable& tt,
    import_context& ctx,
    mutable_fws_multimap<location_idx_t, footpath> const& walks,
    rule_index const& stop_rules,
    route_stop_pairs const& pairs,
    vector_map<route_stop_idx_t, route_stop_idx_t> const& route_stop_location) {
  auto virts = vector_map<route_stop_idx_t, location_idx_t>(
      pairs.route_stops_.size(), location_idx_t::invalid());
  auto n_virts = std::size_t{0U};
  for (auto s = route_stop_idx_t{0U}; s != pairs.route_stops_.size(); ++s) {
    if (route_stop_location[s] != s) {
      continue;
    }
    auto const base = pairs.route_stops_[s].base_;
    if (pairs.is_base(s)) {
      virts[s] = base;
      continue;
    }
    auto const fallback = get_walk(tt, walks, stop_rules, base, base);
    auto const self = pairs.pair_of_.find(route_stop_pair{s, s});
    auto const d = self == end(pairs.pair_of_)
                       ? std::optional{fallback}
                       : pick_duration(pairs.windows_[self->second], fallback);
    utl::verify(d.has_value(), "virtual location without transfer time");
    auto l = location{};
    l.src_ = tt.locations_.src_[base];
    l.pos_ = tt.locations_.coordinates_[base];
    l.type_ = location_type::kVirt;
    l.parent_ = base;
    l.transfer_time_ = to_duration(*d);
    virts[s] = register_location(tt, ctx, l);
    ctx.children_[base].push_back(virts[s]);
    ++n_virts;
  }
  if (n_virts != 0U) {
    tt.locations_.children_ = make_sparse_vecvec(ctx.children_);
  }
  log(log_lvl::info, "loader.virtual_locations", "{} virtual locations",
      n_virts);
  return virts;
}

void add_footpaths(
    timetable const& tt,
    hash_map<transfer_pair, transfer_durations> const& durations_of,
    route_stop_pairs const& pairs,
    vector_map<route_stop_idx_t, route_stop_idx_t> const& route_stop_location,
    vector_map<route_stop_idx_t, location_idx_t> const& virts,
    mutable_fws_multimap<location_idx_t, footpath>& durations) {
  durations[location_idx_t{tt.n_locations() - 1U}];
  auto n_footpaths = std::size_t{0U};
  for (auto p = route_stop_pair_idx_t{0U}; p != pairs.pairs_.size(); ++p) {
    auto const [a, c] = pairs.pairs_[p];
    if (a == c || route_stop_location[a] != a || route_stop_location[c] != c ||
        (pairs.is_base(a) && pairs.is_base(c))) {
      continue;
    }
    auto const fallback = durations_of
                              .at(transfer_pair{pairs.route_stops_[a].base_,
                                                pairs.route_stops_[c].base_})
                              .fallback_;
    auto const d = pick_duration(pairs.windows_[p], fallback);
    utl::verify(d.has_value(), "location pair without duration");
    if (*d == fallback) {
      continue;
    }
    durations[virts[a]].push_back(footpath{virts[c], to_duration(*d)});
    ++n_footpaths;
  }
  log(log_lvl::info, "loader.virtual_locations", "{} footpaths", n_footpaths);
}

void write_routes(
    timetable& tt,
    import_context& ctx,
    signatures const& sigs,
    route_stop_pairs const& pairs,
    transport_routes const& routes,
    vector_map<route_stop_idx_t, route_stop_idx_t> const& route_stop_location,
    vector_map<route_stop_idx_t, location_idx_t> const& virts,
    shapes_storage* shapes) {
  auto const n_routes = tt.n_routes();
  auto route_transports = vecvec<route_idx_t, transport_idx_t>{};
  auto transports = std::vector<transport_idx_t>{};
  for (auto q = route_idx_t{0U}; q != routes.base_route_.size(); ++q) {
    transports.clear();
    for (auto const t : tt.route_transport_ranges_[routes.base_route_[q]]) {
      if (routes.route_[t] == q) {
        transports.push_back(t);
      }
    }
    route_transports.emplace_back(transports);
  }
  auto seq = stop_seq_t{};
  auto const get_seq = [&](route_idx_t const q) -> stop_seq_t const& {
    auto const r = routes.base_route_[q];
    auto const base_seq = tt.route_location_seq_[r];
    seq.assign(base_seq.begin(), base_seq.end());
    for (auto i = stop_idx_t{0U}; i != seq.size(); ++i) {
      auto const l = stop{seq[i]}.location_idx();
      auto s = pairs.route_stop_of_.find(route_stop{q, i, l});
      if (s == end(pairs.route_stop_of_) && q != r) {
        auto const b = pairs.route_stop_of_.find(route_stop{r, i, l});
        if (b != end(pairs.route_stop_of_) &&
            !pairs.route_stops_[b->second].is_regular()) {
          s = b;
        }
      }
      if (s != end(pairs.route_stop_of_)) {
        seq[i] = stop{seq[i]}
                     .with_location(virts[route_stop_location[s->second]])
                     .value();
      }
    }
    return seq;
  };
  for (auto q = route_idx_t{n_routes}; q != routes.base_route_.size(); ++q) {
    tt.route_location_seq_.emplace_back(get_seq(q));
  }
  for (auto r = route_idx_t{0U}; r != n_routes; ++r) {
    if (!sigs.has_rules(r)) {
      continue;
    }
    std::ranges::copy(get_seq(r), tt.route_location_seq_[r].begin());
  }
  rewrite_routes(tt, ctx, shapes, routes.base_route_, route_transports);
  log(log_lvl::info, "loader.virtual_locations", "{} routes -> {}", n_routes,
      tt.n_routes());
}

void apply_transfer_rules(
    timetable& tt,
    import_context& ctx,
    shapes_storage* shapes,
    mutable_fws_multimap<location_idx_t, footpath> const& walks) {
  auto& loc = tt.locations_;
  if (tt.transfer_rules_.rules_.empty()) {
    write_default_profile(tt, walks);
    return;
  }

  auto const timer = scoped_timer{"loader.virtual_locations"};
  auto const stop_rules = get_rule_index(tt);
  auto const sigs = get_signatures(tt);
  auto const durations_of = get_transfer_durations(tt, walks, stop_rules, sigs);
  auto pairs = get_route_stop_pairs(tt, sigs, durations_of);
  auto const routes = split_routes(tt, durations_of, pairs);
  auto const route_stop_location = merge_locations(durations_of, pairs);
  auto const n_locations = tt.n_locations();
  auto const virts =
      add_virts(tt, ctx, walks, stop_rules, pairs, route_stop_location);
  if (tt.n_locations() == n_locations) {
    write_default_profile(tt, walks);
    return;
  }
  auto durations = mutable_fws_multimap<location_idx_t, footpath>{};
  add_footpaths(tt, durations_of, pairs, route_stop_location, virts, durations);
  write_routes(tt, ctx, sigs, pairs, routes, route_stop_location, virts,
               shapes);
  write_default_profile(tt, walks, durations);

  // Routed walks of other profiles (computed before) reach the virtual
  // locations as well, importance counts at the root.
  for (auto prf = profile_idx_t{1U}; prf != kNProfiles; ++prf) {
    if (!loc.footpaths_out_[prf].empty()) {
      expand_profile_to_virts(tt, prf);
    }
  }
  tt.location_location_groups_.resize(tt.n_locations());
  if (tt.location_areas_.size() != 0U) {
    tt.location_areas_.resize(tt.n_locations());
  }
}

}  // namespace nigiri::loader
